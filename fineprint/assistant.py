"""The decision engine: choose the task, ask the model, then check its work.

Everything that can be decided by ordinary code is decided here, deterministically: which
task to run, whether each quote really appears in the document, how findings are ordered,
and which warnings the user sees. The model is only asked for language understanding.
"""

import asyncio
import hashlib
import json
import unicodedata
from array import array
from collections import OrderedDict
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from functools import lru_cache
from typing import NamedTuple

from pydantic import BaseModel

from .llm import Effort, LLMClient
from .prompts import SYSTEM_PROMPT, user_instruction
from .schemas import (
    Analysis,
    Answer,
    AssistRequest,
    AssistResponse,
    Comparison,
    Difference,
    Quote,
    Result,
    Risk,
    Task,
)

DISCLAIMER = (
    "This is information about the text you provided, not legal advice. It doesn't know your "
    "local law, your negotiating position, or anything outside the document, and it can be wrong."
)

_OUTPUT_MODELS: dict[Task, type[Result]] = {
    "analyze": Analysis,
    "ask": Answer,
    "compare": Comparison,
}
# A question is a lookup in the document; analysis and comparison need full reasoning.
_EFFORT: dict[Task, Effort] = {"analyze": "default", "ask": "low", "compare": "default"}
_LABELS = ("A", "B")
_SEVERITY_ORDER = {"high": 0, "medium": 1, "low": 2}
# Money, percentages and section signs carry meaning in legal quotes, so they must match too.
_KEPT_SYMBOLS = frozenset("$£€¥₹%§")


def resolve_task(request: AssistRequest) -> Task:
    """A question is always answered; otherwise two documents are compared, one is analysed."""
    if request.question:
        return "ask"
    if len(request.documents) == 2:
        return "compare"
    return "analyze"


class ResultCache:
    """A bounded, least-recently-used store of finished responses.

    Identical requests (a re-submit, a refresh, the same sample tried twice) are answered
    without another model call, including requests that arrive while the first one is still
    running. It lives in memory for the life of the process only and stores nothing on disk.
    A size of 0 stores nothing.
    """

    def __init__(self, size: int = 128) -> None:
        self.size = size
        self._items: OrderedDict[str, AssistResponse] = OrderedDict()
        self._locks: dict[str, asyncio.Lock] = {}
        self._waiting: dict[str, int] = {}

    @staticmethod
    def key(request: AssistRequest) -> str:
        # Everything that shapes the answer; the task is derived from these fields.
        payload = json.dumps([request.documents, request.question, request.context])
        return hashlib.sha256(payload.encode()).hexdigest()

    def get(self, key: str) -> AssistResponse | None:
        """The stored response, shared rather than copied: nothing modifies it after `put`."""
        if key not in self._items:
            return None
        self._items.move_to_end(key)
        return self._items[key]

    def put(self, key: str, response: AssistResponse) -> None:
        if self.size <= 0:
            return
        self._items[key] = response.model_copy(deep=True)
        self._items.move_to_end(key)
        if len(self._items) > self.size:
            self._items.popitem(last=False)

    @asynccontextmanager
    async def exclusive(self, key: str) -> AsyncIterator[None]:
        """Serialise work on one key, so a duplicate waits for the first result."""
        lock = self._locks.setdefault(key, asyncio.Lock())
        self._waiting[key] = self._waiting.get(key, 0) + 1
        try:
            async with lock:
                yield
        finally:
            self._waiting[key] -= 1
            if not self._waiting[key]:  # last one out removes the lock
                del self._waiting[key], self._locks[key]


async def assist(request: AssistRequest, llm: LLMClient, cache: ResultCache) -> AssistResponse:
    """Run the resolved task and return a result whose quotes are verified against the source.

    Raises:
        LLMError: the model call failed; the error carries a user-facing message. Failures
            are never cached.
    """
    key = ResultCache.key(request)
    async with cache.exclusive(key):
        if (cached := cache.get(key)) is not None:
            return cached
        response = await _run(request, llm)
        cache.put(key, response)
        return response


async def _run(request: AssistRequest, llm: LLMClient) -> AssistResponse:
    task = resolve_task(request)
    documents = dict(zip(_LABELS, request.documents, strict=False))
    result = await llm.complete(
        system=SYSTEM_PROMPT,
        documents=documents,
        request=user_instruction(task, question=request.question, context=request.context),
        output_model=_OUTPUT_MODELS[task],
        effort=_EFFORT[task],
    )
    # Matching is CPU work on up to 300k characters per document: keep it off the event loop.
    unmatched = await asyncio.to_thread(verify_quotes, result, documents)
    if isinstance(result, Analysis):
        result.risks.sort(key=lambda risk: _SEVERITY_ORDER[risk.severity])
    elif isinstance(result, Comparison):
        result.differences.sort(key=lambda difference: _SEVERITY_ORDER[difference.severity])
    return AssistResponse(
        task=task,
        result=result,
        warnings=_warnings(result, unmatched),
        disclaimer=DISCLAIMER,
    )


def _warnings(result: Result, unmatched: int) -> list[str]:
    warnings = []
    if unmatched:
        warnings.append(
            f"Some points ({unmatched}) could not be matched to a passage in your document. "
            "They are marked below; treat them with caution."
        )
    if isinstance(result, (Analysis, Comparison)) and not result.is_legal_document:
        warnings.append(
            "This doesn't look like a legal document, so the results may not mean much."
        )
    if (
        isinstance(result, Answer)
        and result.supported_by_document != "no"
        and not result.quotes  # unmatched quotes are already covered by the warning above
    ):
        warnings.append("The answer cites no passage from your document; treat it with caution.")
    return warnings


class _Source(NamedTuple):
    text: str
    fingerprint: str
    positions: Sequence[int]


@lru_cache(maxsize=4096)
def _fold(char: str) -> str:
    """The comparable form of one character (often empty). A document uses few distinct
    characters, so memoising this skips almost every Unicode normalisation."""
    return "".join(
        folded
        for folded in unicodedata.normalize("NFKD", char).casefold()
        if folded.isalnum() or folded in _KEPT_SYMBOLS
    )


def _fingerprint(text: str) -> tuple[str, Sequence[int]]:
    """Reduce text to the characters that matter for matching, remembering where each came from.

    NFKD plus casefold folds ligatures, accents, full-width forms and case. Keeping only
    letters, digits and a few symbols ignores whitespace, line breaks, hyphenation and quote
    styles, which is exactly where extracted PDF text and the model's copy tend to differ.
    """
    kept: list[str] = []
    positions = array("I")  # 4 bytes per entry instead of a Python int object
    for index, char in enumerate(text):
        folded = _fold(char)
        if not folded:  # whitespace and punctuation: most non-letters
            continue
        kept.append(folded)
        if len(folded) == 1:
            positions.append(index)
        else:  # a ligature such as "ﬁ" folds to several characters
            positions.extend([index] * len(folded))
    return "".join(kept), positions


@lru_cache(maxsize=8)
def _source(text: str) -> _Source:
    """The fingerprinted document, reused across follow-up questions about the same text."""
    return _Source(text, *_fingerprint(text))


def verify_quotes(result: BaseModel, documents: dict[str, str]) -> int:
    """Check every quote against the document it claims to come from.

    A found quote is replaced by the exact source passage (from its first to its last matched
    character, so edge punctuation is trimmed), which lets the UI locate it in the document.
    A quote that cannot be found is blanked, and a risk or difference with no quotes at all
    also counts as unmatched.

    Returns:
        The number of unmatched quotes and evidence-free findings.
    """
    sources = {label: _source(text) for label, text in documents.items()}
    return _count_unmatched(result, sources)


def _count_unmatched(node: object, sources: dict[str, _Source]) -> int:
    if isinstance(node, Quote):
        return 0 if _match(node, sources) else 1
    if isinstance(node, list):
        return sum(_count_unmatched(item, sources) for item in node)
    if isinstance(node, BaseModel):
        missing_evidence = isinstance(node, (Risk, Difference)) and not node.quotes
        return int(missing_evidence) + sum(
            _count_unmatched(getattr(node, field), sources) for field in type(node).model_fields
        )
    return 0


def _match(quote: Quote, sources: dict[str, _Source]) -> bool:
    needle, _ = _fingerprint(quote.text)
    source = sources.get(quote.document)
    # An empty needle would "match" at position 0 and turn the quote into the whole document.
    position = source.fingerprint.find(needle) if needle and source else -1
    if position < 0:
        quote.text = ""
        return False
    start = source.positions[position]
    end = source.positions[position + len(needle) - 1] + 1
    quote.text = source.text[start:end]
    return True
