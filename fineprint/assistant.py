"""The decision engine: choose the task, ask the model, then check its work.

Everything that can be decided by ordinary code is decided here, deterministically: which
task to run, whether each quote really appears in the document, how findings are ordered,
and which warnings the user sees. The model is only asked for language understanding.
"""

import unicodedata
from typing import Any, NamedTuple

from pydantic import BaseModel

from .llm import LLMClient
from .prompts import SYSTEM_PROMPT, user_instruction
from .schemas import (
    Analysis,
    Answer,
    AssistRequest,
    AssistResponse,
    Comparison,
    Difference,
    Quote,
    Risk,
    Task,
)

DISCLAIMER = (
    "This is information about the text you provided, not legal advice. It doesn't know your "
    "local law, your negotiating position, or anything outside the document, and it can be wrong."
)

_OUTPUT_MODELS: dict[Task, type[BaseModel]] = {
    "analyze": Analysis,
    "ask": Answer,
    "compare": Comparison,
}
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


def assist(request: AssistRequest, llm: LLMClient) -> AssistResponse:
    """Run the resolved task and return a result whose quotes are verified against the source.

    Raises:
        LLMError: the model call failed; the error carries a user-facing message.
    """
    task = resolve_task(request)
    result = llm.complete(
        system=SYSTEM_PROMPT,
        content=_build_content(request, task),
        output_model=_OUTPUT_MODELS[task],
    )
    unmatched = verify_quotes(result, dict(zip(_LABELS, request.documents, strict=False)))
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


def _build_content(request: AssistRequest, task: Task) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = [
        {
            "type": "document",
            "source": {"type": "text", "media_type": "text/plain", "data": text},
            "title": f"Document {label}",
        }
        for label, text in zip(_LABELS, request.documents, strict=False)
    ]
    # One cache breakpoint after the documents caches system prompt + schema + documents, so
    # repeated questions about the same document are cheaper and faster.
    blocks[-1]["cache_control"] = {"type": "ephemeral"}
    blocks.append(
        {
            "type": "text",
            "text": user_instruction(task, question=request.question, context=request.context),
        }
    )
    return blocks


def _warnings(result: BaseModel, unmatched: int) -> list[str]:
    warnings = []
    if unmatched:
        warnings.append(
            f"Some points ({unmatched}) could not be matched to a passage in your document. "
            "They are marked below; treat them with caution."
        )
    if isinstance(result, Analysis) and not result.is_legal_document:
        warnings.append(
            "This doesn't look like a legal document, so the results may not mean much."
        )
    if (
        isinstance(result, Answer)
        and result.supported_by_document != "no"
        and not any(quote.text for quote in result.quotes)
    ):
        warnings.append(
            "The answer cites no passage that could be found in your document; "
            "treat it with caution."
        )
    return warnings


class _Source(NamedTuple):
    text: str
    fingerprint: str
    positions: list[int]


def _fingerprint(text: str) -> tuple[str, list[int]]:
    """Reduce text to the characters that matter for matching, remembering where each came from.

    NFKD plus casefold folds ligatures, accents, full-width forms and case. Keeping only
    letters, digits and a few symbols ignores whitespace, line breaks, hyphenation and quote
    styles, which is exactly where extracted PDF text and the model's copy tend to differ.
    """
    kept: list[str] = []
    positions: list[int] = []
    for index, char in enumerate(text):
        for folded in unicodedata.normalize("NFKD", char).casefold():
            if folded.isalnum() or folded in _KEPT_SYMBOLS:
                kept.append(folded)
                positions.append(index)
    return "".join(kept), positions


def verify_quotes(result: BaseModel, documents: dict[str, str]) -> int:
    """Check every quote against the document it claims to come from.

    A found quote is replaced by the exact source passage (from its first to its last matched
    character, so edge punctuation is trimmed), which lets the UI locate it in the document.
    A quote that cannot be found is blanked, and a risk or difference with no quotes at all
    also counts as unmatched.

    Returns:
        The number of unmatched quotes and evidence-free findings.
    """
    sources = {label: _Source(text, *_fingerprint(text)) for label, text in documents.items()}
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
