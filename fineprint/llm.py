"""The only module that talks to the model: Gemini, driven through Google's Agent Development Kit.

`GeminiLLM` turns one request into one validated pydantic object or one `LLMError` whose
message is safe to show to the user. API details go to the log, never to the user.
"""

import logging
import os
import re
import time
from typing import Literal, Protocol, TypeVar

from google.adk.agents import LlmAgent
from google.adk.events import Event
from google.adk.models import Gemini
from google.adk.runners import InMemoryRunner
from google.genai import Client, errors, types
from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

DEFAULT_MODEL = "gemini-3.5-flash-lite"
# Thinking tokens count against this ceiling, so it is set well above the visible output.
MAX_OUTPUT_TOKENS = 32_000
# A hung request must not hold a worker forever; long analyses take well under a minute.
REQUEST_TIMEOUT_MS = 120_000
# Transient failures (429, 5xx) are retried with backoff by the SDK before we give up.
RETRY_ATTEMPTS = 3

Effort = Literal["low", "default"]
LLMErrorCode = Literal["configuration", "declined", "truncated", "malformed", "unavailable"]

_MESSAGES: dict[LLMErrorCode, str] = {
    "configuration": "The AI service is not configured correctly. Check the server's API key "
    "and model settings.",
    "declined": "The AI declined to work on this document.",
    "truncated": "This document is too long for a complete answer. Try a shorter section.",
    "malformed": "The AI returned an unusable response. Please try again.",
    "unavailable": "Could not reach the AI service. Your document is still here; please try "
    "again in a moment.",
}

NO_CREDENTIALS = (
    "No Gemini API key found. Put GOOGLE_API_KEY in .env and start with "
    "`uv run --env-file .env uvicorn --factory fineprint.web:create_app`."
)

_APP = "fineprint"
_USER = "local"
# Any casing or spacing of a closing document tag inside the document text itself.
_CLOSING_TAG = re.compile(r"<(\s*)/(\s*)document", re.IGNORECASE)


class LLMError(Exception):
    """The model call failed. `message` is safe to show to the user."""

    def __init__(self, code: LLMErrorCode, message: str | None = None) -> None:
        self.code = code
        self.message = message or _MESSAGES[code]
        super().__init__(self.message)


class LLMClient(Protocol):
    async def complete(
        self,
        *,
        system: str,
        documents: dict[str, str],
        request: str,
        output_model: type[T],
        effort: Effort = "default",
    ) -> T:
        """Return the model's answer validated as `output_model`, or raise `LLMError`.

        `documents` maps a label ("A", "B") to the document text; `request` is the task.
        `effort="low"` asks for less reasoning, for simple lookups.
        """
        ...


def _code_for_error(exc: errors.APIError) -> LLMErrorCode:
    # The Gemini API reports an invalid key as a 400, so check for it before the status.
    if exc.code in (401, 403, 404) or "API_KEY_INVALID" in str(exc.details):
        return "configuration"
    if exc.code == 429 or exc.code >= 500:  # rate limited, overloaded or down: worth retrying
        return "unavailable"
    return "malformed"  # the request itself was rejected: a bug on our side


def _wrap_document(label: str, text: str) -> str:
    # A document must not be able to close its own tag and add text that reads as ours.
    # Quote matching ignores the extra backslash, so verification is unaffected.
    safe = _CLOSING_TAG.sub(r"<\1\\/\2document", text)
    return f'<document id="{label}">\n{safe}\n</document>'


class GeminiLLM:
    def __init__(self, model: str, client: Client | None = None) -> None:
        try:
            # Reads GOOGLE_API_KEY (or GEMINI_API_KEY) and fails now, at startup, if neither
            # is set, instead of on the user's first request.
            client = client or Client(
                http_options=types.HttpOptions(
                    timeout=REQUEST_TIMEOUT_MS,
                    retry_options=types.HttpRetryOptions(attempts=RETRY_ATTEMPTS),
                )
            )
        except ValueError:
            raise LLMError("configuration", NO_CREDENTIALS) from None
        self.model_name = model
        self._gemini = Gemini(model=model, client=client)
        self._runners: dict[tuple[str, type[BaseModel], Effort], InMemoryRunner] = {}

    @classmethod
    def from_env(cls) -> "GeminiLLM":
        return cls(os.environ.get("GEMINI_MODEL") or DEFAULT_MODEL)

    def _runner(self, system: str, output_model: type[BaseModel], effort: Effort) -> InMemoryRunner:
        """One agent and runner per (prompt, schema, effort), built on first use and reused.

        Without `output_key`, ADK returns the raw text instead of validating it, which lets us
        check why generation stopped before parsing.
        """
        key = (system, output_model, effort)
        if key not in self._runners:
            thinking = (
                types.ThinkingConfig(thinking_level=types.ThinkingLevel.LOW)
                if effort == "low"
                else None
            )
            agent = LlmAgent(
                name=_APP,
                model=self._gemini,
                static_instruction=system,  # sent verbatim, unlike `instruction`'s {templating}
                output_schema=output_model,
                generate_content_config=types.GenerateContentConfig(
                    max_output_tokens=MAX_OUTPUT_TOKENS,
                    thinking_config=thinking,
                    # No tools are used; this also silences the SDK's per-call AFC warning.
                    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
                ),
            )
            self._runners[key] = InMemoryRunner(agent=agent, app_name=_APP)
        return self._runners[key]

    async def complete(
        self,
        *,
        system: str,
        documents: dict[str, str],
        request: str,
        output_model: type[T],
        effort: Effort = "default",
    ) -> T:
        # Documents go first so repeated questions about the same document share a prefix,
        # which Gemini caches implicitly.
        parts = [types.Part(text=_wrap_document(label, text)) for label, text in documents.items()]
        parts.append(types.Part(text=request))
        started = time.perf_counter()
        final = await self._run(self._runner(system, output_model, effort), parts)
        self._log_usage(final, effort, time.perf_counter() - started)
        return _parse(final, output_model)

    async def _run(self, runner: InMemoryRunner, parts: list[types.Part]) -> Event:
        """Run one single-turn session and return its final event."""
        sessions = runner.session_service
        session = await sessions.create_session(app_name=_APP, user_id=_USER)
        final = None
        try:
            async for event in runner.run_async(
                user_id=_USER,
                session_id=session.id,
                new_message=types.Content(role="user", parts=parts),
            ):
                if event.is_final_response():
                    final = event
        except errors.APIError as exc:
            logger.error("Gemini API error %s %s: %s", exc.code, exc.status, exc.message)
            raise LLMError(_code_for_error(exc)) from None
        except OSError as exc:  # connection failures and timeouts
            logger.error("Gemini API unreachable: %s", type(exc).__name__)
            raise LLMError("unavailable") from None
        finally:
            # Each request is one turn; dropping the session keeps memory flat.
            await sessions.delete_session(app_name=_APP, user_id=_USER, session_id=session.id)
        if final is None:
            logger.error("Gemini returned no final response")
            raise LLMError("malformed")
        return final

    def _log_usage(self, final: Event, effort: Effort, seconds: float) -> None:
        usage = final.usage_metadata
        logger.info(
            "model=%s effort=%s finish=%s error=%s input_tokens=%s output_tokens=%s "
            "thinking_tokens=%s cached_tokens=%s duration=%.1fs",
            self.model_name,
            effort,
            final.finish_reason,
            final.error_code,
            usage and usage.prompt_token_count,
            usage and usage.candidates_token_count,
            usage and usage.thoughts_token_count,
            (usage and usage.cached_content_token_count) or 0,
            seconds,
        )


def _parse(final: Event, output_model: type[T]) -> T:
    """Validate the final text, but only after checking why generation stopped."""
    # A truncated response can still be valid JSON that silently omits most of the analysis.
    if final.finish_reason == types.FinishReason.MAX_TOKENS:
        raise LLMError("truncated")
    # A blocked prompt sets error_code; a response stopped by a safety filter can still carry
    # partial text, so anything but a normal stop is refused rather than parsed.
    if final.error_code or final.finish_reason not in (None, types.FinishReason.STOP):
        raise LLMError("declined")
    parts = final.content.parts if final.content and final.content.parts else []
    text = "".join(part.text for part in parts if part.text and not part.thought)
    try:
        return output_model.model_validate_json(text)
    except ValidationError as exc:
        # Log only the failing field paths: pydantic's message embeds the input values,
        # which here are passages of the user's document.
        locations = [error["loc"] for error in exc.errors()]
        logger.error("Model output did not match %s at %s", output_model.__name__, locations)
        raise LLMError("malformed") from None
