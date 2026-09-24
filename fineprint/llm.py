"""The only module that talks to the model: Gemini, driven through Google's Agent Development Kit.

`GeminiLLM` turns one request into one validated pydantic object or one `LLMError` whose
message is safe to show to the user. API details go to the log, never to the user.
"""

import logging
import os
import time
from typing import Literal, Protocol, TypeVar

from google.adk.agents import LlmAgent
from google.adk.models import Gemini
from google.adk.runners import InMemoryRunner
from google.genai import Client, errors, types
from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

DEFAULT_MODEL = "gemini-3.5-flash"
# Thinking tokens count against this ceiling, so it is set well above the visible output.
MAX_OUTPUT_TOKENS = 32_000

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


class LLMError(Exception):
    """The model call failed. `message` is safe to show to the user."""

    def __init__(self, code: LLMErrorCode, message: str | None = None) -> None:
        self.code = code
        self.message = message or _MESSAGES[code]
        super().__init__(self.message)


class LLMClient(Protocol):
    async def complete(
        self, *, system: str, documents: dict[str, str], request: str, output_model: type[T]
    ) -> T:
        """Return the model's answer validated as `output_model`, or raise `LLMError`.

        `documents` maps a label ("A", "B") to the document text; `request` is the task.
        """
        ...


def _code_for_error(exc: errors.APIError) -> LLMErrorCode:
    # The Gemini API reports an invalid key as a 400, so check for it before the status.
    if exc.code in (401, 403, 404) or "API_KEY_INVALID" in str(exc.details):
        return "configuration"
    if exc.code == 429 or exc.code >= 500:  # rate limited, overloaded or down: worth retrying
        return "unavailable"
    return "malformed"  # the request itself was rejected: a bug on our side


class GeminiLLM:
    def __init__(self, model: str, client: Client | None = None) -> None:
        try:
            # Reads GOOGLE_API_KEY (or GEMINI_API_KEY) and fails now, at startup, if neither
            # is set, instead of on the user's first request.
            client = client or Client()
        except ValueError:
            raise LLMError("configuration", NO_CREDENTIALS) from None
        self.model = Gemini(model=model, client=client)

    @classmethod
    def from_env(cls) -> "GeminiLLM":
        return cls(os.environ.get("GEMINI_MODEL") or DEFAULT_MODEL)

    async def complete(
        self, *, system: str, documents: dict[str, str], request: str, output_model: type[T]
    ) -> T:
        # A fresh single-turn agent per request: the app is stateless, so there is no
        # conversation history to keep. Without `output_key`, ADK returns the raw text
        # instead of validating it, which lets us check why generation stopped first.
        agent = LlmAgent(
            name=_APP,
            model=self.model,
            static_instruction=system,  # sent verbatim, unlike `instruction`'s {templating}
            output_schema=output_model,
            generate_content_config=types.GenerateContentConfig(
                max_output_tokens=MAX_OUTPUT_TOKENS
            ),
        )
        runner = InMemoryRunner(agent=agent, app_name=_APP)
        session = await runner.session_service.create_session(app_name=_APP, user_id=_USER)
        # Documents go first so repeated questions about the same document share a prefix,
        # which Gemini caches implicitly.
        parts = [
            types.Part(text=f'<document id="{label}">\n{text}\n</document>')
            for label, text in documents.items()
        ]
        parts.append(types.Part(text=request))

        started = time.perf_counter()
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

        if final is None:
            logger.error("Gemini returned no final response")
            raise LLMError("malformed")
        usage = final.usage_metadata
        logger.info(
            "model=%s finish=%s error=%s input_tokens=%s output_tokens=%s thinking_tokens=%s "
            "cached_tokens=%s duration=%.1fs",
            self.model.model,
            final.finish_reason,
            final.error_code,
            usage and usage.prompt_token_count,
            usage and usage.candidates_token_count,
            usage and usage.thoughts_token_count,
            (usage and usage.cached_content_token_count) or 0,
            time.perf_counter() - started,
        )

        # Check why generation stopped before trusting the text: a truncated response can
        # still be valid JSON that silently omits most of the analysis.
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
