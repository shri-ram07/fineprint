"""The only module that talks to the Anthropic API.

`AnthropicLLM` turns one request into one validated pydantic object or one `LLMError`
whose message is safe to show to the user. SDK details go to the log, never to the user.
"""

import logging
import os
import time
from typing import Any, Literal, Protocol, TypeVar

import anthropic
from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

DEFAULT_MODEL = "claude-opus-5"
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
    "No Anthropic credentials found. Put ANTHROPIC_API_KEY in .env and start with "
    "`uv run --env-file .env uvicorn --factory fineprint.web:create_app`, or run `ant auth login`."
)


class LLMError(Exception):
    """The model call failed. `message` is safe to show to the user."""

    def __init__(self, code: LLMErrorCode, message: str | None = None) -> None:
        self.code = code
        self.message = message or _MESSAGES[code]
        super().__init__(self.message)


class LLMClient(Protocol):
    def complete(self, *, system: str, content: list[dict[str, Any]], output_model: type[T]) -> T:
        """Return the model's answer validated as `output_model`, or raise `LLMError`."""
        ...


def _code_for_status(status: int) -> LLMErrorCode:
    if status in (401, 402, 403, 404):  # bad key, billing, permissions, unknown model
        return "configuration"
    # Rate limited, overloaded or down: worth retrying. Below 400 means an error event arrived
    # mid-stream after the HTTP 200 (typically overloaded_error), which is just as transient.
    if status == 429 or status >= 500 or status < 400:
        return "unavailable"
    return "malformed"  # the request itself was rejected: a bug on our side


class AnthropicLLM:
    def __init__(self, model: str, client: anthropic.Anthropic | None = None) -> None:
        self.model = model
        self.client = client or anthropic.Anthropic()
        # The SDK only notices missing credentials when the first request is sent; fail at
        # startup instead.
        if not (self.client.api_key or self.client.auth_token or self.client.credentials):
            raise LLMError("configuration", NO_CREDENTIALS)

    @classmethod
    def from_env(cls) -> "AnthropicLLM":
        return cls(os.environ.get("ANTHROPIC_MODEL") or DEFAULT_MODEL)

    def complete(self, *, system: str, content: list[dict[str, Any]], output_model: type[T]) -> T:
        started = time.perf_counter()
        try:
            # Streaming avoids HTTP timeouts on long documents. The schema goes through
            # output_config rather than the SDK's output_format helper because the helper
            # parses before stop_reason is known, which would hide truncation and refusals.
            with self.client.messages.stream(
                model=self.model,
                max_tokens=MAX_OUTPUT_TOKENS,
                system=system,
                messages=[{"role": "user", "content": content}],
                output_config={
                    "format": {"type": "json_schema", "schema": output_model.model_json_schema()}
                },
            ) as stream:
                message = stream.get_final_message()
                request_id = stream.request_id
        except anthropic.APIConnectionError as exc:
            logger.error("Anthropic API unreachable: %s", exc)
            raise LLMError("unavailable") from None
        except anthropic.APIStatusError as exc:
            logger.error(
                "Anthropic API error %s (request %s): %s", exc.status_code, exc.request_id, exc
            )
            raise LLMError(_code_for_status(exc.status_code)) from None

        usage = message.usage
        logger.info(
            "model=%s stop=%s input_tokens=%s output_tokens=%s cache_read_tokens=%s "
            "duration=%.1fs request=%s",
            self.model,
            message.stop_reason,
            usage.input_tokens,
            usage.output_tokens,
            usage.cache_read_input_tokens or 0,
            time.perf_counter() - started,
            request_id,
        )

        # Check why generation stopped before trusting the text: a truncated response can
        # still be valid JSON that silently omits most of the analysis.
        if message.stop_reason == "refusal":
            raise LLMError("declined")
        if message.stop_reason == "max_tokens":
            raise LLMError("truncated")

        text = "".join(block.text for block in message.content if block.type == "text")
        try:
            return output_model.model_validate_json(text)
        except ValidationError as exc:
            # Log only the failing field paths: pydantic's message embeds the input values,
            # which here are passages of the user's document.
            locations = [error["loc"] for error in exc.errors()]
            logger.error("Model output did not match %s at %s", output_model.__name__, locations)
            raise LLMError("malformed") from None
