"""The Gemini adapter, run through real ADK with only the network call replaced."""

import asyncio
import json
import logging
from unittest.mock import AsyncMock

import pytest
from google.genai import Client, errors, types

from fineprint.llm import GeminiLLM, LLMError
from fineprint.schemas import Answer

VALID_ANSWER = json.dumps(
    {"answer": "Yes.", "supported_by_document": "yes", "quotes": [], "caveats": []}
)


def gemini_response(text: str = VALID_ANSWER, finish=types.FinishReason.STOP):
    return types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                content=types.Content(
                    role="model",
                    parts=[types.Part(text="thinking...", thought=True), types.Part(text=text)],
                ),
                finish_reason=finish,
            )
        ],
        usage_metadata=types.GenerateContentResponseUsageMetadata(
            prompt_token_count=10, candidates_token_count=5
        ),
    )


def fake_client(result=None) -> Client:
    """A real genai Client whose only network call is replaced by a stub."""
    client = Client(api_key="test-key")
    client.aio.models.generate_content = AsyncMock(
        side_effect=result if isinstance(result, Exception) else None,
        return_value=result or gemini_response(),
    )
    return client


def complete(client: Client) -> Answer:
    llm = GeminiLLM("test-model", client)
    return asyncio.run(
        llm.complete(
            system="system prompt",
            documents={"A": "Rent is £900.", "B": "Rent is £950."},
            request="Which rent is higher?",
            output_model=Answer,
        )
    )


def test_valid_output_is_parsed_and_request_is_shaped_for_structured_output():
    client = fake_client()
    assert complete(client).answer == "Yes."

    request = client.aio.models.generate_content.call_args.kwargs
    assert request["model"] == "test-model"
    assert request["config"].system_instruction is not None
    assert "system prompt" in str(request["config"].system_instruction)
    assert request["config"].response_schema is Answer
    assert request["config"].max_output_tokens == 32_000
    texts = [part.text for part in request["contents"][-1].parts]
    assert texts[0] == '<document id="A">\nRent is £900.\n</document>'
    assert texts[1] == '<document id="B">\nRent is £950.\n</document>'
    assert texts[-1] == "Which rent is higher?"


@pytest.mark.parametrize(
    ("finish", "code"),
    # Valid JSON on purpose: a truncated or blocked response must never be trusted.
    [(types.FinishReason.MAX_TOKENS, "truncated"), (types.FinishReason.SAFETY, "declined")],
)
def test_finish_reason_is_checked_before_the_output(finish, code):
    with pytest.raises(LLMError) as raised:
        complete(fake_client(gemini_response(finish=finish)))
    assert raised.value.code == code


def test_blocked_prompt_is_declined():
    blocked = types.GenerateContentResponse(
        prompt_feedback=types.GenerateContentResponsePromptFeedback(
            block_reason=types.BlockedReason.SAFETY
        )
    )
    with pytest.raises(LLMError) as raised:
        complete(fake_client(blocked))
    assert raised.value.code == "declined"


def test_invalid_output_is_malformed_and_not_logged(caplog):
    caplog.set_level(logging.INFO)
    with pytest.raises(LLMError) as raised:
        complete(fake_client(gemini_response('{"answer": "SECRET-CLAUSE-TEXT"}')))
    assert raised.value.code == "malformed"
    assert "SECRET-CLAUSE-TEXT" not in caplog.text


def api_error(code: int, status: str, reason: str = "") -> errors.APIError:
    body = {"error": {"code": code, "status": status, "message": "upstream detail"}}
    if reason:
        body["error"]["details"] = [{"reason": reason}]
    return (errors.ServerError if code >= 500 else errors.ClientError)(code, body)


@pytest.mark.parametrize(
    ("error", "code"),
    [
        (api_error(400, "INVALID_ARGUMENT", "API_KEY_INVALID"), "configuration"),
        (api_error(403, "PERMISSION_DENIED"), "configuration"),
        (api_error(404, "NOT_FOUND"), "configuration"),
        (api_error(429, "RESOURCE_EXHAUSTED"), "unavailable"),
        (api_error(503, "UNAVAILABLE"), "unavailable"),
        (api_error(400, "INVALID_ARGUMENT"), "malformed"),
        (ConnectionError("refused"), "unavailable"),
    ],
    ids=[
        "bad-key",
        "forbidden",
        "unknown-model",
        "rate-limited",
        "overloaded",
        "bad-request",
        "offline",
    ],
)
def test_api_errors_map_to_user_facing_codes(error, code):
    with pytest.raises(LLMError) as raised:
        complete(fake_client(error))
    assert raised.value.code == code
    assert "upstream detail" not in raised.value.message


def test_missing_api_key_fails_at_construction(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(LLMError) as raised:
        GeminiLLM("test-model")
    assert raised.value.code == "configuration"
    assert "GOOGLE_API_KEY" in raised.value.message


def test_api_key_from_environment_is_accepted(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    GeminiLLM("test-model")
