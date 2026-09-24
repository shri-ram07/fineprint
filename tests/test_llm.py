import json
import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

import anthropic
import httpx2
import pytest

from fineprint.llm import AnthropicLLM, LLMError
from fineprint.schemas import Answer

VALID_ANSWER = json.dumps(
    {"answer": "Yes.", "supported_by_document": "yes", "quotes": [], "caveats": []}
)


def fake_client(stop_reason: str = "end_turn", text: str = VALID_ANSWER) -> MagicMock:
    """An Anthropic client whose streamed response is fixed (MagicMock credentials are truthy)."""
    client = MagicMock()
    stream = client.messages.stream.return_value.__enter__.return_value
    stream.request_id = "req_test"
    stream.get_final_message.return_value = SimpleNamespace(
        stop_reason=stop_reason,
        content=[
            SimpleNamespace(type="thinking", thinking=""),
            SimpleNamespace(type="text", text=text),
        ],
        usage=SimpleNamespace(input_tokens=10, output_tokens=5, cache_read_input_tokens=None),
    )
    return client


def complete(client: MagicMock) -> Answer:
    return AnthropicLLM("test-model", client).complete(
        system="system prompt", content=[{"type": "text", "text": "hi"}], output_model=Answer
    )


def test_valid_output_is_parsed_and_request_is_shaped_for_structured_output():
    client = fake_client()
    assert complete(client).answer == "Yes."

    request = client.messages.stream.call_args.kwargs
    assert request["model"] == "test-model"
    assert request["system"] == "system prompt"
    assert request["messages"] == [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
    assert request["output_config"]["format"]["type"] == "json_schema"
    assert request["output_config"]["format"]["schema"]["title"] == "Answer"


@pytest.mark.parametrize(
    ("stop_reason", "code"),
    # Valid JSON on purpose: a truncated or refused response must never be trusted.
    [("max_tokens", "truncated"), ("refusal", "declined")],
)
def test_stop_reason_is_checked_before_the_output(stop_reason, code):
    with pytest.raises(LLMError) as raised:
        complete(fake_client(stop_reason=stop_reason))
    assert raised.value.code == code


def test_invalid_output_is_malformed_and_not_logged(caplog):
    caplog.set_level(logging.DEBUG)
    with pytest.raises(LLMError) as raised:
        complete(fake_client(text='{"answer": "SECRET-CLAUSE-TEXT"}'))
    assert raised.value.code == "malformed"
    assert "SECRET-CLAUSE-TEXT" not in caplog.text


@pytest.mark.parametrize(
    ("status", "code"),
    [
        (401, "configuration"),
        (402, "configuration"),
        (404, "configuration"),
        (429, "unavailable"),
        (529, "unavailable"),
        (400, "malformed"),
    ],
)
def test_api_errors_map_to_user_facing_codes(status, code):
    client = fake_client()
    response = httpx2.Response(status, request=httpx2.Request("POST", "https://api.test"))
    client.messages.stream.side_effect = anthropic.APIStatusError(
        "upstream detail", response=response, body=None
    )
    with pytest.raises(LLMError) as raised:
        complete(client)
    assert raised.value.code == code
    assert "upstream detail" not in raised.value.message


def test_connection_failure_is_unavailable():
    client = fake_client()
    client.messages.stream.side_effect = anthropic.APIConnectionError(
        request=httpx2.Request("POST", "https://api.test")
    )
    with pytest.raises(LLMError) as raised:
        complete(client)
    assert raised.value.code == "unavailable"


def test_missing_credentials_fail_at_construction():
    client = MagicMock(api_key="", auth_token=None, credentials=None)
    with pytest.raises(LLMError) as raised:
        AnthropicLLM("test-model", client)
    assert raised.value.code == "configuration"
    assert "ANTHROPIC_API_KEY" in raised.value.message
