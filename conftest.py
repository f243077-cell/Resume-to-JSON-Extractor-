import json

import httpx
import instructor
import pytest
from openai import OpenAI


@pytest.fixture
def fake_llm():
    """Build an Instructor client that replays scripted replies and records each request body."""

    def build(*replies):
        queue, requests = list(replies), []

        def respond(request):
            requests.append(json.loads(request.content))
            reply = queue.pop(0)
            if isinstance(reply, Exception):
                raise reply
            message = {"role": "assistant", "content": reply}
            return httpx.Response(200, json={
                "id": "fake", "object": "chat.completion", "created": 0, "model": "fake",
                "choices": [{"index": 0, "finish_reason": "stop", "message": message}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            })

        openai_client = OpenAI(
            base_url="http://localhost:11434/v1",
            api_key="ollama",
            max_retries=0,
            http_client=httpx.Client(transport=httpx.MockTransport(respond)),
        )
        return instructor.from_openai(openai_client, mode=instructor.Mode.JSON_SCHEMA), requests

    return build
