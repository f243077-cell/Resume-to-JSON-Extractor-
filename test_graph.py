import json

import httpx
import pytest
from openai import APIConnectionError

from graph import MAX_ATTEMPTS, build_graph
from schema import Answer, Resume

CHUNKS = ["Ada Lovelace", "EXPERIENCE\nAnalyst at Babbage", "SKILLS\nMath", "EDUCATION\nHome tutoring",
          "PROJECTS\nEngine notes", "AWARDS\nFirst programmer", "HOBBIES\nPoetry"]
RESUME = "\n\n".join(CHUNKS)
GOOD_RESUME = json.dumps({"contact": {"name": "Ada Lovelace"}, "skills": ["Math"]})
EMPTY_RESUME = json.dumps({"contact": {}})
NOT_FOUND = '{"answer": "Not in the excerpts", "quotes": []}'


def run(client, question=None):
    return build_graph(client=client).invoke({"resume_text": RESUME, "question": question})


def test_graph_has_exactly_the_four_nodes():
    nodes = set(build_graph().get_graph().nodes) - {"__start__", "__end__"}
    assert nodes == {"router", "retrieve", "generate", "check"}


def test_no_question_extracts_without_retrieval(fake_llm):
    client, requests = fake_llm(GOOD_RESUME)
    state = run(client)

    assert isinstance(state["result"], Resume)
    assert state["passed"] and state["route"] == "extract"
    assert "chunks" not in state
    assert len(requests) == 1


def test_empty_extraction_fails_check_without_looping(fake_llm):
    client, requests = fake_llm(EMPTY_RESUME)
    state = run(client)

    assert not state["passed"]
    assert len(requests) == 1


def test_question_uses_top_ranked_chunks(fake_llm):
    client, requests = fake_llm('{"answer": "Math", "quotes": ["Math"]}')
    state = run(client, "What skills?")

    assert isinstance(state["result"], Answer)
    assert state["passed"] and state["chunks"][0] == "SKILLS\nMath"
    assert "HOBBIES" not in requests[0]["messages"][-1]["content"]


def test_unsupported_answer_widens_retrieval_then_passes(fake_llm):
    client, requests = fake_llm(NOT_FOUND, '{"answer": "Poetry", "quotes": ["Poetry"]}')
    state = run(client, "What hobbies?")

    assert state["passed"] and state["attempts"] == 2
    assert len(state["chunks"]) == 6
    assert len(requests) == 2


def test_loop_stops_at_attempt_limit(fake_llm):
    client, requests = fake_llm(*[NOT_FOUND] * (MAX_ATTEMPTS + 1))
    state = run(client, "What salary?")

    assert not state["passed"] and state["attempts"] == MAX_ATTEMPTS
    assert len(requests) == MAX_ATTEMPTS


def test_no_retry_when_every_chunk_was_already_sent(fake_llm):
    client, requests = fake_llm(NOT_FOUND)
    state = build_graph(client=client).invoke({"resume_text": "SKILLS\nMath", "question": "Salary?"})

    assert not state["passed"]
    assert len(requests) == 1


def test_server_failure_stops_the_graph(fake_llm):
    client, _ = fake_llm(httpx.ConnectError("down"))
    with pytest.raises(APIConnectionError):
        run(client, "What skills?")


def test_exhausted_validation_retries_end_as_failed_check(fake_llm):
    client, requests = fake_llm(*[json.dumps({"contact": {"name": "Jane Doe"}})] * 3)
    state = run(client)

    assert state["result"] is None and "after 3 attempts" in state["error"]
    assert not state["passed"]
    assert len(requests) == 3
