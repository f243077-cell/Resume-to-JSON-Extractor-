import json

import httpx
import instructor
import pytest
from openai import APIConnectionError

from extractor import answer_question, extract_resume, get_client
from prompt import FEW_SHOT, QA_FEW_SHOT, SYSTEM_PROMPT
from schema import Answer

RESUME = "Ada Lovelace\nada@example.com\n\nEXPERIENCE\nAnalyst, Babbage Ltd (Jun 2021 - current)\n- Wrote programs\n\nSKILLS\nMath"


def reply(company="Babbage Ltd", **overrides):
    return json.dumps({
        "contact": {"name": "Ada Lovelace", "email": "ada@example.com"},
        "skills": ["Math"],
        "experience": [{"company": company, "title": "Analyst", "start_date": "Jun 2021", "end_date": "current"}],
        **overrides,
    })


def test_production_client_uses_schema_constraint():
    assert get_client().mode == instructor.Mode.JSON_SCHEMA


def test_sends_schema_constraint_and_zero_temperature(fake_llm):
    client, requests = fake_llm(reply())
    extract_resume(RESUME, client=client)

    body = requests[0]
    assert body["response_format"]["type"] == "json_schema"
    assert "contact" in body["response_format"]["json_schema"]["schema"]["properties"]
    assert body["temperature"] == 0


def test_validators_normalize_dates_and_skills(fake_llm):
    client, _ = fake_llm(reply(skills=["Math", "math ", "Logic"]))
    resume = extract_resume(RESUME, client=client)

    assert (resume.experience[0].start_date, resume.experience[0].end_date) == ("2021-06", "Present")
    assert resume.skills == ["Math", "Logic"]


def test_unknown_date_text_is_kept(fake_llm):
    client, _ = fake_llm(reply(education=[{"institution": "Babbage Ltd", "start_date": "Summer 2019"}]))
    assert extract_resume(RESUME, client=client).education[0].start_date == "Summer 2019"


def test_value_missing_from_source_is_reasked(fake_llm):
    client, requests = fake_llm(reply(company="Acme Corp"), reply())
    resume = extract_resume(RESUME, client=client)

    assert resume.experience[0].company == "Babbage Ltd"
    assert len(requests) == 2
    assert "not in the resume text" in json.dumps(requests[1]["messages"])


def test_raises_after_retries_run_out(fake_llm):
    client, requests = fake_llm(*[reply(company="Acme Corp")] * 3)
    with pytest.raises(RuntimeError, match="after 3 attempts"):
        extract_resume(RESUME, client=client)
    assert len(requests) == 3


def test_server_failure_is_not_reported_as_bad_output(fake_llm):
    client, requests = fake_llm(httpx.ConnectError("down"))
    with pytest.raises(APIConnectionError):
        extract_resume(RESUME, client=client)
    assert len(requests) == 1


def test_malformed_json_is_reasked(fake_llm):
    client, requests = fake_llm("not json", reply())
    assert extract_resume(RESUME, client=client).contact.name == "Ada Lovelace"
    assert len(requests) == 2


@pytest.mark.parametrize("payload", ["{{ 7 * 7 }}", "{% broken", "{# open", "{{ ''.__class__ }}"])
def test_resume_text_is_never_rendered_as_template(fake_llm, payload):
    text = f"{RESUME}\n{payload}"
    client, requests = fake_llm(reply(company="Acme Corp"), reply())
    extract_resume(text, client=client)

    assert len(requests) == 2
    user_messages = [m["content"] for m in requests[1]["messages"] if m["role"] == "user"]
    assert f"Resume:\n\n{text}\n\nExtract as JSON." in user_messages


def test_static_prompts_survive_rendering(fake_llm):
    client, requests = fake_llm(reply())
    extract_resume(RESUME, client=client)

    sent = [message["content"] for message in requests[0]["messages"]]
    assert SYSTEM_PROMPT.strip() in sent[0]
    assert sent[1:5] == [message["content"] for message in FEW_SHOT]


@pytest.mark.parametrize("text", ["null", "NULL", "None", "n/a", ""])
def test_literal_null_date_becomes_none(fake_llm, text):
    payload = json.loads(reply())
    payload["experience"][0].update(start_date=text, end_date=text)
    client, _ = fake_llm(json.dumps(payload))

    experience = extract_resume(RESUME, client=client).experience[0]
    assert (experience.start_date, experience.end_date) == (None, None)


def test_answer_schema_generates_quotes_first():
    schema = Answer.model_json_schema()
    assert list(schema["properties"]) == ["quotes", "answer"]
    assert schema["required"] == ["quotes", "answer"]


def test_empty_answer_is_reasked(fake_llm):
    client, requests = fake_llm('{"answer": "", "quotes": []}', '{"answer": "Math", "quotes": ["Math"]}')
    assert answer_question("SKILLS\nMath", "Skills?", client=client).answer == "Math"
    assert len(requests) == 2


def test_question_sends_few_shot_examples(fake_llm):
    client, requests = fake_llm('{"answer": "Math", "quotes": ["Math"]}')
    answer_question("SKILLS\nMath", "Skills?", client=client)

    assert [m["content"] for m in requests[0]["messages"][1:5]] == [m["content"] for m in QA_FEW_SHOT]


def test_question_text_is_never_rendered_as_template(fake_llm):
    client, requests = fake_llm('{"answer": "Math", "quotes": ["Math"]}')
    answer = answer_question("SKILLS\nMath", "{{ 7 * 7 }} skills?", client=client)

    assert answer.quotes == ["Math"]
    assert requests[0]["messages"][-1]["content"].endswith("Question: {{ 7 * 7 }} skills?")


def test_fabricated_quote_is_reasked(fake_llm):
    client, requests = fake_llm('{"answer": "x", "quotes": ["Invented"]}', '{"answer": "Math", "quotes": ["Math"]}')
    assert answer_question("SKILLS\nMath", "Skills?", client=client).quotes == ["Math"]
    assert len(requests) == 2
