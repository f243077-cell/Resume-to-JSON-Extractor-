from functools import lru_cache
from typing import Optional, Type, TypeVar

import instructor
from instructor.core import InstructorRetryException
from openai import APIError, OpenAI
from pydantic import BaseModel

from prompt import FEW_SHOT, QA_FEW_SHOT, QA_SYSTEM_PROMPT, SYSTEM_PROMPT
from schema import Answer, Resume

OLLAMA_URL = "http://localhost:11434/v1"
DEFAULT_MODEL = "llama3.1"

Model = TypeVar("Model", bound=BaseModel)


@lru_cache
def get_client() -> instructor.Instructor:
    # JSON_SCHEMA sends the schema as response_format, so Ollama constrains generation to it.
    # instructor.from_provider("ollama/...") would pick Mode.TOOLS, which loses that constraint.
    return instructor.from_openai(
        OpenAI(base_url=OLLAMA_URL, api_key="ollama", timeout=600),
        mode=instructor.Mode.JSON_SCHEMA,
    )


def _create(
    response_model: Type[Model],
    messages: list,
    context: dict,
    model: str,
    max_retries: int,
    client: Optional[instructor.Instructor],
) -> Model:
    """Run a validated completion. Failed validation goes back to the model up to max_retries times.

    Instructor renders every message as a Jinja template when context is set, and also passes context
    to the Pydantic validators. Untrusted text must reach the messages only as a context variable:
    inlined text would be evaluated as a template.
    """
    try:
        return (client or get_client()).chat.completions.create(
            model=model,
            response_model=response_model,
            messages=messages,
            context=context,
            max_retries=max_retries,
            temperature=0,
        )
    except InstructorRetryException as err:
        if isinstance(err.__cause__, APIError):  # server down or model not pulled, not bad model output
            raise err.__cause__
        raise RuntimeError(f"Failed to get valid JSON after {err.n_attempts} attempts: {err}") from err


def extract_resume(
    resume_text: str,
    model: str = DEFAULT_MODEL,
    max_retries: int = 2,
    client: Optional[instructor.Instructor] = None,
) -> Resume:
    """Extract a Resume object from raw resume text using a local Ollama model."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *FEW_SHOT,
        {"role": "user", "content": "Resume:\n\n{{ resume_text }}\n\nExtract as JSON."},
    ]
    return _create(Resume, messages, {"resume_text": resume_text}, model, max_retries, client)


def answer_question(
    excerpts: str,
    question: str,
    model: str = DEFAULT_MODEL,
    max_retries: int = 2,
    client: Optional[instructor.Instructor] = None,
) -> Answer:
    """Answer a question from resume excerpts, with supporting quotes checked against those excerpts."""
    messages = [
        {"role": "system", "content": QA_SYSTEM_PROMPT},
        *QA_FEW_SHOT,
        {"role": "user", "content": "Resume excerpts:\n\n{{ resume_text }}\n\nQuestion: {{ question }}"},
    ]
    context = {"resume_text": excerpts, "question": question}
    return _create(Answer, messages, context, model, max_retries, client)
