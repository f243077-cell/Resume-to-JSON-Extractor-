import json
import requests
from pydantic import ValidationError

from schema import Resume
from prompt import SYSTEM_PROMPT, FEW_SHOT

OLLAMA_URL = "http://localhost:11434/api/chat"
DEFAULT_MODEL = "llama3.1"


def extract_resume(resume_text: str, model: str = DEFAULT_MODEL, max_retries: int = 2) -> Resume:
    """Extract a Resume object from raw resume text using a local Ollama model.

    Uses Ollama's structured-output mode (format=<json schema>) to constrain
    generation, plus a validate-and-retry loop as a second line of defense.
    """
    schema = Resume.model_json_schema()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *FEW_SHOT,
        {"role": "user", "content": f"Resume:\n\n{resume_text}\n\nExtract as JSON."},
    ]

    last_error = None
    for attempt in range(max_retries + 1):
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": model,
                "messages": messages,
                "format": schema,
                "stream": False,
                "options": {"temperature": 0},
            },
            timeout=600,
        )
        response.raise_for_status()
        content = response.json()["message"]["content"]

        try:
            data = json.loads(content)
            return Resume.model_validate(data)
        except (json.JSONDecodeError, ValidationError) as err:
            last_error = err
            messages.append({"role": "assistant", "content": content})
            messages.append(
                {
                    "role": "user",
                    "content": f"That output was invalid: {err}. Return corrected, schema-valid JSON only.",
                }
            )

    raise RuntimeError(f"Failed to extract valid resume JSON after {max_retries + 1} attempts: {last_error}")