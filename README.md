# Resume-to-JSON Extractor

Local, open-weight LLM extracts structured resume data. Prompt-engineered system prompt
(rules + few-shot), Pydantic schema enforcement, Ollama structured-output constraint,
and a validate-retry loop. Includes prompt-injection defense against instructions hidden
inside resume text.

## Setup

```bash
pip install -r requirements.txt
ollama pull llama3.1        # or mistral, qwen2.5, etc.
ollama serve                # if not already running
```

## Run

```bash
python cli.py path/to/resume.txt
python cli.py resume.txt --model mistral -o resume.json
```

## Files

- `schema.py` — Pydantic models defining the target JSON shape
- `prompt.py` — system prompt, injection-defense rules, few-shot example
- `extractor.py` — calls Ollama (`format=schema`), validates, retries on failure
- `cli.py` — command-line entry point

## Extending

- Swap `DEFAULT_MODEL` in `extractor.py` for any local model tag pulled via `ollama pull`.
- Add more `FEW_SHOT` pairs in `prompt.py` for edge cases (multi-column resumes, non-English, etc.).
- Add a batch mode in `cli.py` to loop over a directory of resumes.
