import argparse

from extractor import extract_resume, DEFAULT_MODEL


def main():
    parser = argparse.ArgumentParser(description="Resume-to-JSON extractor (local LLM, structured output)")
    parser.add_argument("resume_file", help="Path to a .txt resume file")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Ollama model tag (default: llama3.1)")
    parser.add_argument("-o", "--output", help="Write JSON to this file instead of stdout")
    args = parser.parse_args()

    with open(args.resume_file, encoding="utf-8") as f:
        text = f.read()

    result = extract_resume(text, model=args.model)
    out = result.model_dump_json(indent=2)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(out)
    else:
        print(out)


if __name__ == "__main__":
    main()