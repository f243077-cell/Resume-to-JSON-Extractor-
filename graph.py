"""Four-node LangGraph over the extractor: router, retrieve, generate, check.

Extraction reads the whole resume, so it skips retrieval. Questions use retrieved excerpts.
If an answer has no supporting quote, check loops back to retrieve for a wider set of excerpts.
"""
import argparse
import re
from typing import List, Literal, Optional, TypedDict, Union

from langgraph.graph import END, START, StateGraph

from extractor import DEFAULT_MODEL, answer_question, extract_resume
from schema import Answer, Resume

TOP_K = 3
MAX_ATTEMPTS = 2


class GraphState(TypedDict, total=False):
    resume_text: str
    question: Optional[str]
    route: Literal["extract", "qa"]
    chunks: List[str]
    exhausted: bool  # retrieve returned every chunk, so a retry cannot widen the context
    result: Union[Resume, Answer, None]
    error: Optional[str]
    attempts: int
    passed: bool


def router(state: GraphState) -> GraphState:
    # Skeleton rule. Replace with an LLM classifier to add more routes.
    return {"route": "qa" if state.get("question") else "extract", "attempts": 0}


def retrieve(state: GraphState) -> GraphState:
    # Skeleton retriever: rank blank-line-separated chunks by word overlap with the question.
    # Replace with embeddings and a vector store.
    chunks = [chunk.strip() for chunk in re.split(r"\n\s*\n", state["resume_text"]) if chunk.strip()]
    terms = set(re.findall(r"\w+", state["question"].lower()))
    ranked = sorted(chunks, key=lambda chunk: -len(terms & set(re.findall(r"\w+", chunk.lower()))))
    top_k = TOP_K * 2 ** state["attempts"]
    return {"chunks": ranked[:top_k], "exhausted": top_k >= len(ranked)}


def after_router(state: GraphState) -> str:
    return "retrieve" if state["route"] == "qa" else "generate"


def after_check(state: GraphState) -> str:
    if state["passed"] or state["route"] != "qa":
        return END
    return END if state["exhausted"] or state["attempts"] >= MAX_ATTEMPTS else "retrieve"


def build_graph(model: str = DEFAULT_MODEL, client=None):
    def generate(state: GraphState) -> GraphState:
        attempts = state["attempts"] + 1
        try:
            if state["route"] == "qa":
                excerpts = "\n\n".join(state["chunks"])
                result = answer_question(excerpts, state["question"], model=model, client=client)
            else:
                result = extract_resume(state["resume_text"], model=model, client=client)
        except RuntimeError as err:  # validation retries ran out; connection errors still propagate
            return {"result": None, "error": str(err), "attempts": attempts}
        return {"result": result, "error": None, "attempts": attempts}

    def check(state: GraphState) -> GraphState:
        # Instructor already enforced the schema, so this judges usefulness, not shape.
        result = state["result"]
        if isinstance(result, Answer):
            return {"passed": bool(result.quotes)}
        if isinstance(result, Resume):
            has_content = any([result.experience, result.education, result.projects, result.skills])
            return {"passed": bool(result.contact.name) and has_content}
        return {"passed": False}

    builder = StateGraph(GraphState)
    builder.add_node("router", router)
    builder.add_node("retrieve", retrieve)
    builder.add_node("generate", generate)
    builder.add_node("check", check)

    builder.add_edge(START, "router")
    builder.add_conditional_edges("router", after_router, ["retrieve", "generate"])
    builder.add_edge("retrieve", "generate")
    builder.add_edge("generate", "check")
    builder.add_conditional_edges("check", after_check, ["retrieve", END])
    return builder.compile()


def main():
    parser = argparse.ArgumentParser(description="Run the resume graph, or print its Mermaid diagram")
    parser.add_argument("resume_file", nargs="?", help="Path to a .txt resume file")
    parser.add_argument("--ask", help="Question about the resume. Omit to extract JSON")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Ollama model tag (default: llama3.1)")
    parser.add_argument("--draw", action="store_true", help="Print the graph as Mermaid and exit")
    args = parser.parse_args()

    graph = build_graph(model=args.model)
    if args.draw:
        print(graph.get_graph().draw_mermaid())
        return
    if not args.resume_file:
        parser.error("resume_file is required unless --draw is set")

    with open(args.resume_file, encoding="utf-8") as f:
        state = graph.invoke({"resume_text": f.read(), "question": args.ask})

    if state["result"] is None:
        raise SystemExit(state["error"])
    print(state["result"].model_dump_json(indent=2))
    if not state["passed"]:
        raise SystemExit("Check failed: review this output before using it.")


if __name__ == "__main__":
    main()
