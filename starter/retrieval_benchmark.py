"""Compare TF-IDF, Day 6 dense retrieval, and RRF hybrid retrieval."""

import argparse
import json
import re
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import dotenv_values

from baseline import TfidfIndex, pack_context


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DAY6_STARTER = (
    ROOT / "K5-L3A-DAY06-NguyenCongHuy-2A2604086-DataIngestion" / "starter"
)
DEFAULT_CHUNKS = ROOT / "starter" / "day6_chunks.json"
DEFAULT_QUESTIONS = ROOT / "starter" / "day6_questions.json"
DEFAULT_OUTPUT = ROOT / "starter" / "data" / "retrieval-benchmark"
DOCUMENT_ID = "Introduction_to_Python_Programming-WEB.pdf"


@dataclass(frozen=True)
class EvidenceFact:
    pages: tuple[int, ...]
    terms: tuple[str, ...]


# PDF physical pages; each fact is checked against the exported page text.
EVIDENCE: dict[str, tuple[EvidenceFact, ...]] = {
    "Q01": (EvidenceFact((40,), ("1982", "Guido", "CWI")),),
    "Q02": (EvidenceFact((40,), ("ABC", "teaching", "prototyping", "advanced")),),
    "Q03": (EvidenceFact((21,), ("print()", "end option", "newline character")),),
    "Q04": (EvidenceFact((51,), ("int(5.9)", "fractional part", "removed")),),
    "Q05": (EvidenceFact((49,), ("REPL", "read-eval", "read/input", "Evaluate/run", "Loop back")),),
    "Q06": (EvidenceFact((92,), ("Tuples are immutable", "fixed size", "less memory", "faster")),),
    "Q07": (EvidenceFact((92,), ("immutable object", "mutable object", "tuples are immutable", "lists are mutable")),),
    "Q08": (EvidenceFact((119,), ("conditional expression", "checking the condition", "expression_if_true", "expression_if_false")),),
    "Q09": (EvidenceFact((143,), ("loop else", "completed without", "break statement")),),
    "Q10": (
        EvidenceFact((137, 143), ("break statement", "exit the loop")),
        EvidenceFact((139, 143), ("continue statement", "remainder of the loop")),
    ),
    "Q11": (EvidenceFact((160,), ("function argument", "function parameter")),),
    "Q12": (EvidenceFact((162,), ("pass-by-object-reference", "mutable object", "outside the function")),),
    "Q13": (EvidenceFact((196,), ("__name__", "executed as a program", "imported by another program")),),
    "Q14": (EvidenceFact((180,), ("name collision", "most recent definition")),),
    "Q15": (
        EvidenceFact((238,), ("duplicate key values",)),
        EvidenceFact((255,), ("keys", "immutable")),
    ),
    "Q16": (EvidenceFact((262,), ("__init__", "new instance", "self refers")),),
    "Q17": (
        EvidenceFact((297,), ("is-a relationship", "subclass", "superclass")),
        EvidenceFact((298,), ("has-a relationship", "contains another class")),
    ),
    "Q18": (EvidenceFact((324,), ("write mode", "overwritten", "append mode", "appending")),),
    "Q19": (
        EvidenceFact((344,), ("NumPy", "homogeneous")),
        EvidenceFact((346,), ("DataFrame", "heterogeneous")),
    ),
    "Q20": (EvidenceFact((73,), ("str(cost)",)),),
}


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).casefold()


def evidence_satisfied(
    chunks: list[dict[str, Any]], facts: tuple[EvidenceFact, ...]
) -> bool:
    for fact in facts:
        page_text = _normalize(
            " ".join(
                chunk["text"]
                for chunk in chunks
                if chunk["source_id"] == DOCUMENT_ID and chunk["page"] in fact.pages
            )
        )
        if not all(_normalize(term) in page_text for term in fact.terms):
            return False
    return True


def reciprocal_rank_fusion(
    dense: list[dict[str, Any]],
    lexical: list[dict[str, Any]],
    *,
    rank_constant: int = 60,
    top_k: int = 3,
) -> list[dict[str, Any]]:
    scores: dict[str, float] = {}
    chunks_by_id: dict[str, dict[str, Any]] = {}
    for ranking in (dense, lexical):
        for rank, chunk in enumerate(ranking, start=1):
            chunk_id = chunk["id"]
            chunks_by_id[chunk_id] = chunk
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (rank_constant + rank)
    ranked = sorted(scores, key=lambda chunk_id: (-scores[chunk_id], chunk_id))
    return [
        {**chunks_by_id[chunk_id], "score": scores[chunk_id]}
        for chunk_id in ranked[:top_k]
    ]


def _dense_search(
    embedder: Any,
    vector_store: Any,
    query: str,
    *,
    strategy: str = "A_fixed",
    filters: dict[str, Any] | None = None,
    top_k: int,
) -> list[dict[str, Any]]:
    vector = embedder.embed_query(query)
    hits = vector_store.search(strategy, vector, top_k=top_k, filters=filters)
    return [
        {
            "id": hit["chunk_id"],
            "source_id": hit["metadata"]["document_id"],
            "source": hit["metadata"]["source"],
            "page": hit["metadata"]["page"],
            "text": hit["text"],
            "score": hit["score"],
            "metadata": hit["metadata"],
        }
        for hit in hits
    ]


def infer_section_filter(question: str) -> dict[str, Any] | None:
    match = re.search(r"\b(?:section|phần)\s+(\d+)\.(\d+)\b", question, re.IGNORECASE)
    if not match:
        return None
    chapter, section = match.groups()
    return {"chapter": int(chapter), "section": f"{chapter}.{section}"}


def _evaluate_method(
    method: str,
    questions: list[dict[str, Any]],
    lexical_index: TfidfIndex,
    embedder: Any,
    vector_store: Any,
    *,
    budget: int,
    candidate_k: int,
    top_k: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    traces = []
    elapsed_values = []
    for question in questions:
        started = time.perf_counter()
        search_filters = None
        if method == "tfidf":
            results = lexical_index.search(question["question"], top_k=top_k)
        elif method in ("dense", "dense_b", "dense_c_filter"):
            strategy = {
                "dense": "A_fixed",
                "dense_b": "B_paragraph",
                "dense_c_filter": "C_paragraph_filter",
            }[method]
            if method == "dense_c_filter":
                search_filters = infer_section_filter(question["question"])
            results = _dense_search(
                embedder,
                vector_store,
                question["question"],
                strategy=strategy,
                filters=search_filters,
                top_k=top_k,
            )
        elif method in ("hybrid", "hybrid_b"):
            strategy = "B_paragraph" if method == "hybrid_b" else "A_fixed"
            results = reciprocal_rank_fusion(
                _dense_search(embedder, vector_store, question["question"],
                              strategy=strategy, top_k=candidate_k),
                lexical_index.search(question["question"], top_k=candidate_k),
                top_k=top_k,
            )
        else:
            raise ValueError(f"Unknown retrieval method: {method}")
        elapsed_ms = (time.perf_counter() - started) * 1000
        elapsed_values.append(elapsed_ms)
        context, context_ids = pack_context(results, budget=budget)
        context_chunks = [chunk for chunk in results if chunk["id"] in context_ids]
        facts = EVIDENCE[question["id"]]
        hit1 = evidence_satisfied(results[:1], facts)
        hit3 = evidence_satisfied(results[:3], facts)
        context_hit = evidence_satisfied(context_chunks, facts)
        traces.append(
            {
                "query_id": question["id"],
                "split": question["split"],
                "question": question["question"],
                "method": method,
                "filters": search_filters,
                "top_k": top_k,
                "ranking_cutoff": 3,
                "candidate_k": candidate_k if method == "hybrid" else 3,
                "budget_chars": budget,
                "results": results,
                "context": context,
                "context_ids": context_ids,
                "elapsed_ms": elapsed_ms,
                "hit1": hit1,
                "hit3": hit3,
                "context_hit": context_hit,
            }
        )
    summary = {
        "method": method,
        "split": questions[0]["split"],
        "n": len(traces),
        "hit1": sum(trace["hit1"] for trace in traces),
        "hit3": sum(trace["hit3"] for trace in traces),
        "context_hit": sum(trace["context_hit"] for trace in traces),
        "median_ms": statistics.median(elapsed_values),
        "avg_context_chars": statistics.mean(len(trace["context"]) for trace in traces),
        "service_cost_usd": 0,
    }
    return traces, summary


def run_benchmark(
    *,
    chunks_path: Path,
    questions_path: Path,
    day6_starter: Path,
    output: Path,
    split: str,
    methods: list[str],
    budget: int,
    candidate_k: int,
    top_k: int,
) -> list[dict[str, Any]]:
    all_chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    questions = [
        question
        for question in json.loads(questions_path.read_text(encoding="utf-8"))
        if question.get("split") == split
    ]
    if not questions:
        raise ValueError(f"No questions found for split {split}")
    missing = [question["id"] for question in questions if question["id"] not in EVIDENCE]
    if missing:
        raise ValueError(f"Missing physical-page evidence labels: {missing}")

    lexical_chunks = [
        chunk
        for chunk in all_chunks
        if chunk.get("metadata", {}).get("day6_strategy") == "A_fixed"
    ]
    if not lexical_chunks:
        raise ValueError("No A_fixed chunks found")
    lexical_index = TfidfIndex(lexical_chunks)
    embedder = None
    vector_store = None
    if any(method != "tfidf" for method in methods):
        import_path = str(day6_starter.resolve())
        if import_path not in sys.path:
            sys.path.insert(0, import_path)
        from src.embeddings import MyEmbedder
        from src.store import PgVectorStore

        settings = dotenv_values(ROOT / "starter" / ".env")
        database_url = settings.get("DATABASE_URL")
        if not database_url:
            raise RuntimeError("DATABASE_URL is missing from starter/.env")
        embedder = MyEmbedder("paraphrase-multilingual-MiniLM-L12-v2")
        vector_store = PgVectorStore(database_url, dimensions=embedder.dimensions)

    output.mkdir(parents=True, exist_ok=True)
    all_summaries = []
    all_traces = []
    for method in methods:
        traces, summary = _evaluate_method(
            method,
            questions,
            lexical_index,
            embedder,
            vector_store,
            budget=budget,
            candidate_k=candidate_k,
            top_k=top_k,
        )
        all_traces.extend(traces)
        all_summaries.append(summary)
    (output / "traces.json").write_text(
        json.dumps(all_traces, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output / "summary.json").write_text(
        json.dumps(
            {
                "split": split,
                "budget_chars": budget,
                "methods": all_summaries,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return all_summaries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("dev", "test"), required=True)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=("tfidf", "dense", "dense_b", "dense_c_filter", "hybrid", "hybrid_b"),
        default=("tfidf", "dense", "hybrid"),
    )
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS)
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--day6-starter", type=Path, default=DEFAULT_DAY6_STARTER)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--budget", type=int, default=3200)
    parser.add_argument("--candidate-k", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()
    summaries = run_benchmark(
        chunks_path=args.chunks,
        questions_path=args.questions,
        day6_starter=args.day6_starter,
        output=args.output,
        split=args.split,
        methods=args.methods,
        budget=args.budget,
        candidate_k=args.candidate_k,
        top_k=args.top_k,
    )
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
