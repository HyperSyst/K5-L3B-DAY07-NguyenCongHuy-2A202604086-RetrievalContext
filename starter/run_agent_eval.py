"""Run real Gemini agent evaluations with the local TF-IDF retriever."""

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any, Protocol

from dotenv import dotenv_values
from google import genai
from google.genai import types

from agentic import run_agent
from baseline import TfidfIndex
from gemini_provider import GeminiController


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASES = ROOT / "starter" / "data" / "agent-test" / "cases.json"
DEFAULT_CHUNKS = ROOT / "starter" / "day6_chunks.json"
DEFAULT_ENV = ROOT / "starter" / ".env"
REQUEST_TIMEOUT_MS = 120_000


class ModelController(Protocol):
    def decide(self, question: str, history: list[dict[str, Any]]) -> dict[str, Any]:
        ...

    def close(self) -> None:
        ...


class TfidfRetriever:
    def __init__(self, index: TfidfIndex) -> None:
        self._index = index

    def search(
        self,
        query: str,
        *,
        filters: dict[str, Any] | None,
        top_k: int,
    ) -> list[dict[str, Any]]:
        if filters is not None:
            raise ValueError("TF-IDF baseline does not support metadata filters")
        return self._index.search(query, top_k=top_k)


class TrackedController:
    def __init__(self, controller: ModelController) -> None:
        self._controller = controller
        self.call_attempts = 0
        self.elapsed_ms = 0.0

    def decide(self, question: str, history: list[dict[str, Any]]) -> dict[str, Any]:
        started = time.perf_counter()
        self.call_attempts += 1
        try:
            return self._controller.decide(question, history)
        finally:
            self.elapsed_ms += (time.perf_counter() - started) * 1000

    def close(self) -> None:
        self._controller.close()


def _load_cases(path: Path, split: str) -> list[dict[str, Any]]:
    cases = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(cases, list):
        raise ValueError("cases phải là JSON array")
    selected = [case for case in cases if case.get("split") == split]
    if not selected:
        raise ValueError(f"Không có case trong split {split}")
    for case in selected:
        if (
            not isinstance(case.get("id"), str)
            or not isinstance(case.get("question"), str)
            or not isinstance(case.get("answerable"), bool)
        ):
            raise ValueError("Mỗi case cần id, question và answerable hợp lệ")
    return selected


def _make_error_message(error: Exception, api_key: str) -> str:
    message = str(error)
    if api_key:
        message = message.replace(api_key, "[REDACTED]")
    return message


def _write_json(path: Path, value: Any) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def _build_summary(
    traces: list[dict[str, Any]],
    *,
    expected_cases: int,
    split: str,
    model: str,
    budget: int,
    top_k: int,
    max_searches: int,
) -> dict[str, Any]:
    elapsed_values = [trace["elapsed_ms"] for trace in traces]
    return {
        "split": split,
        "retrieval_method": "tfidf",
        "model": model,
        "top_k": top_k,
        "budget_chars": budget,
        "max_searches": max_searches,
        "request_timeout_ms": REQUEST_TIMEOUT_MS,
        "cases_expected": expected_cases,
        "cases_completed": len(traces),
        "status_counts": {
            status: sum(trace["status"] == status for trace in traces)
            for status in ("answered", "abstain", "error")
        },
        "model_call_attempts": sum(
            trace["model_call_attempts"] for trace in traces
        ),
        "median_elapsed_ms": (
            statistics.median(elapsed_values) if elapsed_values else None
        ),
        "service_cost_usd": None,
        "cost_note": "Not reported by this runner; verify provider billing separately.",
        "traces": "traces.json",
    }


def run_evaluation(
    *,
    cases_path: Path,
    chunks_path: Path,
    env_path: Path,
    output: Path,
    split: str,
    budget: int,
    top_k: int,
    max_searches: int,
) -> dict[str, Any]:
    traces_path = output / "traces.json"
    summary_path = output / "summary.json"
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite evaluation output: {output}")

    cases = _load_cases(cases_path, split)
    chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    if not isinstance(chunks, list) or not chunks:
        raise ValueError("chunks phải là JSON array không rỗng")
    lexical_chunks = [
        chunk
        for chunk in chunks
        if chunk.get("metadata", {}).get("day6_strategy") == "A_fixed"
    ]
    if not lexical_chunks:
        raise ValueError("Không tìm thấy A_fixed chunks")
    index = TfidfIndex(lexical_chunks)
    retriever = TfidfRetriever(index)

    settings = dotenv_values(env_path)
    api_key = settings.get("GEMINI_API_KEY")
    model = settings.get("GEMINI_MODEL")
    if not isinstance(api_key, str) or not api_key.strip():
        raise ValueError(f"GEMINI_API_KEY is missing from {env_path}")
    if not isinstance(model, str) or not model.strip():
        raise ValueError(f"GEMINI_MODEL is missing from {env_path}")

    output.mkdir(parents=True, exist_ok=True)
    client = genai.Client(
        api_key=api_key.strip(),
        http_options=types.HttpOptions(timeout=REQUEST_TIMEOUT_MS),
    )
    controller = TrackedController(GeminiController(api_key, model, client=client))
    traces = []
    try:
        for case in cases:
            started = time.perf_counter()
            calls_before = controller.call_attempts
            model_time_before = controller.elapsed_ms
            trace: dict[str, Any] = {
                "query_id": case["id"],
                "split": split,
                "answerable": case["answerable"],
                "question": case["question"],
                "retrieval_method": "tfidf",
                "model": model,
                "top_k": top_k,
                "budget_chars": budget,
                "max_searches": max_searches,
            }
            try:
                result = run_agent(
                    case["question"],
                    controller,
                    retriever,
                    max_searches=max_searches,
                    top_k=top_k,
                    budget=budget,
                    query_id=case["id"],
                )
                trace.update(result)
            except Exception as error:
                trace.update(
                    {
                        "status": "error",
                        "error_type": type(error).__name__,
                        "error_message": _make_error_message(error, api_key),
                        "events": [],
                        "citations": [],
                    }
                )
            trace["model_call_attempts"] = controller.call_attempts - calls_before
            trace["model_elapsed_ms"] = controller.elapsed_ms - model_time_before
            trace["elapsed_ms"] = (time.perf_counter() - started) * 1000
            traces.append(trace)
            _write_json(traces_path, traces)
            summary = _build_summary(
                traces,
                expected_cases=len(cases),
                split=split,
                model=model,
                budget=budget,
                top_k=top_k,
                max_searches=max_searches,
            )
            _write_json(summary_path, summary)
            print(
                f"{case['id']}: {trace['status']}, "
                f"model calls={trace['model_call_attempts']}",
                flush=True,
            )
            if trace["status"] == "error":
                raise RuntimeError(
                    f"Agent case {case['id']} failed; partial trace saved to {traces_path}"
                )
    finally:
        controller.close()

    summary = _build_summary(
        traces,
        expected_cases=len(cases),
        split=split,
        model=model,
        budget=budget,
        top_k=top_k,
        max_searches=max_searches,
    )
    _write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run real Gemini agent cases")
    parser.add_argument("--split", choices=("dev", "test"), required=True)
    parser.add_argument("--budget", type=int, choices=(800, 3200), required=True)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--max-searches", type=int, default=4)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS)
    parser.add_argument("--env", type=Path, default=DEFAULT_ENV)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.top_k < 1 or args.max_searches < 1:
        parser.error("--top-k and --max-searches must be positive")
    run_evaluation(
        cases_path=args.cases,
        chunks_path=args.chunks,
        env_path=args.env,
        output=args.output,
        split=args.split,
        budget=args.budget,
        top_k=args.top_k,
        max_searches=args.max_searches,
    )


if __name__ == "__main__":
    main()
