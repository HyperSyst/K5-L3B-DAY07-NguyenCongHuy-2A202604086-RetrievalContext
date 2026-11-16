"""Gemini model adapter for the Day 7 agent. Requires the google-genai SDK."""

import json
from collections.abc import Mapping
from typing import Any, Protocol

from agentic import Controller


class _GeminiResponse(Protocol):
    text: str | None


class _GeminiModels(Protocol):
    def generate_content(
        self,
        *,
        model: str,
        contents: str,
        config: dict[str, str],
    ) -> _GeminiResponse:
        ...


class _GeminiClient(Protocol):
    models: _GeminiModels

    def close(self) -> None:
        ...


class GeminiController(Controller):
    """Call Gemini and validate its JSON action using the Day 7 Controller."""

    def __init__(
        self,
        api_key: str,
        model: str,
        *,
        client: _GeminiClient | None = None,
    ) -> None:
        if not isinstance(api_key, str) or not api_key.strip():
            raise ValueError("api_key must be a non-empty string")
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be a non-empty string")

        if client is None:
            from google import genai

            client = genai.Client(api_key=api_key.strip())

        self._client = client
        self._model = model.strip()
        super().__init__(self._decide_with_gemini)

    def _decide_with_gemini(
        self, question: str, history: list[dict[str, Any]]
    ) -> Mapping[str, Any]:
        prompt = (
            "You are a retrieval question-answering agent. Decide on exactly one "
            "next action. If the search history has no supporting evidence, or "
            "more evidence is needed, return a search action. Otherwise answer "
            "only from the provided context. Cite only chunk IDs listed in "
            "context_ids, and include each cited ID in the answer as [chunk-id]. "
            "Never invent evidence or citations. Answer in the question's language. "
            'Return only a JSON object: either {"action":"search","query":"..."} '
            'or {"action":"answer","answer":"...","citations":["chunk-id"]}.\n'
            + json.dumps(
                {"question": question, "history": history},
                ensure_ascii=False,
            )
        )
        response = self._client.models.generate_content(
            model=self._model,
            contents=prompt,
            config={"response_mime_type": "application/json"},
        )
        if not response.text or not response.text.strip():
            raise RuntimeError("Gemini returned an empty response")
        try:
            action = json.loads(response.text)
        except json.JSONDecodeError as exc:
            raise ValueError("Gemini did not return valid JSON") from exc
        if not isinstance(action, dict):
            raise TypeError("Gemini response must be a JSON object")
        return action

    def close(self) -> None:
        self._client.close()
