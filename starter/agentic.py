"""Agent loop and provider-neutral adapters for a model and Day 6 vector store."""
import math
from collections.abc import Callable, Mapping
from numbers import Real
from typing import Any, Protocol

from baseline import pack_context


class ControllerLike(Protocol):
    def decide(self, question: str, history: list[dict[str, Any]]) -> dict[str, Any]:
        """Call a real model with a search tool; return normalized action.

        Search: {'action': 'search', 'query': '...'}.
        Final: {'action': 'answer', 'answer': '...', 'citations': ['chunk-id']}.
        Citations must also appear as [chunk-id] in the answer text.
        """
        ...


class RetrieverLike(Protocol):
    def search(self, query: str, *, filters: dict[str, Any] | None, top_k: int) -> list[dict[str, Any]]:
        """Query the real Day 6 vector store with a compatible query embedding.

        Return each chunk's id, text, source_id, source, page, metadata and score.
        The query embedding model and dimension must match the document index.
        """
        ...


class QueryEmbedder(Protocol):
    def embed_query(self, question: str) -> list[float]:
        ...


class VectorStore(Protocol):
    def search(
        self,
        strategy: str,
        query_vector: list[float],
        top_k: int = 3,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        ...


class Controller:
    """Validate decisions returned by a provider-specific real-model adapter."""

    def __init__(self, decide_with_model: Callable[[str, list[dict[str, Any]]], Mapping[str, Any]]) -> None:
        if not callable(decide_with_model):
            raise TypeError('decide_with_model phải là hàm gọi được')
        self._decide_with_model = decide_with_model

    def decide(self, question: str, history: list[dict[str, Any]]) -> dict[str, Any]:
        if not isinstance(question, str) or not question.strip():
            raise ValueError('question phải là chuỗi không rỗng')
        if not isinstance(history, list):
            raise ValueError('history phải là danh sách')

        action = self._decide_with_model(question, history)
        if not isinstance(action, Mapping):
            raise TypeError('model adapter phải trả về object action')

        kind = action.get('action')
        if kind == 'search':
            query = action.get('query')
            if not isinstance(query, str) or not query.strip():
                raise ValueError('search query phải là chuỗi không rỗng')
            return {'action': 'search', 'query': query.strip()}
        if kind == 'answer':
            answer, citations = action.get('answer'), action.get('citations')
            if not isinstance(answer, str) or not answer.strip():
                raise ValueError('answer phải là chuỗi không rỗng')
            if (not isinstance(citations, list)
                    or not citations
                    or any(not isinstance(item, str) or not item.strip() for item in citations)):
                raise ValueError('citations phải là danh sách ID không rỗng')
            return {'action': 'answer', 'answer': answer, 'citations': citations}
        raise ValueError('action phải là search hoặc answer')


class Retriever:
    """Use Day 6's embedder and vector store, mapping hits to the agent schema."""

    def __init__(
        self,
        embedder: QueryEmbedder,
        vector_store: VectorStore,
        *,
        strategy: str,
    ) -> None:
        if not callable(getattr(embedder, 'embed_query', None)):
            raise TypeError('embedder phải có phương thức embed_query')
        if not callable(getattr(vector_store, 'search', None)):
            raise TypeError('vector_store phải có phương thức search')
        if not isinstance(strategy, str) or not strategy.strip():
            raise ValueError('strategy phải là chuỗi không rỗng')
        self._embedder = embedder
        self._vector_store = vector_store
        self._strategy = strategy.strip()

    def search(self, query: str, *, filters: dict[str, Any] | None, top_k: int) -> list[dict[str, Any]]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError('query phải là chuỗi không rỗng')
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
            raise ValueError('top_k phải là số nguyên dương')
        if filters is not None and not isinstance(filters, dict):
            raise ValueError('filters phải là object hoặc None')

        raw_embedding = self._embedder.embed_query(query.strip())
        if isinstance(raw_embedding, (str, bytes, Mapping)):
            raise TypeError('embedding phải là dãy số')
        try:
            embedding = list(raw_embedding)
        except TypeError as exc:
            raise TypeError('embedding phải là dãy số') from exc
        if not embedding or any(
            isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value)
            for value in embedding
        ):
            raise ValueError('embedding phải chứa các số hữu hạn và không rỗng')
        vector = [float(value) for value in embedding]

        results = self._vector_store.search(
            strategy=self._strategy,
            query_vector=vector,
            top_k=top_k,
            filters=filters,
        )
        if not isinstance(results, list):
            raise TypeError('vector store phải trả về danh sách chunk')
        if len(results) > top_k:
            raise ValueError('vector store trả về nhiều hơn top_k chunk')

        required = ('chunk_id', 'text', 'metadata', 'score')
        seen: set[str] = set()
        normalized: list[dict[str, Any]] = []
        for chunk in results:
            if not isinstance(chunk, Mapping):
                raise TypeError('mỗi kết quả phải là object chunk')
            if any(key not in chunk or chunk[key] is None for key in required):
                raise ValueError('Day 6 hit phải có chunk_id, text, metadata và score')
            metadata = chunk['metadata']
            if not isinstance(metadata, dict):
                raise ValueError('chunk metadata phải là object')
            if any(key not in metadata or metadata[key] is None
                   for key in ('document_id', 'source', 'page')):
                raise ValueError('Day 6 metadata phải có document_id, source và page')
            if not all(isinstance(metadata[key], str) and metadata[key].strip()
                       for key in ('document_id', 'source')):
                raise ValueError('metadata document_id/source phải là chuỗi không rỗng')
            if not isinstance(metadata['page'], int) or isinstance(metadata['page'], bool) or metadata['page'] < 1:
                raise ValueError('metadata page phải là số nguyên dương')
            chunk_id = chunk['chunk_id']
            if not isinstance(chunk_id, str) or not chunk_id.strip():
                raise ValueError('chunk_id phải là chuỗi không rỗng')
            if not isinstance(chunk['text'], str) or not chunk['text'].strip():
                raise ValueError('chunk text phải là chuỗi không rỗng')
            if chunk_id in seen:
                raise ValueError('vector store trả về ID chunk trùng lặp')
            score = chunk['score']
            if isinstance(score, bool) or not isinstance(score, Real) or not math.isfinite(score):
                raise ValueError('chunk score phải là số hữu hạn')
            seen.add(chunk_id)
            normalized.append({
                'id': chunk_id,
                'text': chunk['text'],
                'source_id': metadata['document_id'],
                'source': metadata['source'],
                'page': metadata['page'],
                'metadata': metadata,
                'score': float(score),
            })
        return normalized


def run_agent(question: str, controller: ControllerLike, retriever: RetrieverLike, *,
              filters: dict[str, Any] | None = None, max_searches: int = 4,
              top_k: int = 3, budget: int = 800,
              query_id: str | None = None) -> dict[str, Any]:
    """Return answer, citation IDs and trace; no benchmark gold enters this loop.

    A final answer without valid citations, an empty context, and a search limit
    lead to abstention. This only checks citation IDs, not semantic grounding.
    Provider/retriever errors propagate; no silent mock fallback.
    """
    if not question.strip() or max_searches < 1 or top_k < 1 or budget < 1:
        raise ValueError('question, max_searches, top_k hoặc budget không hợp lệ')
    if query_id is not None and (not isinstance(query_id, str) or not query_id.strip()):
        raise ValueError('query_id phải là chuỗi không rỗng hoặc None')

    def with_query_id(result: dict[str, Any]) -> dict[str, Any]:
        if query_id is not None:
            result['query_id'] = query_id
        return result

    history: list[dict[str, Any]] = []
    chunks: dict[str, dict[str, Any]] = {}
    candidates: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    for _ in range(max_searches + 1):
        action = controller.decide(question, history)
        if action.get('action') == 'search':
            if sum(e['event'] == 'search' for e in events) >= max_searches:
                return with_query_id({'status': 'abstain', 'answer': 'Không đủ bằng chứng để trả lời.', 'citations': [], 'events': events + [{'event': 'search_limit'}]})
            query = action.get('query')
            if not isinstance(query, str) or not query.strip():
                raise ValueError('search query phải là chuỗi không rỗng')
            results = retriever.search(query, filters=filters, top_k=top_k)
            if len(results) > top_k or len({r['id'] for r in results}) != len(results):
                raise ValueError('retriever phải trả tối đa top_k chunk ID duy nhất')
            candidates.extend(results)
            context, ids = pack_context(candidates, budget=budget)
            chunks = {c['id']: c for c in candidates if c['id'] in ids}
            events.append({'event': 'search', 'query': query, 'filter': filters, 'results': results,
                           'context': context, 'context_ids': ids})
            # Send only the latest cumulative context; avoid multiplying the budget.
            history = [{'tool': 'search', 'queries': [e['query'] for e in events if e['event'] == 'search'],
                        'context': context, 'context_ids': ids}]
        elif action.get('action') == 'answer':
            answer, citations = action.get('answer'), action.get('citations')
            if not isinstance(answer, str) or not isinstance(citations, list) or any(not isinstance(i, str) for i in citations):
                raise ValueError('answer/citations không hợp lệ')
            valid = bool(answer.strip() and citations and all(i in chunks and f'[{i}]' in answer for i in citations))
            if valid:
                return with_query_id({'status': 'answered', 'answer': answer, 'citations': citations, 'events': events + [{'event': 'answer'}]})
            return with_query_id({'status': 'abstain', 'answer': 'Không đủ bằng chứng để trả lời.', 'citations': [], 'events': events + [{'event': 'invalid_or_missing_citations'}]})
        else:
            raise ValueError('action phải là search hoặc answer')
    return with_query_id({'status': 'abstain', 'answer': 'Không đủ bằng chứng để trả lời.', 'citations': [], 'events': events})
