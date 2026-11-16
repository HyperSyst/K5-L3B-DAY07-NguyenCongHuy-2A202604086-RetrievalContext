import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "starter"))
from retrieval_benchmark import (
    EvidenceFact,
    evidence_satisfied,
    infer_section_filter,
    reciprocal_rank_fusion,
)


class RetrievalBenchmarkTests(unittest.TestCase):
    def test_evidence_requires_all_facts_from_the_correct_source_and_page(self):
        facts = (
            EvidenceFact((40,), ("1982", "CWI")),
            EvidenceFact((41,), ("creator of Python",)),
        )
        chunks = [
            {"source_id": "Introduction_to_Python_Programming-WEB.pdf",
             "page": 40, "text": "In 1982, the work was done at CWI."},
            {"source_id": "Introduction_to_Python_Programming-WEB.pdf",
             "page": 41, "text": "Guido was the creator of Python."},
        ]
        self.assertTrue(evidence_satisfied(chunks, facts))
        self.assertFalse(evidence_satisfied(chunks[:1], facts))
        self.assertFalse(
            evidence_satisfied(
                [{**chunks[0], "source_id": "other.pdf"}, chunks[1]], facts
            )
        )

    def test_rrf_combines_rankings_and_deduplicates_chunks(self):
        dense = [{"id": "a", "page": 1}, {"id": "b", "page": 2}]
        lexical = [{"id": "b", "page": 2}, {"id": "c", "page": 3}]
        results = reciprocal_rank_fusion(dense, lexical, top_k=3)
        self.assertEqual([row["id"] for row in results], ["b", "a", "c"])
        self.assertEqual(len({row["id"] for row in results}), 3)

    def test_section_filter_is_inferred_only_from_question_text(self):
        self.assertEqual(
            infer_section_filter("Trong phần 14.2, chế độ append làm gì?"),
            {"chapter": 14, "section": "14.2"},
        )
        self.assertIsNone(infer_section_filter("Khi nào dùng append?"))


if __name__ == "__main__":
    unittest.main()
