"""
Vector store setup for semantic search.

Originally this used Chroma/Chromadb, but that pulled in heavy native
dependencies (like `onnxruntime`) that are not yet available for
Python 3.14, which caused pip backtracking / install failures.

This version implements a lightweight in-memory vector store using
Google embeddings + NumPy cosine similarity so the project runs
everywhere without extra system dependencies.
"""

from typing import List, Dict, Any

import numpy as np
from langchain_google_genai import GoogleGenerativeAIEmbeddings


class EventVectorStore:
    """Simple in-memory vector store for event templates."""

    def __init__(self):
        """Initialize the vector store."""
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        # Each item: {"embedding": np.ndarray, "text": str, "metadata": dict}
        self._items: List[Dict[str, Any]] = []

    def add_templates(self, templates: List[Dict[str, Any]]):
        """
        Add event templates to the vector store.

        Args:
            templates: List of dictionaries with 'text', 'metadata', and 'id' keys
        """
        if not templates:
            return

        texts = [t["text"] for t in templates]
        metadatas = [t.get("metadata", {}) for t in templates]

        # Embed all template texts in one batch
        vectors = self.embeddings.embed_documents(texts)

        for text, metadata, vec in zip(texts, metadatas, vectors):
            self._items.append(
                {
                    "text": text,
                    "metadata": metadata,
                    "embedding": np.array(vec, dtype=float),
                }
            )

    def search(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """
        Search for similar event templates.

        Args:
            query: Search query
            k: Number of results to return

        Returns:
            List of dictionaries with 'text', 'metadata', and 'score'
        """
        if not self._items:
            return []

        query_vec = np.array(self.embeddings.embed_query(query), dtype=float)

        # Precompute norm of query
        query_norm = np.linalg.norm(query_vec) or 1.0

        scored: List[Dict[str, Any]] = []
        for item in self._items:
            vec = item["embedding"]
            denom = (np.linalg.norm(vec) or 1.0) * query_norm
            score = float(np.dot(query_vec, vec) / denom)
            scored.append(
                {
                    "text": item["text"],
                    "metadata": item["metadata"],
                    "score": score,
                }
            )

        # Highest cosine similarity first
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:k]

    def get_relevant_templates(self, event_type: str, query: str = None) -> List[Dict[str, Any]]:
        """
        Get relevant templates for a specific event type.

        Args:
            event_type: Type of event (birthday, corporate, etc.)
            query: Optional additional search query

        Returns:
            List of relevant templates
        """
        search_query = f"{event_type} event planning"
        if query:
            search_query = f"{search_query} {query}"

        return self.search(search_query, k=5)


def create_sample_templates() -> List[Dict[str, Any]]:
    """Create sample event templates for the vector store."""
    templates = [
        {
            "id": "birthday_1",
            "text": "Birthday party for 30 people: Budget ₹20,000. Menu includes biryani, kebabs, cake. Timeline: 6 PM welcome, 7 PM dinner, 8 PM cake cutting, 9 PM music. Decorations: balloons, banners, fairy lights.",
            "metadata": {"event_type": "birthday_party", "guest_count": 30, "budget_range": "15000-25000"}
        },
        {
            "id": "corporate_1",
            "text": "Corporate dinner for 50 people: Budget ₹50,000. Formal setting with multi-course meal. Timeline: 7 PM cocktails, 8 PM dinner, 9:30 PM speeches, 10 PM networking. Venue: hotel banquet hall. Professional decor.",
            "metadata": {"event_type": "corporate_event", "guest_count": 50, "budget_range": "40000-60000"}
        },
        {
            "id": "baby_shower_1",
            "text": "Baby shower for 25 people: Budget ₹15,000. Light refreshments, games, gifts. Timeline: 3 PM welcome, 3:30 PM games, 5 PM cake, 6 PM gifts. Decorations: pastel colors, baby-themed items.",
            "metadata": {"event_type": "baby_shower", "guest_count": 25, "budget_range": "10000-20000"}
        },
        {
            "id": "farewell_1",
            "text": "Farewell party for 20 people: Budget ₹12,000. Casual setting with snacks and drinks. Timeline: 6 PM gathering, 7 PM speeches, 8 PM dinner, 9 PM music. Venue: restaurant or home.",
            "metadata": {"event_type": "farewell_party", "guest_count": 20, "budget_range": "10000-15000"}
        },
        {
            "id": "anniversary_1",
            "text": "Anniversary celebration for 40 people: Budget ₹30,000. Romantic theme with fine dining. Timeline: 7 PM cocktails, 8 PM dinner, 9 PM dance, 10 PM cake. Decorations: flowers, candles, elegant setup.",
            "metadata": {"event_type": "anniversary", "guest_count": 40, "budget_range": "25000-35000"}
        },
        {
            "id": "wedding_1",
            "text": "Wedding reception for 100 people: Budget ₹2,00,000. Grand celebration with full catering, decorations, music. Timeline: 6 PM arrival, 7 PM ceremony, 8 PM dinner, 9 PM dance. Multiple food stations, professional decor.",
            "metadata": {"event_type": "wedding", "guest_count": 100, "budget_range": "150000-250000"}
        }
    ]
    return templates

