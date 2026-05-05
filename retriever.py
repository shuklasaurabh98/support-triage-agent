"""
retriever.py — TF-IDF corpus retriever with source-level filtering.

Builds a single index over all docs, supports filtering by source company.
Uses sublinear TF scaling (approximates BM25 behaviour).
"""

import os, pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Optional

CACHE_DIR = os.path.join(os.path.dirname(__file__), 'data')
INDEX_PATH = os.path.join(CACHE_DIR, 'tfidf_index.pkl')


class CorpusRetriever:
    def __init__(self):
        self.docs: List[Dict] = []
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.matrix = None          # sparse (n_docs × n_features)

    # ── Index management ──────────────────────────────────────────────────

    def build_index(self, docs: List[Dict]):
        print(f"[retriever] Building TF-IDF index over {len(docs)} docs…")
        self.docs = docs
        texts = [f"{d['title']} {d['title']} {d['content']}" for d in docs]  # title boost
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=60_000,
            sublinear_tf=True,
            min_df=1,
            stop_words='english',
        )
        self.matrix = self.vectorizer.fit_transform(texts)
        print(f"[retriever] Index ready — shape: {self.matrix.shape}")
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(INDEX_PATH, 'wb') as f:
            pickle.dump((self.docs, self.vectorizer, self.matrix), f)
        print(f"[retriever] Saved to {INDEX_PATH}")

    def load_index(self) -> bool:
        if not os.path.exists(INDEX_PATH):
            return False
        try:
            with open(INDEX_PATH, 'rb') as f:
                self.docs, self.vectorizer, self.matrix = pickle.load(f)
            print(f"[retriever] Loaded index — {len(self.docs)} docs")
            return True
        except Exception as e:
            print(f"[retriever] Load failed: {e}")
            return False

    # ── Retrieval ─────────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        top_k: int = 6,
        source_filter: Optional[str] = None,   # e.g. "HackerRank"
        min_score: float = 0.02,
    ) -> List[Dict]:
        """Return top-k docs sorted by relevance, optionally filtered by source."""
        if self.vectorizer is None:
            raise RuntimeError("Index not built.")

        qvec = self.vectorizer.transform([query])
        scores = cosine_similarity(qvec, self.matrix).flatten()

        # Zero out docs that don't match the source filter
        if source_filter:
            for i, doc in enumerate(self.docs):
                if doc['source'].lower() != source_filter.lower():
                    scores[i] = 0.0

        top_idx = np.argsort(scores)[::-1]
        results = []
        seen = set()
        for idx in top_idx:
            if scores[idx] < min_score:
                break
            doc = self.docs[idx]
            if doc['id'] not in seen:
                seen.add(doc['id'])
                results.append({**doc, 'score': float(scores[idx])})
            if len(results) >= top_k:
                break
        return results

    def retrieve_best(self, query: str, top_k: int = 6, source: Optional[str] = None) -> List[Dict]:
        """
        Smart retrieval: tries source-filtered first, falls back to global
        if filtered results are weak (top score < 0.05).
        """
        if source:
            results = self.retrieve(query, top_k=top_k, source_filter=source)
            if results and results[0]['score'] >= 0.05:
                return results
        # Fallback: global search
        return self.retrieve(query, top_k=top_k)


def get_retriever(docs: Optional[List[Dict]] = None) -> 'CorpusRetriever':
    """Return a ready retriever (from cache or freshly built)."""
    r = CorpusRetriever()
    if not r.load_index():
        if not docs:
            raise RuntimeError("No index and no docs — run --scrape + --build-index first.")
        r.build_index(docs)
    return r
