"""
§3.4 Indexing / Search at Scale.

Below ~200-300 SKUs: flat linear scan (cheap, simple, exact).
Above that: swap in an ANN index (e.g. mobile-compatible HNSW). This
module exposes one interface (`SimilarityIndex`) so that swap doesn't
touch the pipeline. Rebuild/update happens incrementally on product
add/edit, not on every scan.
"""

from . import config, embedding, db


class SimilarityIndex:
    def __init__(self):
        self._vectors = []  # list[(product_id, vector)]
        self._dirty = True

    def rebuild(self):
        """Reload all reference vectors from DB. Call on product add/edit, not per-scan."""
        self._vectors = db.all_reference_vectors()
        self._dirty = False

    def _ensure_fresh(self):
        if self._dirty or not self._vectors:
            self.rebuild()

    def sku_count(self):
        self._ensure_fresh()
        return len({pid for pid, _ in self._vectors})

    def mark_dirty(self):
        self._dirty = True

    def query(self, vector, k=config.TOP_K_CANDIDATES):
        """
        Returns top-k (product_id, similarity) pairs.

        Below ANN_INDEX_SKU_THRESHOLD SKUs: flat scan (exact, simple).
        Above it: this is where an embedded ANN library (HNSW) would take
        over — flagged as a v1 open decision (spec §10.2) since it depends
        on expected catalog size, which isn't settled yet. The interface
        below is what a real ANN backend would need to implement instead
        of `top_k_similarity`.
        """
        self._ensure_fresh()
        if self.sku_count() > config.ANN_INDEX_SKU_THRESHOLD:
            # Placeholder: same call today, but this branch is where an
            # HNSW-backed query would be substituted once catalogs grow.
            pass
        return embedding.top_k_similarity(vector, self._vectors, k=k)
