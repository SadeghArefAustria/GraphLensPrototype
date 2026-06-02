"""
Link prediction on a KGGraph.

Two families of predictors are available:

``HeuristicPredictor``
    Graph-structural scores (Common Neighbors, Jaccard, Adamic-Adar).
    No ML dependencies — works with the base install.
    Ignores relation types; useful as a fast baseline.

``PyKEENPredictor``
    Knowledge-graph embedding models (TransE, RotatE, DistMult, …) via PyKEEN.
    Requires: ``pip install graphlens[ml]``
    Respects relation types; better for larger, multi-relational graphs.

Evaluation metric
-----------------
Both predictors use the standard *filtered ranking* protocol:

    For every test triple (h, r, t):
    1. Score every entity e as a candidate tail: score(h, r, e).
    2. Remove other known true tails from the ranking (filtered setting).
    3. Find the rank of the true tail t.

Reported metrics: MRR, Hits@1, Hits@3, Hits@10.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import networkx as nx
import numpy as np

if TYPE_CHECKING:
    from graphlens.ml.graph_builder import KGGraph, Triple


# ---------------------------------------------------------------------------
# Shared evaluation helper
# ---------------------------------------------------------------------------

def _ranking_metrics(ranks: list[int]) -> dict:
    """Compute MRR and Hits@K from a list of 1-based ranks."""
    if not ranks:
        return {"mrr": 0.0, "hits@1": 0.0, "hits@3": 0.0, "hits@10": 0.0}
    arr = np.array(ranks, dtype=float)
    return {
        "mrr":    float(np.mean(1.0 / arr)),
        "hits@1": float(np.mean(arr <= 1)),
        "hits@3": float(np.mean(arr <= 3)),
        "hits@10": float(np.mean(arr <= 10)),
        "num_test_triples": len(ranks),
    }


# ---------------------------------------------------------------------------
# HeuristicPredictor
# ---------------------------------------------------------------------------

class HeuristicPredictor:
    """Link prediction using graph-structural heuristics.

    Scores are computed on the *undirected* projection of the graph and
    therefore ignore relation types.  This is useful as a fast baseline
    before investing in embedding models.

    Parameters
    ----------
    kg :    The :class:`~graphlens.ml.graph_builder.KGGraph` to predict on.
    method: One of ``"common_neighbors"``, ``"jaccard"``, ``"adamic_adar"``.
    """

    METHODS = ("common_neighbors", "jaccard", "adamic_adar")

    def __init__(self, kg: "KGGraph", method: str = "common_neighbors") -> None:
        if method not in self.METHODS:
            raise ValueError(f"method must be one of {self.METHODS}")
        self.kg     = kg
        self.method = method
        self._undirected: nx.Graph = kg.graph.to_undirected()
        self._score_matrix: np.ndarray | None = None  # computed lazily

    # ------------------------------------------------------------------
    # Score matrix
    # ------------------------------------------------------------------

    def _build_score_matrix(self) -> np.ndarray:
        """Return an (E, E) float matrix where entry [i, j] = score(i → j)."""
        n   = self.kg.num_entities
        mat = np.zeros((n, n), dtype=float)
        G   = self._undirected
        nodes = list(G.nodes())

        if self.method == "common_neighbors":
            scores_iter = nx.common_neighbors  # (G, u, v) → count
            for u in nodes:
                uid = self.kg.entity_to_id[u]
                for v in nodes:
                    if u != v:
                        mat[uid, self.kg.entity_to_id[v]] = sum(
                            1 for _ in nx.common_neighbors(G, u, v)
                        )

        elif self.method == "jaccard":
            for u, v, p in nx.jaccard_coefficient(G):
                uid = self.kg.entity_to_id.get(u)
                vid = self.kg.entity_to_id.get(v)
                if uid is not None and vid is not None:
                    mat[uid, vid] = p
                    mat[vid, uid] = p  # symmetric

        elif self.method == "adamic_adar":
            for u, v, p in nx.adamic_adar_index(G):
                uid = self.kg.entity_to_id.get(u)
                vid = self.kg.entity_to_id.get(v)
                if uid is not None and vid is not None:
                    mat[uid, vid] = p
                    mat[vid, uid] = p

        return mat

    def score_matrix(self) -> np.ndarray:
        """Return (and cache) the (E, E) score matrix."""
        if self._score_matrix is None:
            self._score_matrix = self._build_score_matrix()
        return self._score_matrix

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_tails(
        self,
        head: str,
        top_k: int = 10,
    ) -> list[tuple[str, float]]:
        """Return the top-k predicted tails for *head*, sorted by score desc."""
        if head not in self.kg.entity_to_id:
            raise KeyError(f"Unknown entity: {head!r}")
        hid   = self.kg.entity_to_id[head]
        mat   = self.score_matrix()
        row   = mat[hid]
        order = np.argsort(-row)
        return [
            (self.kg.id_to_entity[i], float(row[i]))
            for i in order[:top_k]
            if i != hid
        ]

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        test_triples:  list["Triple"],
        train_triples: list["Triple"] | None = None,
    ) -> dict:
        """Evaluate on *test_triples* using the filtered ranking protocol.

        Parameters
        ----------
        test_triples:  Triples to evaluate.
        train_triples: Known true triples to filter out from rankings.
                       If ``None``, all triples in the graph are used.
        """
        mat = self.score_matrix()
        n   = self.kg.num_entities

        # Build set of all known true tails per head (for filtering)
        known: dict[str, set[int]] = {e: set() for e in self.kg.entity_to_id}
        all_triples = (
            list(train_triples) + list(test_triples)
            if train_triples is not None
            else self.kg.triples
        )
        for t in all_triples:
            known.setdefault(t.head, set()).add(t.tail_id)

        ranks: list[int] = []
        for t in test_triples:
            row    = mat[t.head_id].copy()
            # Remove known true tails except the current one
            for tid in known.get(t.head, set()):
                if tid != t.tail_id:
                    row[tid] = -np.inf
            rank = int(np.sum(row > row[t.tail_id])) + 1
            ranks.append(rank)

        return _ranking_metrics(ranks)


# ---------------------------------------------------------------------------
# PyKEENPredictor
# ---------------------------------------------------------------------------

class PyKEENPredictor:
    """Knowledge-graph embedding predictor using the PyKEEN library.

    Supports any PyKEEN model (TransE, RotatE, DistMult, ComplEx, …).
    Respects relation types — generally outperforms heuristic methods
    on multi-relational graphs.

    Requires: ``pip install graphlens[ml]``

    Parameters
    ----------
    kg :         The :class:`~graphlens.ml.graph_builder.KGGraph`.
    model_name:  PyKEEN model class name, e.g. ``"TransE"``, ``"RotatE"``.
    epochs:      Training epochs.
    embedding_dim: Dimension of entity/relation embeddings.
    device:      ``"cpu"`` or ``"cuda"``.
    """

    def __init__(
        self,
        kg:            "KGGraph",
        model_name:    str = "TransE",
        epochs:        int = 100,
        embedding_dim: int = 64,
        device:        str = "cpu",
    ) -> None:
        self.kg            = kg
        self.model_name    = model_name
        self.epochs        = epochs
        self.embedding_dim = embedding_dim
        self.device        = device
        self._pipeline_result = None

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        train_triples: list["Triple"],
        valid_triples: list["Triple"] | None = None,
    ) -> None:
        """Train the embedding model on *train_triples*.

        Parameters
        ----------
        train_triples: Training triples.
        valid_triples: Optional validation triples (used for early stopping).
        """
        try:
            from pykeen.pipeline import pipeline
            from pykeen.triples import TriplesFactory
        except ImportError as exc:
            raise ImportError(
                "pykeen is not installed. Run: pip install graphlens[ml]"
            ) from exc

        def _make_factory(triples: list["Triple"]) -> TriplesFactory:
            arr = np.array(
                [[t.head_id, t.relation_id, t.tail_id] for t in triples],
                dtype=np.int32,
            )
            return TriplesFactory(
                mapped_triples=arr,
                entity_to_id=self.kg.entity_to_id,
                relation_to_id=self.kg.relation_to_id,
            )

        training = _make_factory(train_triples)
        validation = _make_factory(valid_triples) if valid_triples else None

        self._pipeline_result = pipeline(
            training=training,
            validation=validation,
            model=self.model_name,
            model_kwargs={"embedding_dim": self.embedding_dim},
            training_kwargs={"num_epochs": self.epochs},
            device=self.device,
            random_seed=42,
        )
        print(
            f"Trained {self.model_name} for {self.epochs} epochs "
            f"(dim={self.embedding_dim}, device={self.device})."
        )

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, test_triples: list["Triple"]) -> dict:
        """Evaluate the trained model on *test_triples*.

        Returns PyKEEN metric dict with MRR, Hits@K, etc.
        """
        if self._pipeline_result is None:
            raise RuntimeError("Call .train() before .evaluate().")

        try:
            from pykeen.triples import TriplesFactory
        except ImportError as exc:
            raise ImportError(
                "pykeen is not installed. Run: pip install graphlens[ml]"
            ) from exc

        arr = np.array(
            [[t.head_id, t.relation_id, t.tail_id] for t in test_triples],
            dtype=np.int32,
        )
        test_factory = TriplesFactory(
            mapped_triples=arr,
            entity_to_id=self.kg.entity_to_id,
            relation_to_id=self.kg.relation_to_id,
        )

        results = self._pipeline_result.model.predict_h(
            rt_batch=None  # placeholder — use pipeline evaluator below
        )
        # Use the built-in pipeline evaluator for proper filtered ranking
        metric_results = self._pipeline_result.evaluate(
            mapped_triples=test_factory.mapped_triples,
        )
        return metric_results.to_dict()

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_tails(
        self,
        head:     str,
        relation: str,
        top_k:    int = 10,
    ) -> list[tuple[str, float]]:
        """Predict the most likely tails for a given (head, relation) pair.

        Parameters
        ----------
        head:     Entity name.
        relation: Relation/predicate name.
        top_k:    Number of candidates to return.
        """
        if self._pipeline_result is None:
            raise RuntimeError("Call .train() before .predict_tails().")

        if head not in self.kg.entity_to_id:
            raise KeyError(f"Unknown entity: {head!r}")
        if relation not in self.kg.relation_to_id:
            raise KeyError(f"Unknown relation: {relation!r}")

        try:
            import torch
        except ImportError as exc:
            raise ImportError("torch is required.") from exc

        model  = self._pipeline_result.model
        h_id   = self.kg.entity_to_id[head]
        r_id   = self.kg.relation_to_id[relation]
        n      = self.kg.num_entities

        hr_batch = torch.tensor([[h_id, r_id]], dtype=torch.long)
        with torch.no_grad():
            scores = model.predict_t(hr_batch=hr_batch).squeeze().cpu().numpy()

        top_ids = np.argsort(-scores)[:top_k]
        return [
            (self.kg.id_to_entity[int(i)], float(scores[i]))
            for i in top_ids
        ]

    def save(self, directory: str) -> None:
        """Save the trained model to *directory* (PyKEEN format)."""
        if self._pipeline_result is None:
            raise RuntimeError("Nothing to save — call .train() first.")
        self._pipeline_result.save_to_directory(directory)
        print(f"Model saved to {directory}/")

    @classmethod
    def load(cls, kg: "KGGraph", directory: str, **kwargs) -> "PyKEENPredictor":
        """Load a previously saved model from *directory*."""
        try:
            from pykeen.pipeline import PipelineResult
        except ImportError as exc:
            raise ImportError(
                "pykeen is not installed. Run: pip install graphlens[ml]"
            ) from exc

        predictor = cls(kg, **kwargs)
        predictor._pipeline_result = PipelineResult.load_from_directory(directory)
        return predictor