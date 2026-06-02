"""
KGGraph — core data structure for ML on GraphLens knowledge graphs.

Converts the JSON produced by ``graphlens.extractor`` into:
- A NetworkX MultiDiGraph  (structural analysis, heuristic ML)
- Integer-encoded triple arrays  (embedding models, PyG, PyKEEN)

Multiple JSON files can be merged into a single graph before training::

    g1 = KGGraph.from_json("data/output/doc1.json")
    g2 = KGGraph.from_json("data/output/doc2.json")
    kg = KGGraph.merge([g1, g2])
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import NamedTuple

import networkx as nx
import numpy as np


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

class Triple(NamedTuple):
    """A single knowledge-graph triple with both string and integer IDs."""
    head: str
    relation: str
    tail: str
    head_id: int
    relation_id: int
    tail_id: int


# ---------------------------------------------------------------------------
# KGGraph
# ---------------------------------------------------------------------------

class KGGraph:
    """A knowledge graph built from GraphLens extraction output.

    Attributes
    ----------
    graph           NetworkX MultiDiGraph — nodes carry ``type`` and
                    ``description``; edges carry ``relation`` and
                    ``relation_id``.
    triples         All triples as :class:`Triple` named tuples.
    entity_to_id    ``{name: int}`` — entity vocabulary.
    id_to_entity    ``[name]``       — reverse lookup.
    relation_to_id  ``{predicate: int}`` — relation vocabulary.
    id_to_relation  ``[predicate]``  — reverse lookup.
    """

    def __init__(self, kg_dict: dict) -> None:
        self._build(kg_dict)

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_json(cls, path: str | Path) -> "KGGraph":
        """Load a KGGraph from a GraphLens JSON file."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(data)

    @classmethod
    def merge(cls, graphs: list["KGGraph"]) -> "KGGraph":
        """Merge a list of KGGraphs into one (union of entities and triples).

        Duplicate triples (same head, relation, tail) are deduplicated.
        """
        seen_entities: dict[str, dict] = {}
        seen_triples: set[tuple[str, str, str]] = set()
        merged_entities: list[dict] = []
        merged_relations: list[dict] = []

        for g in graphs:
            for name, eid in g.entity_to_id.items():
                if name not in seen_entities:
                    meta = g._entity_meta.get(name, {})
                    seen_entities[name] = meta
                    merged_entities.append(
                        {"name": name, **meta}
                    )

            for t in g.triples:
                key = (t.head, t.relation, t.tail)
                if key not in seen_triples:
                    seen_triples.add(key)
                    # Retrieve evidence if stored on the graph edge
                    edge_data = g.graph.get_edge_data(t.head, t.tail) or {}
                    evidence = ""
                    if edge_data:
                        first = next(iter(edge_data.values()))
                        evidence = first.get("evidence", "")
                    merged_relations.append(
                        {
                            "subject": t.head,
                            "predicate": t.relation,
                            "object": t.tail,
                            "evidence": evidence,
                        }
                    )

        return cls({"entities": merged_entities, "relations": merged_relations})

    # ------------------------------------------------------------------
    # Internal build
    # ------------------------------------------------------------------

    def _build(self, kg_dict: dict) -> None:
        entities  = kg_dict.get("entities", [])
        relations = kg_dict.get("relations", [])

        # Entity index
        self.entity_to_id: dict[str, int] = {}
        self.id_to_entity: list[str] = []
        self._entity_meta: dict[str, dict] = {}

        for e in entities:
            name = e["name"]
            if name not in self.entity_to_id:
                self.entity_to_id[name] = len(self.id_to_entity)
                self.id_to_entity.append(name)
            self._entity_meta[name] = {
                "type": e.get("type", "OTHER"),
                "description": e.get("description", ""),
            }

        # Relation index + triples
        self.relation_to_id: dict[str, int] = {}
        self.id_to_relation: list[str] = []
        self.triples: list[Triple] = []

        for r in relations:
            head = r["subject"]
            rel  = r["predicate"]
            tail = r["object"]

            # Entities referenced in relations but absent from the entity list
            for name in (head, tail):
                if name not in self.entity_to_id:
                    self.entity_to_id[name] = len(self.id_to_entity)
                    self.id_to_entity.append(name)

            if rel not in self.relation_to_id:
                self.relation_to_id[rel] = len(self.id_to_relation)
                self.id_to_relation.append(rel)

            self.triples.append(
                Triple(
                    head=head,
                    relation=rel,
                    tail=tail,
                    head_id=self.entity_to_id[head],
                    relation_id=self.relation_to_id[rel],
                    tail_id=self.entity_to_id[tail],
                )
            )

        # NetworkX graph
        self.graph: nx.MultiDiGraph = nx.MultiDiGraph()

        for name, eid in self.entity_to_id.items():
            meta = self._entity_meta.get(name, {})
            self.graph.add_node(
                name,
                id=eid,
                type=meta.get("type", "OTHER"),
                description=meta.get("description", ""),
            )

        for t in self.triples:
            r_data = next(
                (r for r in relations
                 if r["subject"] == t.head
                 and r["predicate"] == t.relation
                 and r["object"] == t.tail),
                {},
            )
            self.graph.add_edge(
                t.head,
                t.tail,
                key=t.relation,
                relation=t.relation,
                relation_id=t.relation_id,
                evidence=r_data.get("evidence", ""),
            )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def num_entities(self) -> int:
        return len(self.id_to_entity)

    @property
    def num_relations(self) -> int:
        return len(self.id_to_relation)

    def stats(self) -> dict:
        """Return a summary dict suitable for printing or logging."""
        n = self.num_entities
        return {
            "num_entities":   n,
            "num_relations":  self.num_relations,
            "num_triples":    len(self.triples),
            "density":        len(self.triples) / (n * (n - 1)) if n > 1 else 0.0,
            "entity_types":   sorted({m["type"] for m in self._entity_meta.values()}),
            "relation_types": self.id_to_relation,
        }

    # ------------------------------------------------------------------
    # Export formats
    # ------------------------------------------------------------------

    def to_triple_array(self) -> np.ndarray:
        """Return a (N, 3) int32 array of [head_id, relation_id, tail_id]."""
        if not self.triples:
            return np.empty((0, 3), dtype=np.int32)
        return np.array(
            [[t.head_id, t.relation_id, t.tail_id] for t in self.triples],
            dtype=np.int32,
        )

    def train_test_split(
        self,
        test_size:  float = 0.2,
        valid_size: float = 0.0,
        seed:       int   = 42,
    ) -> tuple[list[Triple], ...]:
        """Split triples into train / (optional validation) / test.

        Returns
        -------
        (train, test)            when valid_size == 0
        (train, valid, test)     when valid_size  > 0
        """
        from sklearn.model_selection import train_test_split

        labels = [t.relation_id for t in self.triples]
        stratify = labels if len(set(labels)) > 1 else None

        train, test = train_test_split(
            self.triples,
            test_size=test_size,
            random_state=seed,
            stratify=stratify,
        )

        if valid_size > 0:
            valid_frac = valid_size / (1.0 - test_size)
            train_labels = [t.relation_id for t in train]
            train, valid = train_test_split(
                train,
                test_size=valid_frac,
                random_state=seed,
                stratify=train_labels if len(set(train_labels)) > 1 else None,
            )
            return list(train), list(valid), list(test)

        return list(train), list(test)

    def to_pykeen_triples_factory(self, create_inverse_triples: bool = False):
        """Build a PyKEEN ``TriplesFactory`` from this graph.

        Requires the ``ml`` extra: ``pip install graphlens[ml]``
        """
        try:
            from pykeen.triples import TriplesFactory
        except ImportError as exc:
            raise ImportError(
                "pykeen is not installed. Run: pip install graphlens[ml]"
            ) from exc

        return TriplesFactory(
            mapped_triples=self.to_triple_array(),
            entity_to_id=self.entity_to_id,
            relation_to_id=self.relation_to_id,
            create_inverse_triples=create_inverse_triples,
        )

    def to_pyg_data(self):
        """Build a PyTorch Geometric ``Data`` object (homogeneous graph).

        Node features: one-hot encoded entity type.
        Edge indices: head → tail for every triple.
        Edge attributes: integer relation type.

        Requires: ``pip install torch torch-geometric``
        """
        try:
            import torch
            from torch_geometric.data import Data
        except ImportError as exc:
            raise ImportError(
                "torch and torch_geometric are not installed. "
                "See https://pytorch-geometric.readthedocs.io for install instructions."
            ) from exc

        all_types = sorted({m["type"] for m in self._entity_meta.values()})
        type_to_idx = {t: i for i, t in enumerate(all_types)}

        x = torch.zeros(self.num_entities, len(all_types))
        for name, eid in self.entity_to_id.items():
            t = self._entity_meta.get(name, {}).get("type", "OTHER")
            x[eid, type_to_idx.get(t, 0)] = 1.0

        arr = self.to_triple_array()                          # (N, 3)
        edge_index = torch.tensor(arr[:, [0, 2]].T, dtype=torch.long)   # (2, N)
        edge_attr  = torch.tensor(arr[:, 1],         dtype=torch.long)   # (N,)

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)