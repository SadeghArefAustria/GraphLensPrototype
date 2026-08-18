"""
Neo4j knowledge-graph loader.

Typical usage
-------------
    from graphlens.neo4j_loader import KGLoader

    loader = KGLoader("bolt://localhost:7687", "neo4j", "password")
    loader.verify_connection()
    nodes, rels = loader.load(kg_dict)
    loader.close()

The loader is idempotent: running it multiple times on the same data
merges rather than duplicates nodes and relationships.
"""

from __future__ import annotations

import re

from neo4j import GraphDatabase


VALID_ENTITY_TYPES = {
    "PERSON", "ORG", "LOCATION", "EVENT", "CONCEPT", "PRODUCT", "DATE", "OTHER",
}
VALID_CYPHER_IDENTIFIER = re.compile(r"^[A-Z][A-Z0-9_]*$")


class KGLoader:
    """Load a KG dict (``{"entities": [...], "relations": [...]}``) into Neo4j."""

    def __init__(self, uri: str, user: str, password: str) -> None:
        self._driver = GraphDatabase.driver(uri, auth=(user, password))

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        self._driver.close()

    def __enter__(self) -> "KGLoader":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def verify_connection(self) -> None:
        """Raise if Neo4j is unreachable or credentials are wrong."""
        self._driver.verify_connectivity()

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load(self, data: dict) -> tuple[int, int]:
        """Merge all entities and relations into Neo4j.

        Returns ``(node_count, rel_count)``.
        """
        entities  = data.get("entities", [])
        relations = data.get("relations", [])
        self._validate_for_cypher(entities, relations)

        with self._driver.session() as session:
            session.execute_write(self._ensure_constraint)

            for entity in entities:
                session.execute_write(self._merge_entity, entity)

            for rel in relations:
                session.execute_write(self._merge_relation, rel)

        return len(entities), len(relations)

    @staticmethod
    def _validate_for_cypher(entities: list[dict], relations: list[dict]) -> None:
        """Validate the model-derived identifiers interpolated into Cypher."""
        invalid_types = sorted({e.get("type") for e in entities} - VALID_ENTITY_TYPES)
        if invalid_types:
            raise ValueError(
                "Unsupported entity type(s): " + ", ".join(map(str, invalid_types))
            )
        invalid_predicates = sorted(
            {r.get("predicate", "") for r in relations
             if not VALID_CYPHER_IDENTIFIER.fullmatch(r.get("predicate", ""))}
        )
        if invalid_predicates:
            raise ValueError(
                "Invalid relation predicate(s): " + ", ".join(map(str, invalid_predicates))
            )

    # ------------------------------------------------------------------
    # Transaction helpers (called inside execute_write)
    # ------------------------------------------------------------------

    @staticmethod
    def _ensure_constraint(tx) -> None:
        tx.run(
            "CREATE CONSTRAINT entity_name_unique IF NOT EXISTS "
            "FOR (e:Entity) REQUIRE e.name IS UNIQUE"
        )

    @staticmethod
    def _merge_entity(tx, entity: dict) -> None:
        """Merge a node by its stable :Entity/name identity, then type it.

        Entity types come from a fixed validated enum, so f-string interpolation
        is safe here.
        """
        label = entity["type"]
        cypher = f"""
        MERGE (e:Entity {{name: $name}})
        SET   e.type        = $type,
              e.description = $description,
              e:{label}
        """
        tx.run(cypher, **entity)

    @staticmethod
    def _merge_relation(tx, rel: dict) -> None:
        """Merge a typed directed edge between two entity nodes.

        Predicates are validated SCREAMING_SNAKE_CASE strings, so f-string
        interpolation is safe here.
        """
        predicate = rel["predicate"]
        source_identity = (
            "{source_file_id: $source_file_id}"
            if rel.get("source_file_id")
            else ""
        )
        cypher = f"""
        MATCH (s:Entity {{name: $subject}})
        MATCH (o:Entity {{name: $object}})
        MERGE (s)-[r:{predicate} {source_identity}]->(o)
        SET   r.source_sentence = $source_sentence,
              r.confidence      = $confidence,
              r.page            = $page,
              r.char_span       = $char_span,
              r.source_file_id  = $source_file_id,
              r.source_file_link = $source_file_link
        """
        tx.run(
            cypher,
            subject=rel["subject"],
            object=rel["object"],
            source_sentence=rel.get("source_sentence"),
            confidence=rel.get("confidence"),
            page=rel.get("page"),
            char_span=rel.get("char_span"),
            source_file_id=rel.get("source_file_id"),
            source_file_link=rel.get("source_file_link"),
        )
