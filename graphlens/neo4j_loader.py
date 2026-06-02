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

from neo4j import GraphDatabase


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

        with self._driver.session() as session:
            session.execute_write(self._ensure_constraint)

            for entity in entities:
                session.execute_write(self._merge_entity, entity)

            for rel in relations:
                session.execute_write(self._merge_relation, rel)

        return len(entities), len(relations)

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
        """Merge a node with :Entity and its specific type label.

        Entity types come from a fixed enum (PERSON, ORG, LOCATION, EVENT,
        CONCEPT, PRODUCT, OTHER), so f-string interpolation is safe here.
        """
        label = entity["type"]
        cypher = f"""
        MERGE (e:Entity:{label} {{name: $name}})
        SET   e.type        = $type,
              e.description = $description
        """
        tx.run(cypher, **entity)

    @staticmethod
    def _merge_relation(tx, rel: dict) -> None:
        """Merge a typed directed edge between two entity nodes.

        Predicates are SCREAMING_SNAKE_CASE strings produced by the
        extraction model, so f-string interpolation is safe here.
        """
        predicate = rel["predicate"]
        cypher = f"""
        MATCH (s:Entity {{name: $subject}})
        MATCH (o:Entity {{name: $object}})
        MERGE (s)-[r:{predicate}]->(o)
        SET   r.evidence = $evidence
        """
        tx.run(cypher, **rel)