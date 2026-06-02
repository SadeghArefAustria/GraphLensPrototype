"""
Load extracted knowledge-graph JSON into Neo4j.

Usage:
    python load_to_neo4j.py results.json
    python load_to_neo4j.py results.json --uri bolt://localhost:7687 --user neo4j --password secret

The script is idempotent: re-running it merges nodes/edges rather than
duplicating them.

Requirements:
    pip install neo4j
"""

import argparse
import json
import sys
from pathlib import Path

from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable, AuthError


# ---------------------------------------------------------------------------
# Neo4j loader
# ---------------------------------------------------------------------------

class KGLoader:
    def __init__(self, uri: str, user: str, password: str):
        self._driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self._driver.close()

    def verify_connection(self):
        self._driver.verify_connectivity()

    def load(self, data: dict) -> tuple[int, int]:
        """Load entities and relations. Returns (node_count, rel_count)."""
        entities  = data.get("entities", [])
        relations = data.get("relations", [])

        node_count = rel_count = 0

        with self._driver.session() as session:
            # ------------------------------------------------------------------
            # 1. Create / merge entity nodes.
            #    Each entity becomes a node labelled with its type AND :Entity.
            #    A uniqueness constraint on (Entity, name) makes MERGE fast and
            #    prevents duplicates even when the script is re-run.
            # ------------------------------------------------------------------
            session.execute_write(self._ensure_constraint)

            for entity in entities:
                session.execute_write(self._merge_entity, entity)
                node_count += 1

            # ------------------------------------------------------------------
            # 2. Create / merge relation edges.
            # ------------------------------------------------------------------
            for rel in relations:
                session.execute_write(self._merge_relation, rel)
                rel_count += 1

        return node_count, rel_count

    # ------------------------------------------------------------------
    # Transaction functions (called inside execute_write)
    # ------------------------------------------------------------------

    @staticmethod
    def _ensure_constraint(tx):
        # Uniqueness constraint on Entity.name so MERGE is O(log n).
        tx.run(
            "CREATE CONSTRAINT entity_name_unique IF NOT EXISTS "
            "FOR (e:Entity) REQUIRE e.name IS UNIQUE"
        )

    @staticmethod
    def _merge_entity(tx, entity: dict):
        """Merge a node with both :Entity and a type-specific label.

        Entity types come from a fixed enum in the extraction schema
        (PERSON, ORG, LOCATION, EVENT, CONCEPT, PRODUCT, OTHER), so
        interpolating them directly into the Cypher string is safe.
        """
        label = entity["type"]
        cypher = f"""
        MERGE (e:Entity:{label} {{name: $name}})
        SET   e.type        = $type,
              e.description = $description
        """
        tx.run(cypher, **entity)

    @staticmethod
    def _merge_relation(tx, rel: dict):
        """Merge a typed relationship between two entities.

        Relationship types (predicates) are SCREAMING_SNAKE_CASE strings
        produced by the extraction model, so interpolating them is safe.
        """
        predicate = rel["predicate"]
        cypher = f"""
        MATCH (s:Entity {{name: $subject}})
        MATCH (o:Entity {{name: $object}})
        MERGE (s)-[r:{predicate}]->(o)
        SET   r.evidence = $evidence
        """
        tx.run(cypher, **rel)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load KG JSON into Neo4j."
    )
    parser.add_argument("json_file", help="Path to the extracted JSON file.")
    parser.add_argument(
        "--uri", default="bolt://localhost:7687",
        help="Neo4j Bolt URI (default: bolt://localhost:7687)."
    )
    parser.add_argument(
        "--user", default="neo4j",
        help="Neo4j username (default: neo4j)."
    )
    parser.add_argument(
        "--password", default="password",
        help="Neo4j password (default: password)."
    )
    return parser.parse_args()


def main():
    args = parse_args()

    json_path = Path(args.json_file)
    if not json_path.is_file():
        print(f"Error: file not found: {json_path}", file=sys.stderr)
        sys.exit(1)

    data = json.loads(json_path.read_text(encoding="utf-8"))

    print(f"Connecting to Neo4j at {args.uri} …")
    loader = KGLoader(args.uri, args.user, args.password)

    try:
        loader.verify_connection()
        print("  Connected.\n")
    except ServiceUnavailable:
        print(
            "Error: cannot reach Neo4j. Make sure it is running and the URI is correct.",
            file=sys.stderr,
        )
        sys.exit(1)
    except AuthError:
        print(
            "Error: authentication failed. Check --user / --password.",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        nodes, rels = loader.load(data)
        print(f"Done.  Merged {nodes} entity nodes and {rels} relations into Neo4j.")
        print(
            "\nOpen Neo4j Browser and run:\n"
            "  MATCH (n:Entity) RETURN n LIMIT 50\n"
            "to see the graph."
        )
    finally:
        loader.close()


if __name__ == "__main__":
    main()