"""
CLI: load a KG JSON file into Neo4j.

Usage
-----
    python scripts/load_to_neo4j.py data/output/results.json
    python scripts/load_to_neo4j.py data/output/results.json \\
        --uri bolt://localhost:7687 --user neo4j --password secret
"""

import argparse
import json
import sys
from pathlib import Path

from neo4j.exceptions import ServiceUnavailable, AuthError

from graphlens.neo4j_loader import KGLoader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load an extracted KG JSON file into Neo4j."
    )
    parser.add_argument("json_file", help="Path to the KG JSON file.")
    parser.add_argument(
        "--uri", default="bolt://localhost:7687",
        help="Neo4j Bolt URI  (default: bolt://localhost:7687).",
    )
    parser.add_argument(
        "--user", default="neo4j",
        help="Neo4j username  (default: neo4j).",
    )
    parser.add_argument(
        "--password", default="password",
        help="Neo4j password  (default: password).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    json_path = Path(args.json_file)
    if not json_path.is_file():
        print(f"Error: file not found: {json_path}", file=sys.stderr)
        sys.exit(1)

    data = json.loads(json_path.read_text(encoding="utf-8"))

    print(f"Connecting to Neo4j at {args.uri} …")
    try:
        with KGLoader(args.uri, args.user, args.password) as loader:
            loader.verify_connection()
            print("  Connected.\n")
            nodes, rels = loader.load(data)
    except ServiceUnavailable:
        print(
            "Error: cannot reach Neo4j — check the URI and that the server is running.",
            file=sys.stderr,
        )
        sys.exit(1)
    except AuthError:
        print(
            "Error: authentication failed — check --user / --password.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Done.  Merged {nodes} entity nodes and {rels} relations.")
    print(
        "\nNeo4j Browser — useful queries:\n"
        "  MATCH (n:Entity)-[r]->(m) RETURN n, r, m\n"
        "  MATCH (n:Entity {type: 'PERSON'}) RETURN n\n"
    )


if __name__ == "__main__":
    main()