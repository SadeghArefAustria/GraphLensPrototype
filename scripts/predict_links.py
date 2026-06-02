"""
CLI: run link prediction on a KG JSON file.

Usage
-----
    # Heuristic baseline (no ML deps needed)
    python scripts/predict_links.py data/output/results.json

    # Merge multiple extractions before predicting
    python scripts/predict_links.py data/output/doc1.json data/output/doc2.json

    # Choose heuristic method
    python scripts/predict_links.py results.json --method jaccard

    # Use a PyKEEN embedding model  (requires: pip install graphlens[ml])
    python scripts/predict_links.py results.json --model TransE --epochs 200

    # Save the trained model
    python scripts/predict_links.py results.json --model RotatE --save models/rotate

    # Predict tails for a specific (head, relation) pair
    python scripts/predict_links.py results.json --model TransE \\
        --predict-head "TU Graz" --predict-relation PARTNERED_WITH
"""

import argparse
import json
import sys
from pathlib import Path

from graphlens.ml.graph_builder import KGGraph
from graphlens.ml.link_prediction import HeuristicPredictor, PyKEENPredictor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def print_metrics(metrics: dict, label: str) -> None:
    print(f"\n── {label} ──")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:<16} {v:.4f}")
        else:
            print(f"  {k:<16} {v}")


def print_stats(kg: KGGraph) -> None:
    s = kg.stats()
    print("\nGraph stats:")
    print(f"  Entities  : {s['num_entities']}")
    print(f"  Relations : {s['num_relations']}")
    print(f"  Triples   : {s['num_triples']}")
    print(f"  Density   : {s['density']:.4f}")
    print(f"  Entity types   : {', '.join(s['entity_types'])}")
    print(f"  Relation types : {', '.join(s['relation_types'])}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Link prediction on a GraphLens KG JSON file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "json_files", nargs="+",
        help="One or more KG JSON files (merged before training).",
    )
    parser.add_argument(
        "--test-size", type=float, default=0.2,
        help="Fraction of triples held out for evaluation (default: 0.2).",
    )
    parser.add_argument(
        "--valid-size", type=float, default=0.0,
        help="Fraction of triples used for validation (PyKEEN only, default: 0).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42).",
    )

    heuristic = parser.add_argument_group("Heuristic predictor")
    heuristic.add_argument(
        "--method",
        choices=["common_neighbors", "jaccard", "adamic_adar"],
        default="common_neighbors",
        help="Graph-structural scoring method (default: common_neighbors).",
    )

    embedding = parser.add_argument_group("Embedding predictor (requires pykeen)")
    embedding.add_argument(
        "--model",
        metavar="MODEL_NAME",
        help="PyKEEN model to train (e.g. TransE, RotatE, DistMult). "
             "If omitted, only the heuristic predictor runs.",
    )
    embedding.add_argument(
        "--epochs", type=int, default=100,
        help="Training epochs for the embedding model (default: 100).",
    )
    embedding.add_argument(
        "--dim", type=int, default=64,
        help="Embedding dimension (default: 64).",
    )
    embedding.add_argument(
        "--device", default="cpu",
        help="Torch device: 'cpu' or 'cuda' (default: cpu).",
    )
    embedding.add_argument(
        "--save", metavar="DIR",
        help="Save the trained embedding model to this directory.",
    )

    predict = parser.add_argument_group("Tail prediction for a specific pair")
    predict.add_argument("--predict-head",     metavar="ENTITY")
    predict.add_argument("--predict-relation", metavar="RELATION")
    predict.add_argument(
        "--top-k", type=int, default=10,
        help="Number of predicted tails to show (default: 10).",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ------------------------------------------------------------------
    # Build graph (merge if multiple files)
    # ------------------------------------------------------------------
    graphs = [KGGraph.from_json(p) for p in args.json_files]
    kg     = KGGraph.merge(graphs) if len(graphs) > 1 else graphs[0]

    print_stats(kg)

    if kg.num_entities < 3 or len(kg.triples) < 3:
        print(
            "\nWarning: graph is very small — link prediction metrics "
            "will not be reliable. Extract more documents first.",
            file=sys.stderr,
        )

    # ------------------------------------------------------------------
    # Train / test split
    # ------------------------------------------------------------------
    if args.valid_size > 0:
        train_triples, valid_triples, test_triples = kg.train_test_split(
            test_size=args.test_size,
            valid_size=args.valid_size,
            seed=args.seed,
        )
    else:
        train_triples, test_triples = kg.train_test_split(
            test_size=args.test_size,
            seed=args.seed,
        )
        valid_triples = None

    print(
        f"\nSplit: {len(train_triples)} train / "
        + (f"{len(valid_triples)} valid / " if valid_triples else "")
        + f"{len(test_triples)} test triples."
    )

    # ------------------------------------------------------------------
    # Heuristic predictor (always runs)
    # ------------------------------------------------------------------
    print(f"\nRunning heuristic predictor ({args.method}) …")
    heuristic = HeuristicPredictor(kg, method=args.method)
    h_metrics  = heuristic.evaluate(test_triples, train_triples)
    print_metrics(h_metrics, f"Heuristic ({args.method})")

    # ------------------------------------------------------------------
    # Embedding predictor (optional)
    # ------------------------------------------------------------------
    embedding_predictor: PyKEENPredictor | None = None

    if args.model:
        print(f"\nTraining {args.model} embedding model …")
        embedding_predictor = PyKEENPredictor(
            kg,
            model_name=args.model,
            epochs=args.epochs,
            embedding_dim=args.dim,
            device=args.device,
        )
        embedding_predictor.train(train_triples, valid_triples)
        e_metrics = embedding_predictor.evaluate(test_triples)
        print_metrics(e_metrics, f"{args.model} (embedding)")

        if args.save:
            embedding_predictor.save(args.save)

    # ------------------------------------------------------------------
    # Tail prediction for a specific (head, relation) pair
    # ------------------------------------------------------------------
    if args.predict_head and args.predict_relation:
        print(
            f"\nTop-{args.top_k} predicted tails for "
            f"({args.predict_head!r}, {args.predict_relation!r}) → ?"
        )

        if embedding_predictor is not None:
            print(f"  [{args.model}]")
            for entity, score in embedding_predictor.predict_tails(
                args.predict_head, args.predict_relation, top_k=args.top_k
            ):
                print(f"    {score:+.3f}  {entity}")
        else:
            print(f"  [Heuristic – {args.method}]")
            for entity, score in heuristic.predict_tails(
                args.predict_head, top_k=args.top_k
            ):
                print(f"    {score:.3f}  {entity}")


if __name__ == "__main__":
    main()