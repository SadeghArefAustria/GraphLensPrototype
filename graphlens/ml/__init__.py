"""graphlens.ml — machine-learning utilities for knowledge graphs."""

from graphlens.ml.graph_builder import KGGraph
from graphlens.ml.link_prediction import HeuristicPredictor, PyKEENPredictor

__all__ = ["KGGraph", "HeuristicPredictor", "PyKEENPredictor"]