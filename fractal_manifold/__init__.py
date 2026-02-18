"""
Fractal Manifold - A Categorical Framework for Predictive Analysis

A practical system with rigorous mathematical foundation that predicts 
rather than just describes phenomena through categorical theory.
"""

__version__ = "0.1.0"

from .categorical import Category, Functor, NaturalTransformation
from .symmetry import SymmetryAnalyzer
from .entanglement import EntanglementAnalyzer
from .conservation import ConservationLawAnalyzer
from .spectral import SpectralSequence
from .renormalization import RenormalizationGroup
from .predictor import PhenomenonPredictor

__all__ = [
    "Category",
    "Functor", 
    "NaturalTransformation",
    "SymmetryAnalyzer",
    "EntanglementAnalyzer",
    "ConservationLawAnalyzer",
    "SpectralSequence",
    "RenormalizationGroup",
    "PhenomenonPredictor",
]
