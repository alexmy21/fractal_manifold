"""
NitroSAT Package for fractal_manifold
=====================================

A Python wrapper for the NitroSAT physics-informed MaxSAT solver.

This package provides:
- `NitroSatSolver`: High-level solver interface
- `CNFFormula`: CNF formula construction
- `DisambiguationProblem`: Token disambiguation as SAT
- `EntanglementProblem`: Perceptron entanglement detection as SAT

Quick Start:
    >>> from nitrosat import NitroSatSolver, CNFFormula
    >>> 
    >>> # Create a simple formula
    >>> formula = CNFFormula(3).add_clauses([
    ...     [1, 2],      # x1 OR x2
    ...     [-1, 3],     # NOT x1 OR x3
    ...     [2, -3],     # x2 OR NOT x3
    ... ])
    >>> 
    >>> # Solve
    >>> solver = NitroSatSolver()
    >>> result = solver.solve(formula)
    >>> print(f"Solved: {result.solved}")

Build the library first:
    cd nitrosat
    python build.py

Based on NitroSAT by Sethu Iyer (sethuiyer95@gmail.com)
"""

from .nitrosat import (
    NitroSatSolver,
    NitroSatResult,
    CNFFormula,
    TopologyInfo,
    solve,
    solve_file,
)

from .hllset_sat import (
    TokenVarMap,
    DisambiguationProblem,
    EntanglementProblem,
    optimal_set_cover,
)

__version__ = "1.0.0"
__author__ = "fractal_manifold project"
__credits__ = "Based on NitroSAT by Sethu Iyer"

__all__ = [
    # Core solver
    "NitroSatSolver",
    "NitroSatResult",
    "CNFFormula",
    "TopologyInfo",
    "solve",
    "solve_file",
    # HLLSet integration
    "TokenVarMap",
    "DisambiguationProblem",
    "EntanglementProblem",
    "optimal_set_cover",
]
