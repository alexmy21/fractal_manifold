"""
NitroSAT Python Wrapper
=======================

A high-level Python interface to the NitroSAT MaxSAT solver.

NitroSAT is a physics-informed MaxSAT solver using:
- Spectral geometry for initialization (XOR-Laplacian)
- Heat kernel smoothing for gradient flow
- Persistent homology (Betti numbers) for topological analysis
- BAHA (Branch-Aware Holonomy Annealing) for phase transitions
- Prime-weighted clause importance (Riemann zeta connection)

This wrapper enables integration with fractal_manifold's HLLSet
operations for disambiguation, entanglement detection, and optimization.

Author: fractal_manifold project
Based on NitroSAT by Sethu Iyer (sethuiyer95@gmail.com)

Citation:
    @software{sethurathienam_iyer_2026_18753235,
        author = {Sethurathienam Iyer},
        title = {NitroSAT: A Physics-Informed MaxSAT Solver},
        year = 2026,
        publisher = {Zenodo},
        doi = {10.5281/zenodo.18753235}
    }
"""

from __future__ import annotations

import ctypes
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Sequence
import tempfile

# ============================================================================
# Library Loading
# ============================================================================

def _find_library() -> Optional[Path]:
    """Find the NitroSAT shared library."""
    lib_dir = Path(__file__).parent
    
    if sys.platform == "darwin":
        lib_path = lib_dir / "libnitrosat.dylib"
    else:
        lib_path = lib_dir / "libnitrosat.so"
    
    if lib_path.exists():
        return lib_path
    return None


def _load_library() -> ctypes.CDLL:
    """Load the NitroSAT shared library."""
    lib_path = _find_library()
    
    if lib_path is None:
        raise RuntimeError(
            "NitroSAT library not found. Build it with:\n"
            "    cd nitrosat && python build.py"
        )
    
    lib = ctypes.CDLL(str(lib_path))
    
    # Define function signatures
    
    # nitrosat_version
    lib.nitrosat_version.argtypes = []
    lib.nitrosat_version.restype = ctypes.c_char_p
    
    # nitrosat_set_seed
    lib.nitrosat_set_seed.argtypes = [ctypes.c_double]
    lib.nitrosat_set_seed.restype = None
    
    # nitrosat_solve_file
    lib.nitrosat_solve_file.argtypes = [
        ctypes.c_char_p,  # filename
        ctypes.c_int,     # max_steps
        ctypes.c_int,     # use_dcw
        ctypes.c_int,     # use_topo
        ctypes.c_int,     # verbose
    ]
    lib.nitrosat_solve_file.restype = ctypes.c_void_p
    
    # nitrosat_solve_arrays
    lib.nitrosat_solve_arrays.argtypes = [
        ctypes.c_int,                          # num_vars
        ctypes.c_int,                          # num_clauses
        ctypes.POINTER(ctypes.c_int),          # clause_sizes
        ctypes.POINTER(ctypes.c_int),          # literals
        ctypes.c_int,                          # max_steps
        ctypes.c_int,                          # use_dcw
        ctypes.c_int,                          # use_topo
        ctypes.c_int,                          # verbose
    ]
    lib.nitrosat_solve_arrays.restype = ctypes.c_void_p
    
    # nitrosat_free_result
    lib.nitrosat_free_result.argtypes = [ctypes.c_void_p]
    lib.nitrosat_free_result.restype = None
    
    return lib


# Result structure matching C struct
class _NitroSatResultStruct(ctypes.Structure):
    _fields_ = [
        ("solved", ctypes.c_int),
        ("satisfied", ctypes.c_int),
        ("unsatisfied", ctypes.c_int),
        ("num_vars", ctypes.c_int),
        ("num_clauses", ctypes.c_int),
        ("solve_time", ctypes.c_double),
        ("initial_beta0", ctypes.c_int),
        ("final_beta0", ctypes.c_int),
        ("initial_beta1", ctypes.c_int),
        ("final_beta1", ctypes.c_int),
        ("persistence_events", ctypes.c_int),
        ("complexity_trend", ctypes.c_double),
        ("assignment", ctypes.POINTER(ctypes.c_int8)),
        ("assignment_size", ctypes.c_int),
    ]


# ============================================================================
# Python Data Classes
# ============================================================================

@dataclass
class TopologyInfo:
    """Persistent homology information from solver."""
    initial_beta0: int = 0   # Initial connected components
    final_beta0: int = 0     # Final connected components
    initial_beta1: int = 0   # Initial 1-cycles (holes)
    final_beta1: int = 0     # Final 1-cycles
    persistence_events: int = 0   # Topological changes during solve
    complexity_trend: float = 0.0  # Change in complexity score
    
    @property
    def beta0_change(self) -> int:
        """Change in connected components."""
        return self.final_beta0 - self.initial_beta0
    
    @property
    def beta1_change(self) -> int:
        """Change in 1-cycles."""
        return self.final_beta1 - self.initial_beta1


@dataclass
class NitroSatResult:
    """Result from NitroSAT solver."""
    solved: bool              # True if all clauses satisfied
    satisfied: int            # Number of satisfied clauses
    unsatisfied: int          # Number of unsatisfied clauses
    num_vars: int             # Number of variables
    num_clauses: int          # Number of clauses
    solve_time: float         # Time in seconds
    assignment: List[int]     # Variable assignments (1-indexed, 0 or 1)
    topology: TopologyInfo    # Topological analysis info
    
    @property
    def satisfaction_ratio(self) -> float:
        """Ratio of satisfied clauses."""
        if self.num_clauses == 0:
            return 1.0
        return self.satisfied / self.num_clauses
    
    def get_assignment(self, var: int) -> int:
        """Get assignment for a variable (1-indexed)."""
        if 1 <= var <= len(self.assignment) - 1:
            return self.assignment[var]
        raise IndexError(f"Variable {var} out of range [1, {len(self.assignment)-1}]")
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "solved": self.solved,
            "satisfied": self.satisfied,
            "unsatisfied": self.unsatisfied,
            "num_vars": self.num_vars,
            "num_clauses": self.num_clauses,
            "solve_time": self.solve_time,
            "satisfaction_ratio": self.satisfaction_ratio,
            "topology": {
                "initial_beta0": self.topology.initial_beta0,
                "final_beta0": self.topology.final_beta0,
                "initial_beta1": self.topology.initial_beta1,
                "final_beta1": self.topology.final_beta1,
                "persistence_events": self.topology.persistence_events,
                "complexity_trend": self.topology.complexity_trend,
            },
        }


@dataclass
class CNFFormula:
    """
    CNF (Conjunctive Normal Form) formula representation.
    
    A formula is a conjunction (AND) of clauses.
    Each clause is a disjunction (OR) of literals.
    A literal is a variable or its negation.
    
    Variables are 1-indexed positive integers.
    Negative integers represent negations.
    
    Example:
        (x1 OR NOT x2) AND (x2 OR x3)
        = [[1, -2], [2, 3]]
    """
    num_vars: int
    clauses: List[List[int]] = field(default_factory=list)
    
    def add_clause(self, *literals: int) -> "CNFFormula":
        """Add a clause to the formula."""
        clause = list(literals)
        # Validate literals
        for lit in clause:
            if lit == 0:
                raise ValueError("Literal cannot be 0")
            if abs(lit) > self.num_vars:
                raise ValueError(f"Literal {lit} exceeds num_vars {self.num_vars}")
        self.clauses.append(clause)
        return self
    
    def add_clauses(self, clauses: Sequence[Sequence[int]]) -> "CNFFormula":
        """Add multiple clauses."""
        for clause in clauses:
            self.add_clause(*clause)
        return self
    
    @property
    def num_clauses(self) -> int:
        return len(self.clauses)
    
    def to_dimacs(self) -> str:
        """Convert to DIMACS CNF format string."""
        lines = [f"p cnf {self.num_vars} {self.num_clauses}"]
        for clause in self.clauses:
            lines.append(" ".join(str(lit) for lit in clause) + " 0")
        return "\n".join(lines)
    
    def save_dimacs(self, filename: str) -> None:
        """Save to DIMACS CNF file."""
        with open(filename, "w") as f:
            f.write(self.to_dimacs())
    
    @classmethod
    def from_dimacs(cls, content: str) -> "CNFFormula":
        """Parse from DIMACS CNF format string."""
        num_vars = 0
        clauses = []
        
        for line in content.strip().split("\n"):
            line = line.strip()
            if not line or line.startswith("c"):
                continue
            if line.startswith("p"):
                parts = line.split()
                num_vars = int(parts[2])
            else:
                lits = [int(x) for x in line.split() if int(x) != 0]
                if lits:
                    clauses.append(lits)
        
        formula = cls(num_vars=num_vars)
        formula.clauses = clauses
        return formula
    
    @classmethod
    def from_file(cls, filename: str) -> "CNFFormula":
        """Load from DIMACS CNF file."""
        with open(filename) as f:
            return cls.from_dimacs(f.read())


# ============================================================================
# Solver Class
# ============================================================================

class NitroSatSolver:
    """
    High-level interface to NitroSAT solver.
    
    NitroSAT uses physics-inspired techniques:
    - Spectral initialization via XOR-Laplacian eigenvectors
    - Heat kernel smoothing for gradient flow
    - Persistent homology for topology tracking
    - BAHA (Branch-Aware Holonomy Annealing) for phase transitions
    - Dynamic Clause Weighting (DCW) for hard instances
    
    Example:
        >>> solver = NitroSatSolver()
        >>> formula = CNFFormula(3).add_clauses([
        ...     [1, 2],      # x1 OR x2
        ...     [-1, 3],     # NOT x1 OR x3
        ...     [2, -3],     # x2 OR NOT x3
        ... ])
        >>> result = solver.solve(formula)
        >>> print(f"Solved: {result.solved}")
    """
    
    def __init__(
        self,
        max_steps: int = 3000,
        use_dcw: bool = True,
        use_topology: bool = True,
        verbose: bool = False,
        seed: Optional[float] = None,
    ):
        """
        Initialize solver.
        
        Args:
            max_steps: Maximum optimization steps per pass
            use_dcw: Use Dynamic Clause Weighting (recommended for hard instances)
            use_topology: Enable persistent homology analysis
            verbose: Print progress to stdout
            seed: Random seed for reproducibility
        """
        self.max_steps = max_steps
        self.use_dcw = use_dcw
        self.use_topology = use_topology
        self.verbose = verbose
        
        self._lib = _load_library()
        
        if seed is not None:
            self.set_seed(seed)
    
    def set_seed(self, seed: float) -> None:
        """Set random seed for reproducibility."""
        self._lib.nitrosat_set_seed(seed)
    
    @property
    def version(self) -> str:
        """Get NitroSAT version string."""
        return self._lib.nitrosat_version().decode()
    
    def solve_file(self, filename: str) -> NitroSatResult:
        """
        Solve a CNF formula from a DIMACS file.
        
        Args:
            filename: Path to DIMACS CNF file
            
        Returns:
            NitroSatResult with solution and statistics
        """
        result_ptr = self._lib.nitrosat_solve_file(
            filename.encode(),
            self.max_steps,
            1 if self.use_dcw else 0,
            1 if self.use_topology else 0,
            1 if self.verbose else 0,
        )
        
        return self._parse_result(result_ptr)
    
    def solve(self, formula: CNFFormula) -> NitroSatResult:
        """
        Solve a CNF formula.
        
        Args:
            formula: CNFFormula object
            
        Returns:
            NitroSatResult with solution and statistics
        """
        # Write to temporary file (simplest approach)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.cnf', delete=False) as f:
            f.write(formula.to_dimacs())
            temp_path = f.name
        
        try:
            return self.solve_file(temp_path)
        finally:
            os.unlink(temp_path)
    
    def solve_clauses(
        self,
        num_vars: int,
        clauses: Sequence[Sequence[int]],
    ) -> NitroSatResult:
        """
        Solve from raw clauses.
        
        Args:
            num_vars: Number of variables
            clauses: List of clauses (each clause is list of literals)
            
        Returns:
            NitroSatResult with solution and statistics
        """
        formula = CNFFormula(num_vars)
        formula.clauses = [list(c) for c in clauses]
        return self.solve(formula)
    
    def _parse_result(self, result_ptr: int) -> NitroSatResult:
        """Parse C result structure into Python object."""
        if not result_ptr:
            raise RuntimeError("NitroSAT returned null result")
        
        result_struct = ctypes.cast(
            result_ptr,
            ctypes.POINTER(_NitroSatResultStruct)
        ).contents
        
        # Extract assignment
        assignment = [0]  # Index 0 unused (1-indexed)
        if result_struct.assignment and result_struct.assignment_size > 0:
            for i in range(1, result_struct.assignment_size + 1):
                assignment.append(int(result_struct.assignment[i]))
        
        # Build topology info
        topology = TopologyInfo(
            initial_beta0=result_struct.initial_beta0,
            final_beta0=result_struct.final_beta0,
            initial_beta1=result_struct.initial_beta1,
            final_beta1=result_struct.final_beta1,
            persistence_events=result_struct.persistence_events,
            complexity_trend=result_struct.complexity_trend,
        )
        
        # Build result
        result = NitroSatResult(
            solved=bool(result_struct.solved),
            satisfied=result_struct.satisfied,
            unsatisfied=result_struct.unsatisfied,
            num_vars=result_struct.num_vars,
            num_clauses=result_struct.num_clauses,
            solve_time=result_struct.solve_time,
            assignment=assignment,
            topology=topology,
        )
        
        # Free C memory
        self._lib.nitrosat_free_result(result_ptr)
        
        return result


# ============================================================================
# Convenience Functions
# ============================================================================

def solve(
    formula: CNFFormula,
    max_steps: int = 3000,
    use_dcw: bool = True,
    verbose: bool = False,
) -> NitroSatResult:
    """
    Convenience function to solve a CNF formula.
    
    Args:
        formula: CNFFormula object
        max_steps: Maximum optimization steps
        use_dcw: Use Dynamic Clause Weighting
        verbose: Print progress
        
    Returns:
        NitroSatResult with solution and statistics
    """
    solver = NitroSatSolver(
        max_steps=max_steps,
        use_dcw=use_dcw,
        verbose=verbose,
    )
    return solver.solve(formula)


def solve_file(
    filename: str,
    max_steps: int = 3000,
    verbose: bool = False,
) -> NitroSatResult:
    """
    Solve a DIMACS CNF file.
    
    Args:
        filename: Path to DIMACS CNF file
        max_steps: Maximum optimization steps
        verbose: Print progress
        
    Returns:
        NitroSatResult with solution and statistics
    """
    solver = NitroSatSolver(max_steps=max_steps, verbose=verbose)
    return solver.solve_file(filename)


# ============================================================================
# Module Exports
# ============================================================================

__all__ = [
    "NitroSatSolver",
    "NitroSatResult",
    "CNFFormula",
    "TopologyInfo",
    "solve",
    "solve_file",
]
