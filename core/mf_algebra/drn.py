"""
Fractal D/R/N Evolution (Deleted/Retained/New)

Higher-level constructs built on the DRN primitives from kernel.

Core primitives (in kernel.py):
    - DRNDecomposition: Basic D/R/N decomposition
    - drn_decompose(): Kernel method for HLLSet comparison
    - drn_delta(), drn_stability_ratio(): Convenience methods

Higher-level structures (in this module):
    - FractalW: 3D W lattice with fractal D/R/N evolution tracking
    - TemporalDRN: Bidirectional temporal analysis (history + prediction)

Use Cases:
    - Entanglement detection (correlated perceptrons have similar D/R/N patterns)
    - Stability monitoring (Δ → 0 indicates equilibrium)
    - Evolution fingerprinting (DFT of Δ series for correlation)
    - Historical analysis (backward convergence = memory depth)
    - Prediction bounds (forward convergence = forecast horizon)
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

# Import from kernel (canonical location for DRN primitives)
from ..kernel import (
    Kernel,
    DRNDecomposition,
    DRN_CONVERGENCE_THRESHOLD,
)
from ..hllset import HLLSet


@dataclass
class FractalW:
    """
    3D W lattice with fractal D/R/N evolution tracking.
    
    Dimensions:
        - layer (n): N-gram level (0=unigram, 1=bigram, 2=trigram)
        - row/col: Token indices (reg, zeros)
        - depth (k): D/R/N decomposition level
    
    Structure at each depth k:
        - k=0: 3 sets (D, R, N)
        - k=1: 9 sets (D_d, D_r, D_n, R_d, R_r, R_n, N_d, N_r, N_n)
        - k=k: 3^(k+1) sets
    
    Convergence Property:
        The fractal decomposition CONVERGES because cardinality decreases
        at each level. The recursion terminates when:
        
        |D| + |R| + |N| < DRN_CONVERGENCE_THRESHOLD
    """
    max_depth: int = 3
    p_bits: int = 10
    convergence_threshold: float = DRN_CONVERGENCE_THRESHOLD
    
    # Structure: depth -> DRN_path -> HLLSet
    layers: Dict[int, Dict[str, HLLSet]] = field(default_factory=dict)
    
    # Track actual convergence depth (may be < max_depth)
    _convergence_depth: int = field(default=-1, repr=False)
    
    def __post_init__(self):
        """Initialize empty structure at each depth."""
        for depth in range(self.max_depth + 1):
            if depth not in self.layers:
                self.layers[depth] = {}
    
    def drn_paths_at_depth(self, depth: int) -> List[str]:
        """
        Generate all D/R/N paths at given depth.
        
        depth=0: ['D', 'R', 'N']
        depth=1: ['D_D', 'D_R', 'D_N', 'R_D', 'R_R', 'R_N', 'N_D', 'N_R', 'N_N']
        """
        if depth == 0:
            return ['D', 'R', 'N']
        
        paths = []
        for parent in self.drn_paths_at_depth(depth - 1):
            for suffix in ['D', 'R', 'N']:
                paths.append(f"{parent}_{suffix}")
        return paths
    
    def set_hll(self, depth: int, path: str, hll: HLLSet):
        """Set HLLSet at specific depth and path."""
        if depth > self.max_depth:
            raise ValueError(f"Depth {depth} exceeds max_depth {self.max_depth}")
        if depth not in self.layers:
            self.layers[depth] = {}
        self.layers[depth][path] = hll
    
    def get_hll(self, depth: int, path: str) -> Optional[HLLSet]:
        """Get HLLSet at specific depth and path."""
        if depth not in self.layers:
            return None
        return self.layers[depth].get(path)
    
    def total_sets_at_depth(self, depth: int) -> int:
        """Number of D/R/N sets at given depth: 3^(depth+1)"""
        return 3 ** (depth + 1)
    
    def filled_sets_at_depth(self, depth: int) -> int:
        """Count of non-empty sets at given depth."""
        if depth not in self.layers:
            return 0
        return sum(1 for hll in self.layers[depth].values() 
                   if hll.cardinality() > 0)
    
    def total_cardinality_at_depth(self, depth: int) -> float:
        """Sum of all cardinalities at given depth."""
        if depth not in self.layers:
            return 0.0
        return sum(hll.cardinality() for hll in self.layers[depth].values())
    
    def is_converged_at_depth(self, depth: int) -> bool:
        """Check if decomposition has converged at given depth."""
        return self.total_cardinality_at_depth(depth) < self.convergence_threshold
    
    @property
    def convergence_depth(self) -> int:
        """
        Find the depth at which the fractal decomposition converges.
        
        Returns the last depth with meaningful signal (above threshold).
        """
        if self._convergence_depth >= 0:
            return self._convergence_depth
        
        for depth in range(self.max_depth + 1):
            if self.is_converged_at_depth(depth):
                self._convergence_depth = max(0, depth - 1)
                return self._convergence_depth
        
        self._convergence_depth = self.max_depth
        return self._convergence_depth
    
    def update_from_decomposition(
        self, 
        decomp: DRNDecomposition, 
        depth: int = 0,
        stop_on_convergence: bool = True
    ):
        """Update FractalW from a DRNDecomposition."""
        if depth > self.max_depth:
            return
        
        if stop_on_convergence and decomp.is_converged():
            return
        
        self.set_hll(depth, 'D', decomp.deleted)
        self.set_hll(depth, 'R', decomp.retained)
        self.set_hll(depth, 'N', decomp.new)
        
        self._convergence_depth = -1
        
        if depth < self.max_depth:
            if decomp.d_decomp and not decomp.d_decomp.converged:
                self._update_sub(decomp.d_decomp, depth + 1, 'D', stop_on_convergence)
            if decomp.r_decomp and not decomp.r_decomp.converged:
                self._update_sub(decomp.r_decomp, depth + 1, 'R', stop_on_convergence)
            if decomp.n_decomp and not decomp.n_decomp.converged:
                self._update_sub(decomp.n_decomp, depth + 1, 'N', stop_on_convergence)
    
    def _update_sub(
        self, 
        decomp: DRNDecomposition, 
        depth: int, 
        prefix: str,
        stop_on_convergence: bool = True
    ):
        """Helper to update sub-decomposition with path prefix."""
        if depth > self.max_depth:
            return
        
        if stop_on_convergence and decomp.is_converged():
            return
        
        self.set_hll(depth, f'{prefix}_D', decomp.deleted)
        self.set_hll(depth, f'{prefix}_R', decomp.retained)
        self.set_hll(depth, f'{prefix}_N', decomp.new)
        
        if depth < self.max_depth:
            if decomp.d_decomp and not decomp.d_decomp.converged:
                self._update_sub(decomp.d_decomp, depth + 1, f'{prefix}_D', stop_on_convergence)
            if decomp.r_decomp and not decomp.r_decomp.converged:
                self._update_sub(decomp.r_decomp, depth + 1, f'{prefix}_R', stop_on_convergence)
            if decomp.n_decomp and not decomp.n_decomp.converged:
                self._update_sub(decomp.n_decomp, depth + 1, f'{prefix}_N', stop_on_convergence)
    
    def cardinality_vector(self, depth: int = 0) -> List[float]:
        """Get cardinality vector at given depth."""
        paths = self.drn_paths_at_depth(depth)
        kernel = Kernel(p_bits=self.p_bits)
        empty = kernel.absorb(set())
        return [
            self.layers[depth].get(p, empty).cardinality()
            for p in paths
        ]
    
    def summary(self) -> Dict[str, Any]:
        """Get summary of fractal structure with convergence info."""
        result = {
            "convergence_depth": self.convergence_depth,
            "convergence_threshold": self.convergence_threshold,
        }
        for depth in range(self.max_depth + 1):
            total = self.total_sets_at_depth(depth)
            filled = self.filled_sets_at_depth(depth)
            total_card = self.total_cardinality_at_depth(depth)
            result[f"depth_{depth}"] = {
                "total_sets": total,
                "filled_sets": filled,
                "fill_ratio": filled / total if total > 0 else 0,
                "total_cardinality": round(total_card, 1),
                "converged": total_card < self.convergence_threshold
            }
        return result
    
    def __repr__(self) -> str:
        filled_0 = self.filled_sets_at_depth(0)
        conv = self.convergence_depth
        return f"FractalW(max_depth={self.max_depth}, filled@0={filled_0}/3, converges@{conv})"


@dataclass
class TemporalDRN:
    """
    Bidirectional temporal D/R/N analysis.
    
    Captures both history (backward) and prediction (forward) horizons
    using the same fractal D/R/N structure.
    """
    # History analysis (backward from current D)
    history: FractalW = field(default_factory=lambda: FractalW(max_depth=3))
    
    # Prediction analysis (forward from current N)
    prediction: FractalW = field(default_factory=lambda: FractalW(max_depth=3))
    
    # Current state
    current_drn: Optional[DRNDecomposition] = None
    
    @property
    def memory_depth(self) -> int:
        """How far back historical influences persist."""
        return self.history.convergence_depth
    
    @property
    def prediction_horizon(self) -> int:
        """How far forward we can reliably project."""
        return self.prediction.convergence_depth
    
    @property
    def temporal_symmetry(self) -> float:
        """
        Measure of past/future symmetry.
        
        Returns 1.0 if memory_depth == prediction_horizon (balanced)
        Returns <1.0 if asymmetric (phase transition indicator)
        """
        m, p = self.memory_depth, self.prediction_horizon
        if m == 0 and p == 0:
            return 1.0
        return 1.0 - abs(m - p) / max(m, p, 1)
    
    def update_backward(
        self, 
        hll_history: List[HLLSet],
        p_bits: int = 10
    ):
        """
        Build history tail from sequence of past HLLSets.
        
        Args:
            hll_history: [hll(t), hll(t-1), hll(t-2), ...] oldest last
        """
        if len(hll_history) < 2:
            return
        
        for i in range(len(hll_history) - 1):
            hll_curr = hll_history[i]
            hll_prev = hll_history[i + 1]
            
            drn = DRNDecomposition.from_evolution(
                hll_prev, hll_curr, 
                p_bits=p_bits,
                depth=i
            )
            drn.direction = 'backward'
            
            if i == 0:
                self.current_drn = drn
                self.history.update_from_decomposition(drn, depth=0)
            else:
                self.history.set_hll(i, 'D', drn.deleted)
                self.history.set_hll(i, 'R', drn.retained)
                self.history.set_hll(i, 'N', drn.new)
            
            if drn.is_converged():
                break
    
    def update_forward(
        self,
        hll_future: List[HLLSet],
        p_bits: int = 10
    ):
        """
        Build prediction horizon from sequence of future HLLSets.
        
        Args:
            hll_future: [hll(t), hll(t+1), hll(t+2), ...] oldest first
        """
        if len(hll_future) < 2:
            return
        
        for i in range(len(hll_future) - 1):
            hll_curr = hll_future[i]
            hll_next = hll_future[i + 1]
            
            drn = DRNDecomposition.from_evolution(
                hll_curr, hll_next,
                p_bits=p_bits,
                depth=i
            )
            drn.direction = 'forward'
            
            if i == 0:
                if self.current_drn is None:
                    self.current_drn = drn
                self.prediction.update_from_decomposition(drn, depth=0)
            else:
                self.prediction.set_hll(i, 'D', drn.deleted)
                self.prediction.set_hll(i, 'R', drn.retained)
                self.prediction.set_hll(i, 'N', drn.new)
            
            if drn.is_converged():
                break
    
    def summary(self) -> Dict[str, Any]:
        """Get summary of temporal structure."""
        return {
            "memory_depth": self.memory_depth,
            "prediction_horizon": self.prediction_horizon,
            "temporal_symmetry": round(self.temporal_symmetry, 3),
            "history": self.history.summary(),
            "prediction": self.prediction.summary(),
        }


__all__ = [
    'DRN_CONVERGENCE_THRESHOLD',
    'DRNDecomposition',
    'FractalW',
    'TemporalDRN',
]
