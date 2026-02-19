"""
Hash Relational Tensor (HRT) - Immutable PyTorch-based Implementation

HRT combines:
1. Adjacency Matrix (AM) - ImmutableTensor of dimension N x N
2. HLLSet Lattice W - organized with same dimension as AM
3. Perceptron mappings - hash-based token indices

Key Properties:
- Immutable and idempotent
- PyTorch-based tensors (GPU-ready)
- Content-addressed naming (SHA1)
- Three-state evolution: In-Process → Current → History

Architecture:
- HRT evolves through discrete steps
- Each step: ingest → merge → commit
- Parent pointer maintains lineage
"""

from __future__ import annotations
from typing import List, Dict, Set, Optional, Tuple, Union, FrozenSet
from dataclasses import dataclass, field
from enum import Enum, auto
import time

from .immutable_tensor import (
    ImmutableTensor, compute_element_hash, 
    compute_aggregate_hash, compute_structural_hash
)
from .hllset import HLLSet, compute_sha1
from .constants import P_BITS as KERNEL_P_BITS


# =============================================================================
# SECTION 1: Configuration
# =============================================================================

@dataclass(frozen=True)
class HRTConfig:
    """
    Configuration for Hash Relational Tensor.
    
    From paper Section 2.1: HLLSet = (H, φ, τ, ρ)
    - H: Array of registers
    - φ: Tokenization functor
    - τ: Inclusion tolerance threshold (morphism exists if BSS_τ ≥ τ)
    - ρ: Exclusion intolerance threshold (morphism exists if BSS_ρ ≤ ρ)
    
    Dimension = 2^P * h_bits + 2
    +2 for START and END special tokens
    
    Constraint: 0 ≤ ρ < τ ≤ 1
    """
    p_bits: int = 10           # HLL precision (m = 2^p registers)
    h_bits: int = 32           # Hash bit size for element hashes
    tau: float = 0.7           # Inclusion tolerance threshold (τ)
    rho: float = 0.3           # Exclusion intolerance threshold (ρ)
    epsilon: float = 0.1       # ε-isomorphism tolerance for entanglement
    
    def __post_init__(self):
        """Validate thresholds."""
        if not (0 <= self.rho < self.tau <= 1):
            raise ValueError(f"Thresholds must satisfy 0 ≤ ρ < τ ≤ 1, got ρ={self.rho}, τ={self.tau}")
        if not (0 < self.epsilon < 1):
            raise ValueError(f"Epsilon must satisfy 0 < ε < 1, got ε={self.epsilon}")
    
    @property
    def dimension(self) -> int:
        """AM dimension."""
        return (1 << self.p_bits) * self.h_bits + 2
    
    @property
    def num_basic_hllsets(self) -> int:
        """Total basic HLLSets = 2 * dimension (rows + columns)."""
        return 2 * self.dimension


# =============================================================================
# SECTION 2: Adjacency Matrix (Immutable)
# =============================================================================

@dataclass(frozen=True)
class AdjacencyMatrix:
    """
    Immutable Adjacency Matrix using ImmutableTensor.
    
    AM[i, j] represents relationship strength between:
    - Row i (via row basic HLLSet r_i)
    - Column j (via column basic HLLSet c_j)
    
    Rows: "future" projection (what could come next)
    Columns: "past" reconstruction (what context gave birth)
    """
    tensor: ImmutableTensor
    config: HRTConfig
    
    @classmethod
    def empty(cls, config: HRTConfig) -> AdjacencyMatrix:
        """Create empty adjacency matrix."""
        tensor = ImmutableTensor.zeros(config.dimension, config.dimension)
        return cls(tensor=tensor, config=config)
    
    def with_entry(self, row: int, col: int, value: float) -> AdjacencyMatrix:
        """Return new AM with entry set."""
        new_tensor = self.tensor.with_value((row, col), value)
        return AdjacencyMatrix(tensor=new_tensor, config=self.config)
    
    def with_entries(self, entries: List[Tuple[int, int, float]]) -> AdjacencyMatrix:
        """Return new AM with multiple entries set."""
        new_tensor = self.tensor
        for row, col, value in entries:
            new_tensor = new_tensor.with_value((row, col), value)
        return AdjacencyMatrix(tensor=new_tensor, config=self.config)
    
    def merge(self, other: AdjacencyMatrix) -> AdjacencyMatrix:
        """Element-wise maximum merge (for HRT evolution)."""
        if self.config != other.config:
            raise ValueError("Cannot merge AMs with different configs")
        merged_tensor = self.tensor.maximum(other.tensor)
        return AdjacencyMatrix(tensor=merged_tensor, config=self.config)
    
    def get(self, row: int, col: int) -> float:
        """Get value at position."""
        return float(self.tensor.data[row, col])
    
    def project_rows(self, col_indices: List[int]) -> ImmutableTensor:
        """
        Project columns to rows (future projection).
        Returns new ImmutableTensor of row activations.
        """
        if not col_indices:
            zeros = ImmutableTensor.zeros(self.config.dimension)
            return zeros
        
        # Sum over specified columns
        projection = self.tensor.project_rows(col_indices)
        return ImmutableTensor.from_tensor(projection)
    
    def project_cols(self, row_indices: List[int]) -> ImmutableTensor:
        """
        Project rows to columns (past reconstruction).
        Returns new ImmutableTensor of column activations.
        """
        if not row_indices:
            zeros = ImmutableTensor.zeros(self.config.dimension)
            return zeros
        
        projection = self.tensor.project_cols(row_indices)
        return ImmutableTensor.from_tensor(projection)
    
    def nonzero_entries(self) -> List[Tuple[int, int, float]]:
        """Get all non-zero entries as (row, col, value)."""
        entries = self.tensor.nonzero_with_values()
        # Filter for 2D indices
        return [(r, c, v) for (r, c), v in entries if isinstance(r, int) and isinstance(c, int)]
    
    @property
    def name(self) -> str:
        """Content-addressed name."""
        return self.tensor.name
    
    def __repr__(self) -> str:
        nonzero = len(self.nonzero_entries())
        return f"AM({self.config.dimension}x{self.config.dimension}, {nonzero} nz, {self.name[:8]}...)"


# =============================================================================
# SECTION 3: Basic HLLSet (Immutable)
# =============================================================================

@dataclass(frozen=True)
class BasicHLLSet:
    """
    A basic HLLSet in the lattice.
    
    From paper Section 3.1: Objects in category HLL with morphisms defined by BSS.
    
    Building blocks of the system:
    - Row basic: r[0] to r[dimension-1] (future projections)
    - Column basic: c[0] to c[dimension-1] (past reconstructions)
    
    Morphisms (paper Section 2.2):
    - f: A → B exists iff BSS_τ(A→B) ≥ τ_A and BSS_ρ(A→B) ≤ ρ_B
    """
    index: int
    is_row: bool
    hllset: HLLSet
    config: HRTConfig = field(compare=False)
    
    @property
    def name(self) -> str:
        """Unique identifier: 'r_i' or 'c_i'."""
        prefix = 'r' if self.is_row else 'c'
        return f"{prefix}_{self.index}"
    
    @property
    def is_start(self) -> bool:
        """Check if this is the START special token (index 0, row)."""
        return self.is_row and self.index == 0
    
    @property
    def is_end(self) -> bool:
        """Check if this is the END special token (index dimension-1, row)."""
        return self.is_row and self.index == self.config.dimension - 1
    
    def bss_tau(self, other: BasicHLLSet) -> float:
        """
        Bell State Similarity (inclusion).
        
        From paper Eq. 2.2:
        BSS_τ(A → B) = |A ∩ B| / |B|
        
        Measures how much of B is covered by A.
        """
        if other.hllset.cardinality() == 0:
            return 0.0
        intersection_card = self.hllset.intersect(other.hllset).cardinality()
        return intersection_card / other.hllset.cardinality()
    
    def bss_rho(self, other: BasicHLLSet) -> float:
        """
        Bell State Similarity (exclusion).
        
        From paper Eq. 2.2:
        BSS_ρ(A → B) = |A \\ B| / |B|
        
        Measures how much of A is outside B (relative to |B|).
        """
        if other.hllset.cardinality() == 0:
            return 0.0
        intersection_card = self.hllset.intersect(other.hllset).cardinality()
        difference = self.hllset.cardinality() - intersection_card
        return difference / other.hllset.cardinality()
    
    def has_morphism_to(self, other: BasicHLLSet) -> bool:
        """
        Check if morphism f: self → other exists.
        
        From paper Section 2.2:
        Morphism exists iff BSS_τ(A→B) ≥ τ and BSS_ρ(A→B) ≤ ρ
        """
        return (self.bss_tau(other) >= self.config.tau and 
                self.bss_rho(other) <= self.config.rho)
    
    def __repr__(self) -> str:
        special = ""
        if self.is_start:
            special = " [START]"
        elif self.is_end:
            special = " [END]"
        return f"Basic({self.name}{special}, {self.hllset.short_name}...)"


# =============================================================================
# SECTION 4: HLLSet Lattice (Immutable)
# =============================================================================

@dataclass(frozen=True)
class HLLSetLattice:
    """
    Immutable lattice of basic HLLSets.
    
    Contains:
    - Row basic HLLSets: r[0] to r[dimension-1]
    - Column basic HLLSets: c[0] to c[dimension-1]
    
    The lattice is frozen after creation. "Modification" creates new lattice.
    """
    config: HRTConfig
    row_basic: Tuple[BasicHLLSet, ...] = field(default_factory=tuple)
    col_basic: Tuple[BasicHLLSet, ...] = field(default_factory=tuple)
    
    @classmethod
    def empty(cls, config: HRTConfig) -> HLLSetLattice:
        """Create empty lattice with initialized basic HLLSets."""
        from .hllset import HLLSet
        
        p_bits = KERNEL_P_BITS
        dim = config.dimension
        
        # Create row basic HLLSets (empty)
        rows = []
        for i in range(dim):
            hllset = HLLSet(p_bits=p_bits)
            basic = BasicHLLSet(index=i, is_row=True, hllset=hllset, config=config)
            rows.append(basic)
        
        # Create column basic HLLSets (empty)
        cols = []
        for i in range(dim):
            hllset = HLLSet(p_bits=p_bits)
            basic = BasicHLLSet(index=i, is_row=False, hllset=hllset, config=config)
            cols.append(basic)
        
        return cls(config=config, row_basic=tuple(rows), col_basic=tuple(cols))
    
    def with_row_basic(self, index: int, hllset: HLLSet) -> HLLSetLattice:
        """Return new lattice with row basic HLLSet at index updated."""
        if not 0 <= index < len(self.row_basic):
            raise IndexError(f"Index {index} out of range")
        
        new_basic = BasicHLLSet(
            index=index,
            is_row=True,
            hllset=hllset,
            config=self.config
        )
        new_rows = list(self.row_basic)
        new_rows[index] = new_basic
        
        return HLLSetLattice(
            config=self.config,
            row_basic=tuple(new_rows),
            col_basic=self.col_basic
        )
    
    def with_col_basic(self, index: int, hllset: HLLSet) -> HLLSetLattice:
        """Return new lattice with column basic HLLSet at index updated."""
        if not 0 <= index < len(self.col_basic):
            raise IndexError(f"Index {index} out of range")
        
        new_basic = BasicHLLSet(
            index=index,
            is_row=False,
            hllset=hllset,
            config=self.config
        )
        new_cols = list(self.col_basic)
        new_cols[index] = new_basic
        
        return HLLSetLattice(
            config=self.config,
            row_basic=self.row_basic,
            col_basic=tuple(new_cols)
        )
    
    def merge(self, other: HLLSetLattice, kernel) -> HLLSetLattice:
        """
        Merge two lattices by unioning basic HLLSets.
        """
        if self.config != other.config:
            raise ValueError("Cannot merge lattices with different configs")
        
        new_rows = []
        for i in range(self.config.dimension):
            row_union = kernel.union(self.row_basic[i].hllset, other.row_basic[i].hllset)
            new_rows.append(BasicHLLSet(
                index=i, is_row=True, hllset=row_union, config=self.config
            ))
        
        new_cols = []
        for i in range(self.config.dimension):
            col_union = kernel.union(self.col_basic[i].hllset, other.col_basic[i].hllset)
            new_cols.append(BasicHLLSet(
                index=i, is_row=False, hllset=col_union, config=self.config
            ))
        
        return HLLSetLattice(
            config=self.config,
            row_basic=tuple(new_rows),
            col_basic=tuple(new_cols)
        )
    
    def _compute_node_degrees(self) -> Tuple[List[int], List[int]]:
        """
        Compute node degrees (number of morphisms) for each basic HLLSet.
        
        Returns:
            (row_degrees, col_degrees): Lists of out-degrees for each node
        """
        row_degrees = []
        col_degrees = []
        
        # Row degrees: count morphisms from each row to all columns
        for r in self.row_basic:
            degree = sum(1 for c in self.col_basic if r.has_morphism_to(c))
            row_degrees.append(degree)
        
        # Column degrees: count morphisms from each column to all rows
        for c in self.col_basic:
            degree = sum(1 for r in self.row_basic if c.has_morphism_to(r))
            col_degrees.append(degree)
        
        return row_degrees, col_degrees
    
    def compare_lattices(self, other: 'HLLSetLattice') -> Dict[str, float]:
        """
        Standardized lattice comparison for dual-lattice systems.
        
        Compares two lattices by STRUCTURE (graph topology), not by node content.
        This structural comparison is particularly useful when comparing lattices
        with different token spaces (e.g., cross-modal systems) or when measuring
        entanglement between lattices with overlapping but non-identical HLLSets.
        
        Algorithm:
        1. Compute node degrees (morphism counts) in each lattice
        2. Match nodes by degree similarity (simplified structural matching)
        3. Measure structural alignment using degree correlation
        
        Returns:
            dict with keys:
                - 'row_degree_correlation': Pearson correlation of row degrees
                - 'col_degree_correlation': Pearson correlation of col degrees
                - 'overall_structure_match': Combined structural similarity
                - 'epsilon_isomorphic_prob': Probability of ε-isomorphism by structure
                - 'row_degree_distance': Normalized L1 distance of row degrees
                - 'col_degree_distance': Normalized L1 distance of col degrees
        """
        if len(self.row_basic) != len(other.row_basic):
            raise ValueError("Lattices must have same dimension")
        
        # Compute degrees for both lattices
        self_row_deg, self_col_deg = self._compute_node_degrees()
        other_row_deg, other_col_deg = other._compute_node_degrees()
        
        # Compute degree correlations (structural similarity)
        row_corr = self._degree_correlation(self_row_deg, other_row_deg)
        col_corr = self._degree_correlation(self_col_deg, other_col_deg)
        
        # Compute degree distances (normalized L1)
        row_dist = self._degree_distance(self_row_deg, other_row_deg)
        col_dist = self._degree_distance(self_col_deg, other_col_deg)
        
        # Overall structural match (average correlation)
        overall_match = (row_corr + col_corr) / 2.0
        
        # ε-isomorphism probability based on structural similarity
        # High correlation + low distance → high ε-isomorphic probability
        epsilon = self.config.epsilon
        epsilon_prob = overall_match * (1.0 - (row_dist + col_dist) / 2.0)
        epsilon_prob = max(0.0, min(1.0, epsilon_prob))  # Clamp to [0, 1]
        
        return {
            'row_degree_correlation': row_corr,
            'col_degree_correlation': col_corr,
            'overall_structure_match': overall_match,
            'epsilon_isomorphic_prob': epsilon_prob,
            'row_degree_distance': row_dist,
            'col_degree_distance': col_dist
        }
    
    def _degree_correlation(self, deg1: List[int], deg2: List[int]) -> float:
        """
        Compute Pearson correlation between two degree sequences.
        Returns 0.0 if either sequence has zero variance.
        """
        if len(deg1) != len(deg2) or len(deg1) == 0:
            return 0.0
        
        n = len(deg1)
        mean1 = sum(deg1) / n
        mean2 = sum(deg2) / n
        
        numerator = sum((d1 - mean1) * (d2 - mean2) for d1, d2 in zip(deg1, deg2))
        
        var1 = sum((d1 - mean1) ** 2 for d1 in deg1)
        var2 = sum((d2 - mean2) ** 2 for d2 in deg2)
        
        if var1 == 0 or var2 == 0:
            return 0.0
        
        denominator = (var1 * var2) ** 0.5
        return numerator / denominator if denominator > 0 else 0.0
    
    def _degree_distance(self, deg1: List[int], deg2: List[int]) -> float:
        """
        Compute normalized L1 distance between degree sequences.
        Returns value in [0, 1].
        """
        if len(deg1) != len(deg2) or len(deg1) == 0:
            return 1.0
        
        total_distance = sum(abs(d1 - d2) for d1, d2 in zip(deg1, deg2))
        max_possible = sum(max(d1, d2) for d1, d2 in zip(deg1, deg2))
        
        if max_possible == 0:
            return 0.0
        
        return total_distance / max_possible
    
    def semantic_grounding_level(self, other: 'HLLSetLattice') -> str:
        """
        Classify semantic grounding level between two lattices.
        
        Uses structural similarity (ε-isomorphism probability based on
        graph structure) to categorize:
        - Strong: P ≥ 0.7 (well-grounded, similar structure)
        - Moderate: 0.5 ≤ P < 0.7 (partial grounding)
        - Weak: 0.3 ≤ P < 0.5 (abstract, weak structural similarity)
        - Disconnected: P < 0.3 (no grounding, different structures)
        
        Returns:
            str: Grounding level classification
        """
        metrics = self.compare_lattices(other)
        prob = metrics['epsilon_isomorphic_prob']
        
        if prob >= 0.7:
            return "Strong (well-grounded)"
        elif prob >= 0.5:
            return "Moderate (partial grounding)"
        elif prob >= 0.3:
            return "Weak (abstract)"
        else:
            return "Disconnected (no grounding)"
    
    def get_row_basic(self, index: int) -> Optional[BasicHLLSet]:

        """Get row basic HLLSet at index."""
        if 0 <= index < len(self.row_basic):
            return self.row_basic[index]
        return None
    
    def get_col_basic(self, index: int) -> Optional[BasicHLLSet]:
        """Get column basic HLLSet at index."""
        if 0 <= index < len(self.col_basic):
            return self.col_basic[index]
        return None
    
    def get_basic(self, name: str) -> Optional[BasicHLLSet]:
        """Get basic HLLSet by name (e.g., 'r_5' or 'c_3')."""
        if name.startswith('r_'):
            idx = int(name[2:])
            return self.get_row_basic(idx)
        elif name.startswith('c_'):
            idx = int(name[2:])
            return self.get_col_basic(idx)
        return None
    
    @property
    def name(self) -> str:
        """Content-addressed name (hash of all basic HLLSet names)."""
        hashes = [b.hllset.name for b in self.row_basic + self.col_basic]
        return compute_structural_hash(*hashes)
    
    def is_epsilon_isomorphic(self, other: HLLSetLattice, epsilon: Optional[float] = None) -> bool:
        """
        Check if two lattices are ε-isomorphic.
        
        From paper Definition 4.1:
        Two lattices are ε-isomorphic if there exists a bijection φ such that:
        |BSS(A, B) - BSS(φ(A), φ(B))| ≤ ε for all A, B
        
        Simplified implementation: check pairwise BSS preservation.
        """
        if self.config != other.config:
            return False
        
        eps = epsilon if epsilon is not None else self.config.epsilon
        
        # Check row basic HLLSets
        for i in range(len(self.row_basic)):
            for j in range(len(self.row_basic)):
                bss_self = self.row_basic[i].bss_tau(self.row_basic[j])
                bss_other = other.row_basic[i].bss_tau(other.row_basic[j])
                if abs(bss_self - bss_other) > eps:
                    return False
        
        # Check column basic HLLSets
        for i in range(len(self.col_basic)):
            for j in range(len(self.col_basic)):
                bss_self = self.col_basic[i].bss_tau(self.col_basic[j])
                bss_other = other.col_basic[i].bss_tau(other.col_basic[j])
                if abs(bss_self - bss_other) > eps:
                    return False
        
        return True
    
    def entanglement_probability(self, num_datasets: int, dataset_size: int) -> float:
        """
        Compute probability of entanglement failure.
        
        From paper Theorem 4.2:
        P(not ε-isomorphic) ≤ min(1, n² · (d²/2^m + exp(-ε²d/2)))
        
        where:
        - n = num_datasets
        - d = dataset_size
        - m = 2^p_bits (number of registers)
        - ε = epsilon
        """
        import math
        n = num_datasets
        d = dataset_size
        m = 1 << self.config.p_bits
        eps = self.config.epsilon
        
        collision_term = (d * d) / (2 ** m)
        deviation_term = math.exp(-(eps * eps * d) / 2)
        
        prob = min(1.0, n * n * (collision_term + deviation_term))
        return prob
    
    def __repr__(self) -> str:
        return f"Lattice({len(self.row_basic)} rows, {len(self.col_basic)} cols, {self.name[:8]}...)"


# =============================================================================
# SECTION 5: Cover - Optimal Composition
# =============================================================================

@dataclass(frozen=True)
class Cover:
    """
    Optimal composition of basic HLLSets.
    
    Represents a non-basic HLLSet as minimal set of basic HLLSets.
    """
    row_indices: FrozenSet[int]
    col_indices: FrozenSet[int]
    target_hash: str  # Hash of the HLLSet this covers
    
    @property
    def size(self) -> int:
        """Total number of basic HLLSets in cover."""
        return len(self.row_indices) + len(self.col_indices)
    
    def get_basic_names(self) -> List[str]:
        """Get list of basic HLLSet names."""
        names = [f"r_{i}" for i in sorted(self.row_indices)]
        names.extend([f"c_{i}" for i in sorted(self.col_indices)])
        return names
    
    @property
    def name(self) -> str:
        """Content-addressed name of the cover."""
        components = [
            self.target_hash,
            ",".join(map(str, sorted(self.row_indices))),
            ",".join(map(str, sorted(self.col_indices)))
        ]
        return compute_structural_hash(*components)
    
    def __repr__(self) -> str:
        return f"Cover({self.name[:8]}..., rows={len(self.row_indices)}, cols={len(self.col_indices)})"


# =============================================================================
# SECTION 6: Noether Current and Conservation
# =============================================================================

@dataclass(frozen=True)
class NoetherCurrent:
    """
    Conservation tracking for HRT evolution.
    
    From paper Section 6.2:
    Noether current J_uv(p) = p[u]·(Ap)[v] - p[v]·(A^T·p)[u]
    
    Total flux Φ = Σ J_uv is conserved: dΦ/dt = 0
    """
    flux: float
    timestamp: float
    step_number: int
    
    @classmethod
    def compute(cls, am: AdjacencyMatrix, distribution: ImmutableTensor, step_number: int) -> NoetherCurrent:
        """
        Compute Noether current for given AM and probability distribution.
        
        Args:
            am: Adjacency matrix
            distribution: Probability distribution over states
            step_number: Current evolution step
        """
        import torch
        
        # Forward: Ap
        forward = torch.matmul(am.tensor.data, distribution.data)
        
        # Retro: A^T p
        retro = torch.matmul(am.tensor.data.T, distribution.data)
        
        # Compute flux: Σ_uv p[u]·(Ap)[v] - p[v]·(A^T·p)[u]
        # Simplified: sum of differences
        flux = float(torch.sum(distribution.data * forward - distribution.data * retro))
        
        return cls(
            flux=flux,
            timestamp=time.time(),
            step_number=step_number
        )
    
    def __repr__(self) -> str:
        return f"NoetherCurrent(Φ={self.flux:.6f}, step={self.step_number})"


@dataclass(frozen=True)
class EvolutionTriple:
    """
    Explicit tracking of evolution dynamics.
    
    From paper Section 5.1:
    H(t+1) = [H(t) \\ D] ∪ N
    
    where:
    - D: Deleted information (forget)
    - R: Retained information (H(t) \\ D)
    - N: New information (add)
    """
    deleted: Set[str]       # Hashes of deleted HLLSets
    retained: Set[str]      # Hashes of retained HLLSets
    new: Set[str]           # Hashes of new HLLSets
    
    @property
    def cardinality_change(self) -> int:
        """
        Net change in cardinality: |N| - |D|
        
        From paper: When |N| = |D|, evolution is conservative (Noether current = 0)
        """
        return len(self.new) - len(self.deleted)
    
    @property
    def is_conservative(self) -> bool:
        """Check if evolution conserves cardinality."""
        return self.cardinality_change == 0
    
    def __repr__(self) -> str:
        return f"EvolutionTriple(D={len(self.deleted)}, R={len(self.retained)}, N={len(self.new)}, ΔQ={self.cardinality_change})"


# =============================================================================
# SECTION 7: Contextual Selection
# =============================================================================

@dataclass(frozen=True)
class ContextualSelection:
    """
    Contextual Selection Operator.
    
    From paper Section 7.1:
    S_C: U → {0,1}
    S_C(x) = 1 iff BSS(F_C, F_x) ≥ τ_C and exclusion(F_C, F_x) ≤ ρ_C
    
    **Key insight**: τ/ρ thresholds are the MECHANICS of contextual selection.
    They define the concrete mechanism by which a context determines what fits "in it".
    
    The fundamental inversion: Context C actively selects compatible elements,
    not vice versa. The context precedes and determines content.
    
    Process:
    1. Context C has thresholds τ (inclusion) and ρ (exclusion)
    2. For each candidate x, compute BSS_τ(C, x) and BSS_ρ(C, x)
    3. If BSS_τ ≥ τ AND BSS_ρ ≤ ρ, then C selects x
    4. Selected elements are "in" the context (active selection, not passive membership)
    """
    context_hash: str
    selected_hashes: FrozenSet[str]
    tau_threshold: float
    rho_threshold: float
    selection_power: float  # Conservation tracking
    
    @classmethod
    def from_context(cls, context: BasicHLLSet, candidates: List[BasicHLLSet]) -> ContextualSelection:
        """
        Apply contextual selection.
        
        Context selects candidates that satisfy BSS thresholds.
        """
        selected = []
        total_power = 0.0
        
        for candidate in candidates:
            bss_tau = context.bss_tau(candidate)
            bss_rho = context.bss_rho(candidate)
            
            if bss_tau >= context.config.tau and bss_rho <= context.config.rho:
                selected.append(candidate.hllset.name)
                total_power += bss_tau
        
        return cls(
            context_hash=context.hllset.name,
            selected_hashes=frozenset(selected),
            tau_threshold=context.config.tau,
            rho_threshold=context.config.rho,
            selection_power=total_power
        )
    
    def __repr__(self) -> str:
        return f"Selection(ctx={self.context_hash[:8]}..., selected={len(self.selected_hashes)}, power={self.selection_power:.2f})"


# =============================================================================
# SECTION 8: Hash Relational Tensor (HRT) - Immutable
# =============================================================================

@dataclass(frozen=True)
class HRT:
    """
    Hash Relational Tensor - Immutable evolution unit.
    
    From paper: HRT combines AM (adjacency) and W (HLLSet lattice).
    
    Category Theory (Section 3):
    - Objects: HRTs with HLLSet lattices
    - Morphisms: Evolution steps preserving structural isomorphism
    - Composition: Sequential evolution with conservation laws
    
    Components:
    - am: Adjacency Matrix (relationship structure)
    - lattice: HLLSetLattice (basic HLLSets) - the W lattice
    - covers: Optional cover cache (doesn't affect identity)
    - noether_current: Conservation tracking (Section 6.2)
    - evolution_triple: (D, R, N) tracking (Section 5.1)
    
    Evolution:
    - parent_hrt: Hash of previous HRT (history pointer)
    - step_number: Position in evolution sequence
    
    Identity:
    - name: SHA1 of (am_hash, lattice_hash, parent_hrt, step_number)
    """
    config: HRTConfig
    am: AdjacencyMatrix
    lattice: HLLSetLattice
    parent_hrt: Optional[str] = None
    step_number: int = 0
    covers: Tuple[Cover, ...] = field(default_factory=tuple, compare=False)
    noether_current: Optional[NoetherCurrent] = field(default=None, compare=False)
    evolution_triple: Optional[EvolutionTriple] = field(default=None, compare=False)
    timestamp: float = field(default_factory=time.time, compare=False)
    
    @classmethod
    def empty(cls, config: HRTConfig) -> HRT:
        """Create empty HRT (genesis state)."""
        am = AdjacencyMatrix.empty(config)
        lattice = HLLSetLattice.empty(config)
        return cls(
            config=config,
            am=am,
            lattice=lattice,
            parent_hrt=None,
            step_number=0
        )
    
    @classmethod
    def from_perceptrons(cls,
                         perceptron_data: Dict[str, Set[str]],
                         kernel,
                         config: HRTConfig,
                         parent_hrt: Optional[str] = None,
                         step_number: int = 0) -> HRT:
        """
        Create HRT from perceptron data (ingestion step).
        
        This creates an 'in_process' HRT ready to be merged with current.
        """
        am = AdjacencyMatrix.empty(config)
        lattice = HLLSetLattice.empty(config)
        covers = []
        
        # Map tokens to AM indices using element hash
        token_to_idx = {}
        
        for source_name, tokens in perceptron_data.items():
            for token in tokens:
                if token not in token_to_idx:
                    # Use element hash for index assignment
                    token_hash = compute_element_hash(token, bits=config.h_bits)
                    idx = (token_hash % (config.dimension - 2)) + 2
                    token_to_idx[token] = idx
        
        # Populate AM and create HLLSets
        am_entries = []
        for source_name, tokens in perceptron_data.items():
            # Create HLLSet for this perceptron
            hllset = kernel.absorb(tokens)
            
            # Map source to row index
            source_hash = compute_element_hash(source_name, bits=config.h_bits)
            source_idx = source_hash % config.dimension
            
            # Add connections to AM
            for token in tokens:
                col_idx = token_to_idx[token]
                am_entries.append((source_idx, col_idx, 1.0))
        
        # Create AM with entries
        am = am.with_entries(am_entries)
        
        return cls(
            config=config,
            am=am,
            lattice=lattice,
            parent_hrt=parent_hrt,
            step_number=step_number,
            covers=tuple(covers)
        )
    
    def merge(self, other: HRT, kernel) -> HRT:
        """
        Merge two HRTs to create new HRT.
        
        From paper Section 5.1:
        H(t+1) = [H(t) \\ D] ∪ N
        
        Used in evolution: in_process.merge(current) → new_current
        Tracks (D, R, N) triple for conservation analysis.
        """
        if self.config != other.config:
            raise ValueError("Cannot merge HRTs with different configs")
        
        # Track evolution triple (D, R, N)
        self_hashes = set(b.hllset.name for b in self.lattice.row_basic + self.lattice.col_basic)
        other_hashes = set(b.hllset.name for b in other.lattice.row_basic + other.lattice.col_basic)
        
        deleted = other_hashes - self_hashes
        retained = self_hashes & other_hashes
        new = self_hashes - other_hashes
        
        evo_triple = EvolutionTriple(
            deleted=deleted,
            retained=retained,
            new=new
        )
        
        # Merge adjacency matrices
        merged_am = self.am.merge(other.am)
        
        # Merge lattices
        merged_lattice = self.lattice.merge(other.lattice, kernel)
        
        # Combine covers
        merged_covers = tuple(set(self.covers) | set(other.covers))
        
        # Compute Noether current with uniform distribution
        import torch
        uniform_data = torch.ones(self.config.dimension, dtype=torch.float32) / self.config.dimension
        uniform_dist = ImmutableTensor.from_tensor(uniform_data)
        new_step = max(self.step_number, other.step_number) + 1
        noether = NoetherCurrent.compute(merged_am, uniform_dist, new_step)
        
        return HRT(
            config=self.config,
            am=merged_am,
            lattice=merged_lattice,
            parent_hrt=self.name,  # This HRT becomes the parent
            step_number=new_step,
            covers=merged_covers,
            noether_current=noether,
            evolution_triple=evo_triple
        )
    
    def with_cover(self, cover: Cover) -> HRT:
        """Return new HRT with additional cover."""
        new_covers = tuple(set(self.covers) | {cover})
        return HRT(
            config=self.config,
            am=self.am,
            lattice=self.lattice,
            parent_hrt=self.parent_hrt,
            step_number=self.step_number,
            covers=new_covers
        )
    
    def project_future(self, col_indices: List[int]) -> ImmutableTensor:
        """
        Project columns to rows (future).
        
        From paper Section 6.1:
        p_forward = normalize(A · p)
        """
        return self.am.project_rows(col_indices)
    
    def project_past(self, row_indices: List[int]) -> ImmutableTensor:
        """
        Project rows to columns (past/retro).
        
        From paper Section 6.1:
        p_retro = normalize(A^T · p)
        
        Enables time-reversible dynamics.
        """
        return self.am.project_cols(row_indices)
    
    def contextual_select(self, context_index: int, is_row: bool = True) -> ContextualSelection:
        """
        Apply contextual selection from a basic HLLSet.
        
        From paper Section 7: Contextual Selection Principle
        
        **The fundamental inversion**: Context actively selects compatible elements.
        
        **τ/ρ as mechanics**: The thresholds define HOW context determines what fits "in it":
        - τ (tau): Minimum similarity required (inclusion threshold)
        - ρ (rho): Maximum dissimilarity tolerated (exclusion threshold)
        - Selection: BSS_τ(context, x) ≥ τ AND BSS_ρ(context, x) ≤ ρ
        
        This is not passive membership - the context precedes and determines content!
        
        Args:
            context_index: Index of the context basic HLLSet
            is_row: Whether context is from row or column basics
        
        Returns:
            ContextualSelection with selected candidates
        """
        if is_row:
            context = self.lattice.row_basic[context_index]
            candidates = list(self.lattice.col_basic)
        else:
            context = self.lattice.col_basic[context_index]
            candidates = list(self.lattice.row_basic)
        
        return ContextualSelection.from_context(context, candidates)
    
    def check_entanglement(self, other: HRT, epsilon: Optional[float] = None) -> Tuple[bool, float]:
        """
        Check if this HRT is ε-isomorphic (entangled) with another.
        
        From paper Section 4: Entanglement via structural isomorphism.
        
        Returns:
            (is_entangled, failure_probability)
        """
        is_iso = self.lattice.is_epsilon_isomorphic(other.lattice, epsilon)
        
        # Estimate from typical dataset sizes
        num_datasets = 2
        dataset_size = 1000  # Conservative estimate
        prob_failure = self.lattice.entanglement_probability(num_datasets, dataset_size)
        
        return is_iso, prob_failure
    
    def select_next_by_priority(self, current_index: int, is_row: bool = True, 
                                strategy: str = 'greedy') -> Optional[Tuple[int, float]]:
        """
        Select next basic HLLSet using priority weighting via BSS(τ, ρ).
        
        **Key insights from theory**:
        1. AM shows raw co-occurrence (tokens appeared together)
        2. W shows semantic connections (BSS thresholds satisfied)
        3. τ/ρ act as comprehensive priority weights for path selection
        4. **τ/ρ are the mechanics of contextual selection** - they define how
           a context actively selects what fits "in it"
        
        Process:
        - Current HLLSet (context) uses its τ/ρ thresholds
        - Evaluates each candidate via BSS_τ and BSS_ρ
        - Selects only those satisfying: BSS_τ ≥ τ AND BSS_ρ ≤ ρ
        - Ranks valid candidates by priority = BSS_τ - BSS_ρ
        - Context determines what is "in it" (not passive membership!)
        
        Args:
            current_index: Index of current basic HLLSet (the context)
            is_row: Whether current is from row or column basics
            strategy: Selection strategy ('greedy', 'stochastic')
        
        Returns:
            (selected_index, priority_score) or None if no valid candidates
        """
        if is_row:
            current = self.lattice.row_basic[current_index]
            candidates = list(enumerate(self.lattice.col_basic))
        else:
            current = self.lattice.col_basic[current_index]
            candidates = list(enumerate(self.lattice.row_basic))
        
        # Compute priorities for all candidates
        priorities = []
        for idx, candidate in candidates:
            if candidate.hllset.cardinality() == 0:
                continue  # Skip empty sets
            
            bss_tau = current.bss_tau(candidate)
            bss_rho = current.bss_rho(candidate)
            
            # Check thresholds
            if bss_tau >= self.config.tau and bss_rho <= self.config.rho:
                # Priority = inclusion - exclusion (higher is better)
                priority = bss_tau - bss_rho
                priorities.append((idx, priority, bss_tau, bss_rho))
        
        if not priorities:
            return None
        
        if strategy == 'greedy':
            # Select highest priority
            selected = max(priorities, key=lambda x: x[1])
            return (selected[0], selected[1])
        
        elif strategy == 'stochastic':
            # Sample proportional to priority (softmax)
            import math
            scores = [p[1] for p in priorities]
            max_score = max(scores)
            exp_scores = [math.exp(s - max_score) for s in scores]  # Numerical stability
            total = sum(exp_scores)
            probs = [e / total for e in exp_scores]
            
            # Sample
            import random
            r = random.random()
            cumulative = 0.0
            for i, prob in enumerate(probs):
                cumulative += prob
                if r <= cumulative:
                    return (priorities[i][0], priorities[i][1])
            
            # Fallback to last
            return (priorities[-1][0], priorities[-1][1])
        
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    def find_path_by_priority(self, start_index: int, end_index: int, 
                             max_steps: int = 100) -> Optional[List[Tuple[int, float]]]:
        """
        Find path from start to end using priority-weighted selection.
        
        Demonstrates how τ/ρ thresholds filter AM connectivity:
        - AM may show edges between all nodes (raw co-occurrence)
        - W only allows edges satisfying BSS(τ, ρ) thresholds
        - This creates semantic filtering for meaningful paths
        
        Returns:
            List of (index, priority) tuples representing the path, or None
        """
        path = [(start_index, 1.0)]  # Start with initial node
        current_idx = start_index
        visited = {start_index}
        is_row = True  # Alternate between row and column
        
        for step in range(max_steps):
            if current_idx == end_index:
                return path  # Found path to end
            
            # Select next node by priority
            result = self.select_next_by_priority(current_idx, is_row=is_row, strategy='greedy')
            
            if result is None:
                return None  # No valid next node (filtered by τ/ρ)
            
            next_idx, priority = result
            
            if next_idx in visited:
                return None  # Cycle detected
            
            path.append((next_idx, priority))
            visited.add(next_idx)
            current_idx = next_idx
            is_row = not is_row  # Alternate
        
        return None  # Max steps exceeded
    
    def find_cover(self, target_hash: str) -> Optional[Cover]:
        """Find cached cover for target HLLSet hash."""
        for cover in self.covers:
            if cover.target_hash == target_hash:
                return cover
        return None
    
    def compute_cover(self, target: HLLSet, kernel) -> Cover:
        """
        Compute optimal cover for target HLLSet.
        Greedy algorithm based on similarity.
        """
        selected_rows = set()
        selected_cols = set()
        
        # Greedy selection based on similarity
        for basic in self.lattice.row_basic:
            sim = target.similarity(basic.hllset)
            if sim > 0.1:
                selected_rows.add(basic.index)
        
        for basic in self.lattice.col_basic:
            sim = target.similarity(basic.hllset)
            if sim > 0.1:
                selected_cols.add(basic.index)
        
        return Cover(
            row_indices=frozenset(selected_rows),
            col_indices=frozenset(selected_cols),
            target_hash=target.name
        )
    
    @property
    def am_hash(self) -> str:
        """Hash of adjacency matrix."""
        return self.am.name
    
    @property
    def lattice_hash(self) -> str:
        """Hash of lattice."""
        return self.lattice.name
    
    @property
    def name(self) -> str:
        """Content-addressed name of HRT."""
        components = [
            self.am_hash,
            self.lattice_hash,
            str(self.step_number),
            self.parent_hrt or "genesis"
        ]
        return compute_structural_hash(*components)
    
    def conservation_health(self) -> Optional[str]:
        """
        Check conservation health using Noether current.
        
        From paper Section 6.2:
        System is healthy when flux Φ ≈ 0 (conserved)
        Drift indicates hash collisions or numerical errors.
        """
        if self.noether_current is None:
            return None
        
        flux = abs(self.noether_current.flux)
        if flux < 1e-6:
            return "HEALTHY: Flux ≈ 0 (perfect conservation)"
        elif flux < 1e-3:
            return f"GOOD: Small flux {flux:.6f}"
        elif flux < 0.1:
            return f"WARNING: Moderate flux {flux:.6f}"
        else:
            return f"ALERT: Large flux {flux:.6f} - check for collisions"
    
    def __repr__(self) -> str:
        extras = []
        if self.noether_current:
            extras.append(f"Φ={self.noether_current.flux:.3f}")
        if self.evolution_triple:
            extras.append(f"ΔQ={self.evolution_triple.cardinality_change}")
        extra_str = ", " + ", ".join(extras) if extras else ""
        return f"HRT({self.name[:16]}..., step={self.step_number}, dim={self.config.dimension}{extra_str})"
    
    def __hash__(self) -> int:
        return hash(self.name)
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, HRT):
            return False
        return self.name == other.name


# =============================================================================
# SECTION 9: HRT Evolution Manager
# =============================================================================

@dataclass(frozen=True)
class HRTEvolutionTriple:
    """
    Three-state model for HRT evolution:
    - in_process: Newly ingested data (not yet merged)
    - current: Active state in memory
    - history_hash: Hash of last committed state in Git
    """
    in_process: Optional[HRT]
    current: HRT
    history_hash: str
    step_number: int = 0
    
    def __repr__(self) -> str:
        return (f"HRTEvolutionTriple(step={self.step_number}, "
                f"in_process={'exists' if self.in_process else 'None'}, "
                f"current={self.current.name[:8]}..., "
                f"history={self.history_hash[:8]}...)")


class HRTEvolution:
    """
    Manages HRT evolution through the three-state cycle.
    
    Evolution Step:
    1. Ingest → set in_process
    2. Evolve → merge in_process+current, commit current to history
    3. Cycle repeats
    """
    
    def __init__(self, config: HRTConfig, genesis_hrt: Optional[HRT] = None):
        self.config = config
        self.triple = HRTEvolutionTriple(
            in_process=None,
            current=genesis_hrt or HRT.empty(config),
            history_hash=(genesis_hrt.name if genesis_hrt else "genesis"),
            step_number=0
        )
        self._history: List[str] = [self.triple.history_hash]
    
    def ingest(self, perceptron_data: Dict[str, Set[str]], kernel) -> HRTEvolutionTriple:
        """
        Ingest new perceptron data.
        Creates in_process HRT from data.
        """
        if self.triple.in_process is not None:
            raise RuntimeError("Cannot ingest: in_process already exists. Must evolve first.")
        
        in_process = HRT.from_perceptrons(
            perceptron_data=perceptron_data,
            kernel=kernel,
            config=self.config,
            parent_hrt=self.triple.current.name,
            step_number=self.triple.step_number + 1
        )
        
        self.triple = HRTEvolutionTriple(
            in_process=in_process,
            current=self.triple.current,
            history_hash=self.triple.history_hash,
            step_number=self.triple.step_number
        )
        return self.triple
    
    def evolve(self, kernel, commit_fn: callable) -> HRTEvolutionTriple:
        """
        Execute evolution step.
        
        Args:
            kernel: Kernel for HLLSet operations
            commit_fn: Function to commit HRT to persistent store
        
        Returns:
            Updated evolution triple
        """
        if self.triple.in_process is None:
            raise RuntimeError("Cannot evolve: no in_process data. Must ingest first.")
        
        # Step 1: Merge in_process with current
        new_current = self.triple.in_process.merge(self.triple.current, kernel)
        
        # Step 2: Commit current to history
        new_history_hash = commit_fn(self.triple.current)
        
        # Step 3: Update triple
        self.triple = HRTEvolutionTriple(
            in_process=None,
            current=new_current,
            history_hash=new_history_hash,
            step_number=self.triple.step_number + 1
        )
        self._history.append(new_history_hash)
        
        return self.triple
    
    def get_lineage(self) -> List[str]:
        """Get full commit history."""
        return list(self._history)
    
    def get_current(self) -> HRT:
        """Get current HRT."""
        return self.triple.current
    
    def get_in_process(self) -> Optional[HRT]:
        """Get in_process HRT if exists."""
        return self.triple.in_process
    
    def project_next_state(self, tau_proj: float, rho_proj: float, kernel) -> Tuple[Optional[HRT], float, Dict]:
        """
        Project possible next state based on BSS structure of current HRT.
        
        This is NON-DETERMINISTIC: different (tau_proj, rho_proj) produce different projections.
        
        Args:
            tau_proj: Projection inclusion threshold [0, 1]
            rho_proj: Projection exclusion threshold [0, 1]
            kernel: Kernel for HLLSet operations
        
        Returns:
            Tuple of:
            - projected_hrt: Predicted next HRT state (or None if no strong connections)
            - confidence: Projection confidence [0, 1]
            - metrics: Dict with projection statistics
        """
        current = self.triple.current
        
        # Can't project from empty HRT
        if not current.lattices:
            return None, 0.0, {"reason": "empty_hrt", "connections": 0}
        
        # Analyze BSS connections in each lattice
        all_connections = []
        all_strengths = []
        
        for name, lattice in current.lattices.items():
            if not lattice.row_basic or not lattice.col_basic:
                continue
            
            # Check all row-column basic pairs
            for r in lattice.row_basic:
                for c in lattice.col_basic:
                    bss_tau = r.bss_tau(c, kernel)
                    bss_rho = r.bss_rho(c, kernel)
                    
                    # Connection passes thresholds → likely to persist/strengthen
                    if bss_tau >= tau_proj and bss_rho <= rho_proj:
                        strength = bss_tau * (1 - bss_rho)  # Combined strength
                        all_connections.append((name, r, c, bss_tau, bss_rho, strength))
                        all_strengths.append(strength)
        
        # No strong connections → uncertain projection
        if not all_connections:
            return None, 0.0, {
                "reason": "no_strong_connections",
                "connections": 0,
                "tau_proj": tau_proj,
                "rho_proj": rho_proj
            }
        
        # Compute confidence from BSS distribution
        import numpy as np
        mean_strength = np.mean(all_strengths)
        std_strength = np.std(all_strengths) if len(all_strengths) > 1 else 0.0
        
        # High mean, low variance → high confidence
        # Low mean, high variance → low confidence
        confidence = mean_strength * (1 - min(std_strength, 1.0))
        
        # Build projection: Keep strong connections, predict their evolution
        # This is a simplified projection - in practice would build new HRT
        # For now, return metrics showing what would be retained
        
        projected_connections_by_lattice = {}
        for name, r, c, bss_tau, bss_rho, strength in all_connections:
            if name not in projected_connections_by_lattice:
                projected_connections_by_lattice[name] = []
            projected_connections_by_lattice[name].append({
                'row_tokens': len(r.data),
                'col_tokens': len(c.data),
                'bss_tau': bss_tau,
                'bss_rho': bss_rho,
                'strength': strength
            })
        
        metrics = {
            "connections": len(all_connections),
            "confidence": confidence,
            "mean_strength": mean_strength,
            "std_strength": std_strength,
            "tau_proj": tau_proj,
            "rho_proj": rho_proj,
            "lattices": projected_connections_by_lattice
        }
        
        # For full implementation, would construct projected HRT here
        # For now, return None but with full metrics
        return None, confidence, metrics
    
    def compute_noether_current(self, kernel) -> Optional[NoetherCurrent]:
        """
        Compute Noether current for last evolution step.
        
        Requires:
        - in_process exists (ephemeral state)
        - current exists (committed state)
        - history exists (previous state)
        
        Returns:
            NoetherCurrent with flux Φ = (|N| - |D|) / (|R| + |N|)
        """
        if self.triple.in_process is None:
            return None  # No evolution step to measure
        
        current = self.triple.current
        ephemeral = self.triple.in_process
        
        # Compute what would be deleted, retained, new
        deleted_tokens = set()
        retained_tokens = set()
        new_tokens = set()
        
        # Analyze all lattices
        for name in set(current.lattices.keys()) | set(ephemeral.lattices.keys()):
            current_lattice = current.lattices.get(name)
            ephemeral_lattice = ephemeral.lattices.get(name)
            
            # Get all tokens from both
            current_tokens = set()
            ephemeral_tokens = set()
            
            if current_lattice:
                for r in current_lattice.row_basic:
                    current_tokens.update(r.data)
                for c in current_lattice.col_basic:
                    current_tokens.update(c.data)
            
            if ephemeral_lattice:
                for r in ephemeral_lattice.row_basic:
                    ephemeral_tokens.update(r.data)
                for c in ephemeral_lattice.col_basic:
                    ephemeral_tokens.update(c.data)
            
            # Classify tokens
            deleted_tokens.update(current_tokens - ephemeral_tokens)
            retained_tokens.update(current_tokens & ephemeral_tokens)
            new_tokens.update(ephemeral_tokens - current_tokens)
        
        # Build EvolutionTriple
        evolution_triple = EvolutionTriple(
            deleted=deleted_tokens,
            retained=retained_tokens,
            new=new_tokens
        )
        
        # Compute flux
        total_after = len(retained_tokens) + len(new_tokens)
        if total_after == 0:
            flux = 0.0  # Degenerate case
        else:
            flux = (len(new_tokens) - len(deleted_tokens)) / total_after
        
        return NoetherCurrent(
            flux=flux,
            timestamp=self.triple.step_number,
            step_number=self.triple.step_number,
            evolution_triple=evolution_triple
        )
    
    def analyze_sustainability(self, target_flux: float = 0.0, tolerance: float = 0.1) -> Dict:
        """
        Analyze system sustainability based on Noether current.
        
        Args:
            target_flux: Target Φ value (default 0.0 = conservation)
            tolerance: Acceptable deviation from target
        
        Returns:
            Dict with sustainability analysis
        """
        current = self.compute_noether_current(None)
        
        if current is None:
            return {
                "sustainable": None,
                "reason": "no_evolution_step",
                "recommendation": "ingest data first"
            }
        
        flux = current.flux
        deviation = abs(flux - target_flux)
        sustainable = deviation <= tolerance
        
        analysis = {
            "sustainable": sustainable,
            "flux": flux,
            "target": target_flux,
            "deviation": deviation,
            "tolerance": tolerance,
            "evolution": {
                "deleted": len(current.evolution_triple.deleted),
                "retained": len(current.evolution_triple.retained),
                "new": len(current.evolution_triple.new),
                "cardinality_change": current.evolution_triple.cardinality_change
            }
        }
        
        # Recommendations
        if flux > target_flux + tolerance:
            analysis["recommendation"] = "too_much_growth: increase tau, decrease rho (more selective)"
        elif flux < target_flux - tolerance:
            analysis["recommendation"] = "too_much_decay: decrease tau, increase rho (less selective)"
        else:
            analysis["recommendation"] = "balanced: maintain current parameters"
        
        return analysis


# =============================================================================
# SECTION 10: Example Usage
# =============================================================================

def main():
    """Example HRT usage with immutable tensors."""
    from .kernel import Kernel
    
    print("="*70)
    print("HASH RELATIONAL TENSOR (HRT) - Immutable PyTorch Implementation")
    print("="*70)
    
    # Create kernel
    kernel = Kernel()
    
    # Create config
    config = HRTConfig(p_bits=8, h_bits=16)
    print(f"\nConfig: dim={config.dimension}, basic_sets={config.num_basic_hllsets}")
    
    # Create genesis HRT
    print("\n1. Genesis State")
    print("-" * 40)
    
    genesis = HRT.empty(config)
    print(f"Genesis: {genesis}")
    print(f"AM: {genesis.am}")
    print(f"Lattice: {genesis.lattice}")
    
    # Create evolution manager
    print("\n2. Evolution Manager")
    print("-" * 40)
    
    evolution = HRTEvolution(config, genesis)
    print(f"Initial: {evolution.triple}")
    
    # Ingest data
    print("\n3. Ingest Perceptron Data")
    print("-" * 40)
    
    perceptron_data = {
        "visual": {"red", "green", "blue"},
        "audio": {"low", "mid", "high"},
    }
    
    evolution.ingest(perceptron_data, kernel)
    print(f"After ingest: {evolution.triple}")
    print(f"In-process HRT: {evolution.triple.in_process}")
    print(f"AM entries: {len(evolution.triple.in_process.am.nonzero_entries())}")
    
    # Evolve
    print("\n4. Evolution Step")
    print("-" * 40)
    
    def commit_fn(hrt: HRT) -> str:
        """Mock commit - would save to Git."""
        print(f"  Committing HRT {hrt.name[:16]}... to history")
        return hrt.name
    
    evolution.evolve(kernel, commit_fn)
    print(f"After evolve: {evolution.triple}")
    print(f"Lineage: {[h[:8] for h in evolution.get_lineage()]}")
    
    # Another cycle
    print("\n5. Another Evolution Cycle")
    print("-" * 40)
    
    more_data = {"tactile": {"soft", "hard", "rough"}}
    evolution.ingest(more_data, kernel)
    evolution.evolve(kernel, commit_fn)
    print(f"After second evolve: {evolution.triple}")
    
    # Test immutability
    print("\n6. Immutability Verification")
    print("-" * 40)
    
    current = evolution.get_current()
    modified_am = current.am.with_entry(0, 0, 5.0)
    
    print(f"Original AM: {current.am.name[:16]}...")
    print(f"Modified AM: {modified_am.name[:16]}...")
    print(f"Same? {current.am.name == modified_am.name}")
    print(f"Current unchanged? {evolution.get_current().am.name == current.am.name}")
    
    # Projections
    print("\n7. Future/Past Projections (Time-Reversible)")
    print("-" * 40)
    
    future = evolution.triple.current.project_future([2, 3, 4])
    past = evolution.triple.current.project_past([0, 1])
    print(f"Future projection shape: {future.shape}")
    print(f"Past projection shape: {past.shape}")
    
    # Conservation analysis
    print("\n8. Conservation Analysis (Noether Current)")
    print("-" * 40)
    
    current_hrt = evolution.get_current()
    if current_hrt.noether_current:
        print(f"Noether current: {current_hrt.noether_current}")
        print(f"Health: {current_hrt.conservation_health()}")
    
    if current_hrt.evolution_triple:
        print(f"Evolution triple: {current_hrt.evolution_triple}")
        print(f"Conservative? {current_hrt.evolution_triple.is_conservative}")
    
    # Contextual selection
    print("\n9. Contextual Selection Principle")
    print("-" * 40)
    
    print("The Fundamental Inversion:")
    print("  Context ACTIVELY SELECTS compatible elements")
    print("  τ/ρ are the MECHANICS of this selection")
    print()
    
    selection = current_hrt.contextual_select(context_index=2, is_row=True)
    print(f"Selection: {selection}")
    print(f"Context selects {len(selection.selected_hashes)} compatible elements")
    print()
    print(f"How it works:")
    print(f"  1. Context (r_2) uses τ={config.tau}, ρ={config.rho}")
    print(f"  2. For each candidate x: compute BSS_τ(C,x) and BSS_ρ(C,x)")
    print(f"  3. Select if BSS_τ ≥ {config.tau} AND BSS_ρ ≤ {config.rho}")
    print(f"  4. Context determines what is 'in it' (not passive membership!)")
    
    # BSS morphisms
    print("\n10. Bell State Similarity (BSS) Morphisms")
    print("-" * 40)
    
    basic_a = current_hrt.lattice.row_basic[2]
    basic_b = current_hrt.lattice.row_basic[3]
    
    bss_tau = basic_a.bss_tau(basic_b)
    bss_rho = basic_a.bss_rho(basic_b)
    has_morphism = basic_a.has_morphism_to(basic_b)
    
    print(f"BSS_τ({basic_a.name} → {basic_b.name}) = {bss_tau:.3f}")
    print(f"BSS_ρ({basic_a.name} → {basic_b.name}) = {bss_rho:.3f}")
    print(f"Morphism exists? {has_morphism} (τ={config.tau}, ρ={config.rho})")
    
    # Priority-weighted path selection
    print("\n11. Priority-Weighted Path Selection (AM ≠ W)")
    print("-" * 40)
    
    print("Completing the Picture:")
    print("  • Contextual Selection Principle: Context actively selects")
    print("  • τ/ρ Mechanics: BSS thresholds implement the selection")
    print("  • Result: AM ≠ W (raw co-occurrence ≠ semantic connection)")
    print()
    print(f"AM shows: Raw co-occurrence (tokens appeared together)")
    print(f"W shows:  Semantic connections (BSS(τ, ρ) satisfied)")
    print(f"τ/ρ = Mechanics making context select what fits 'in it'")
    
    # Try to select next from current position
    start_idx = 5
    next_result = current_hrt.select_next_by_priority(start_idx, is_row=True, strategy='greedy')
    
    if next_result:
        next_idx, priority = next_result
        print(f"\nFrom row basic r_{start_idx}:")
        print(f"  Selected: column basic c_{next_idx}")
        print(f"  Priority score: {priority:.3f}")
        print(f"  (BSS_τ - BSS_ρ, filtered by τ={config.tau}, ρ={config.rho})")
    else:
        print(f"\nFrom row basic r_{start_idx}:")
        print(f"  No valid candidates (all filtered by τ/ρ thresholds)")
        print(f"  This shows W disconnects nodes that AM might connect!")
    
    # Try to find a path
    path_result = current_hrt.find_path_by_priority(start_index=2, end_index=10, max_steps=20)
    
    if path_result:
        print(f"\nPath from index 2 to 10:")
        print(f"  Steps: {len(path_result)}")
        print(f"  Path (index, priority): {[(idx, f'{p:.2f}') for idx, p in path_result[:5]]}")
        if len(path_result) > 5:
            print(f"  ... ({len(path_result) - 5} more steps)")
    else:
        print(f"\nNo path found from 2 to 10")
        print(f"  τ/ρ thresholds may have disconnected the path in W!")
    
    print("\n" + "="*70)
    print("HRT Evolution Ready")
    print("="*70)
    
    return evolution


if __name__ == "__main__":
    main()
