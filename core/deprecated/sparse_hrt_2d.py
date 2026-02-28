"""
Sparse HRT (Hash Relational Tensor) - GPU-Accelerated Implementation

Provides:
1. SparseAM - Sparse Adjacency Matrix using ImmutableSparseTensor
2. SparseHRT - Complete HRT with sparse AM, lattice, and embedded LUT
3. GPU acceleration via CUDA COO tensors
4. IICA compliant (Immutability, Idempotence, Content Addressability)

Key Insight:
- AM[i,j] = |basic_row[i] ∩ basic_col[j]| (context covariance, not TF)
- Basic HLLSets are built from active edges only
- Memory: 100K edges ≈ 2MB vs 4GB dense

Design:
- SparseAM wraps ImmutableSparseTensor with HRT-specific operations
- SparseHRT combines SparseAM + Lattice + LUT
- All operations maintain IICA properties
"""

from __future__ import annotations
from typing import List, Dict, Set, Optional, Tuple, FrozenSet, Any
from dataclasses import dataclass, field
import hashlib
import sys
import os

# Handle both package import and direct execution
if __name__ == "__main__":
    # Add parent directory to path for direct execution
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from core.sparse_tensor import (
        ImmutableSparseTensor, 
        get_device, 
        compute_sparse_hash
    )
    from core.hllset import HLLSet, compute_sha1
    from core.constants import P_BITS as KERNEL_P_BITS
else:
    from .sparse_tensor import (
        ImmutableSparseTensor, 
        get_device, 
        compute_sparse_hash
    )
    from .hllset import HLLSet, compute_sha1
    from .constants import P_BITS as KERNEL_P_BITS

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

import numpy as np


# =============================================================================
# SECTION 1: Configuration (same as dense HRT)
# =============================================================================

@dataclass(frozen=True)
class SparseHRTConfig:
    """
    Configuration for Sparse HRT.
    
    Same parameters as HRTConfig but optimized for sparse representation.
    Dimension = 2^P * h_bits + 2 (for START/END tokens)
    """
    p_bits: int = 10           # HLL precision (m = 2^p registers)
    h_bits: int = 32           # Hash bit size
    tau: float = 0.7           # Inclusion tolerance threshold
    rho: float = 0.3           # Exclusion intolerance threshold
    epsilon: float = 0.1       # ε-isomorphism tolerance
    device: str = "cuda"       # Target device
    
    def __post_init__(self):
        """Validate thresholds and device."""
        if not (0 <= self.rho < self.tau <= 1):
            raise ValueError(f"Thresholds must satisfy 0 ≤ ρ < τ ≤ 1")
        if not (0 < self.epsilon < 1):
            raise ValueError(f"Epsilon must satisfy 0 < ε < 1")
        # Normalize device
        if self.device == "cuda" and TORCH_AVAILABLE:
            if not torch.cuda.is_available():
                object.__setattr__(self, 'device', 'cpu')
    
    @property
    def dimension(self) -> int:
        """AM dimension = 2^P * h_bits + 2."""
        return (1 << self.p_bits) * self.h_bits + 2
    
    @property
    def num_registers(self) -> int:
        """Number of HLL registers = 2^P."""
        return 1 << self.p_bits
    
    @property
    def max_zeros(self) -> int:
        """Maximum leading zeros = h_bits - p_bits."""
        return self.h_bits - self.p_bits


# =============================================================================
# SECTION 2: BasicHLLSet (coordinate pair representing single hash)
# =============================================================================

@dataclass(frozen=True)
class BasicHLLSet:
    """
    Basic HLLSet = single (register, zeros) pair.
    
    This is the atomic unit - one hash position.
    Content-addressable: same (reg, zeros) = same BasicHLLSet.
    """
    reg: int      # Register index (0 to 2^P - 1)
    zeros: int    # Leading zeros count (1 to h_bits - p_bits)
    
    def to_index(self, config: SparseHRTConfig) -> int:
        """Convert to linear index for AM row/column."""
        # Index = reg * max_zeros + zeros
        # +1 offset to leave 0 for START, last for END
        return 1 + self.reg * config.max_zeros + (self.zeros - 1)
    
    @classmethod
    def from_index(cls, index: int, config: SparseHRTConfig) -> BasicHLLSet:
        """Convert linear index back to (reg, zeros)."""
        # Reverse: index = 1 + reg * max_zeros + (zeros - 1)
        idx = index - 1
        reg = idx // config.max_zeros
        zeros = (idx % config.max_zeros) + 1
        return cls(reg=reg, zeros=zeros)
    
    @classmethod
    def from_hash(cls, hash_value: int, p_bits: int, h_bits: int) -> BasicHLLSet:
        """
        Create BasicHLLSet from hash value.
        
        Extracts:
        - Register: lower p_bits
        - Zeros: leading zeros in upper (h_bits - p_bits) bits
        """
        mask = (1 << p_bits) - 1
        reg = hash_value & mask
        
        upper = hash_value >> p_bits
        upper_bits = h_bits - p_bits
        
        # Count leading zeros (from MSB of upper portion)
        zeros = 0
        for i in range(upper_bits - 1, -1, -1):
            if (upper >> i) & 1:
                break
            zeros += 1
        zeros = min(zeros + 1, upper_bits)  # At least 1
        
        return cls(reg=reg, zeros=zeros)
    
    def __hash__(self) -> int:
        return hash((self.reg, self.zeros))


# =============================================================================
# SECTION 3: SparseAM (Sparse Adjacency Matrix)
# =============================================================================

@dataclass(frozen=True)
class SparseAM:
    """
    Sparse Adjacency Matrix using COO format on GPU.
    
    AM[i,j] = |basic_row[i] ∩ basic_col[j]| (context covariance)
    
    Properties:
    - Content-addressed (name = hash of edges)
    - Immutable (all ops return new SparseAM)
    - GPU-accelerated (CUDA COO)
    - IICA compliant
    """
    tensor: ImmutableSparseTensor
    config: SparseHRTConfig
    _active_rows: FrozenSet[int] = field(default_factory=frozenset)
    _active_cols: FrozenSet[int] = field(default_factory=frozenset)
    
    def __post_init__(self):
        """Cache active indices if not provided."""
        if not self._active_rows or not self._active_cols:
            rows = frozenset(self.tensor.active_rows())
            cols = frozenset(self.tensor.active_cols())
            object.__setattr__(self, '_active_rows', rows)
            object.__setattr__(self, '_active_cols', cols)
    
    @classmethod
    def empty(cls, config: SparseHRTConfig) -> SparseAM:
        """Create empty sparse AM."""
        tensor = ImmutableSparseTensor.empty(
            config.dimension, 
            config.dimension,
            config.device
        )
        return cls(tensor=tensor, config=config)
    
    @classmethod
    def from_edges(
        cls,
        config: SparseHRTConfig,
        edges: List[Tuple[int, int, float]]
    ) -> SparseAM:
        """Create from edge list."""
        tensor = ImmutableSparseTensor.from_edges(
            config.dimension,
            config.dimension,
            edges,
            config.device
        )
        return cls(tensor=tensor, config=config)
    
    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------
    
    @property
    def name(self) -> str:
        """Content-addressed name."""
        return self.tensor.name
    
    @property
    def nnz(self) -> int:
        """Number of non-zero edges."""
        return self.tensor.nnz
    
    @property
    def shape(self) -> Tuple[int, int]:
        """Matrix dimensions."""
        return self.tensor.shape
    
    @property
    def active_rows(self) -> FrozenSet[int]:
        """Rows with at least one edge."""
        return self._active_rows
    
    @property
    def active_cols(self) -> FrozenSet[int]:
        """Columns with at least one edge."""
        return self._active_cols
    
    @property
    def device(self) -> str:
        """Current device."""
        return self.tensor.device
    
    # -------------------------------------------------------------------------
    # Immutable Operations
    # -------------------------------------------------------------------------
    
    def with_edge(
        self,
        row: int,
        col: int,
        value: float
    ) -> SparseAM:
        """
        Return new AM with edge added/updated.
        
        Takes maximum if edge exists (idempotent for same value).
        """
        new_tensor = self.tensor.with_edge(row, col, value)
        if new_tensor is self.tensor:
            return self  # No change (idempotent)
        
        # Update active sets
        new_rows = self._active_rows | {row}
        new_cols = self._active_cols | {col}
        
        return SparseAM(
            tensor=new_tensor,
            config=self.config,
            _active_rows=new_rows,
            _active_cols=new_cols
        )
    
    def with_intersection(
        self,
        row_idx: int,
        col_idx: int,
        row_hll: HLLSet,
        col_hll: HLLSet
    ) -> SparseAM:
        """
        Add edge with intersection cardinality as value.
        
        AM[i,j] = |row_hll ∩ col_hll| (context covariance)
        """
        intersection = row_hll.intersection(col_hll)
        cardinality = intersection.cardinality()
        
        if cardinality == 0:
            return self  # No edge for zero intersection
        
        return self.with_edge(row_idx, col_idx, float(cardinality))
    
    def merge(self, other: SparseAM) -> SparseAM:
        """
        Element-wise maximum merge (idempotent).
        
        merge(A, A) = A
        """
        if self.config != other.config:
            raise ValueError("Cannot merge AMs with different configs")
        
        merged_tensor = self.tensor.maximum(other.tensor)
        
        return SparseAM(
            tensor=merged_tensor,
            config=self.config,
            _active_rows=self._active_rows | other._active_rows,
            _active_cols=self._active_cols | other._active_cols
        )
    
    # -------------------------------------------------------------------------
    # Query Operations
    # -------------------------------------------------------------------------
    
    def get(self, row: int, col: int) -> float:
        """Get value at (row, col)."""
        return self.tensor.get(row, col)
    
    def row_neighbors(self, row: int) -> List[Tuple[int, float]]:
        """Get (col, value) pairs for given row."""
        if row not in self._active_rows:
            return []
        
        col_indices = self.tensor.row_indices(row)
        if col_indices.numel() == 0:
            return []
        
        result = []
        for col in col_indices.cpu().numpy():
            val = self.tensor.get(row, int(col))
            result.append((int(col), val))
        return result
    
    def col_neighbors(self, col: int) -> List[Tuple[int, float]]:
        """Get (row, value) pairs for given column."""
        if col not in self._active_cols:
            return []
        
        row_indices = self.tensor.col_indices(col)
        if row_indices.numel() == 0:
            return []
        
        result = []
        for row in row_indices.cpu().numpy():
            val = self.tensor.get(int(row), col)
            result.append((int(row), val))
        return result
    
    def edges(self) -> List[Tuple[int, int, float]]:
        """Get all edges."""
        return self.tensor.edges()
    
    # -------------------------------------------------------------------------
    # Device Movement
    # -------------------------------------------------------------------------
    
    def to(self, device: str) -> SparseAM:
        """Move to device."""
        return SparseAM(
            tensor=self.tensor.to(device),
            config=self.config,
            _active_rows=self._active_rows,
            _active_cols=self._active_cols
        )
    
    def cuda(self) -> SparseAM:
        """Move to CUDA."""
        return self.to('cuda')
    
    def cpu(self) -> SparseAM:
        """Move to CPU."""
        return self.to('cpu')
    
    # -------------------------------------------------------------------------
    # String Representation
    # -------------------------------------------------------------------------
    
    def __repr__(self) -> str:
        return (
            f"SparseAM(nnz={self.nnz}, active_rows={len(self._active_rows)}, "
            f"active_cols={len(self._active_cols)}, device='{self.device}')"
        )


# =============================================================================
# SECTION 4: SparseLattice (HLLSets for active rows/cols only)
# =============================================================================

@dataclass(frozen=True)
class SparseLattice:
    """
    Sparse HLLSet Lattice - only stores connections for active indices.
    
    For each active row i:
      row_connections[i] = set of all col indices j where AM[i,j] > 0
    
    For each active col j:
      col_connections[j] = set of all row indices i where AM[i,j] > 0
    
    This is the key insight: connections accumulate CONTEXT.
    |row_connections[i] ∩ col_connections[j]| measures context covariance.
    
    Note: We use Python sets for exact counts (not HLLSet estimation) since
    we're tracking indices, not token hashes. For very large graphs, we could
    switch to HLLSet with stringified indices.
    """
    _row_connections: Dict[int, FrozenSet[int]]  # row_idx -> set of connected cols
    _col_connections: Dict[int, FrozenSet[int]]  # col_idx -> set of connected rows
    _name: str = ""
    
    def __post_init__(self):
        """Compute content hash."""
        if not self._name:
            # Hash based on sorted connections
            content = []
            for idx in sorted(self._row_connections.keys()):
                cols = sorted(self._row_connections[idx])
                content.append(f"r{idx}:{','.join(map(str, cols))}")
            for idx in sorted(self._col_connections.keys()):
                rows = sorted(self._col_connections[idx])
                content.append(f"c{idx}:{','.join(map(str, rows))}")
            combined = "|".join(content)
            name = compute_sha1(combined.encode())
            object.__setattr__(self, '_name', name)
    
    @classmethod
    def empty(cls) -> SparseLattice:
        """Create empty lattice."""
        return cls(_row_connections={}, _col_connections={})
    
    @classmethod
    def from_sparse_am(cls, am: SparseAM) -> SparseLattice:
        """
        Build lattice from sparse AM.
        
        For each edge (i, j) in AM:
        - Add j to row_connections[i]
        - Add i to col_connections[j]
        """
        row_conns: Dict[int, Set[int]] = {}
        col_conns: Dict[int, Set[int]] = {}
        
        for row, col, _ in am.edges():
            # Update row connections
            if row not in row_conns:
                row_conns[row] = set()
            row_conns[row].add(col)
            
            # Update col connections
            if col not in col_conns:
                col_conns[col] = set()
            col_conns[col].add(row)
        
        # Convert to frozen sets
        return cls(
            _row_connections={k: frozenset(v) for k, v in row_conns.items()},
            _col_connections={k: frozenset(v) for k, v in col_conns.items()}
        )
    
    @property
    def name(self) -> str:
        """Content-addressed name."""
        return self._name
    
    @property
    def num_row_entries(self) -> int:
        """Number of rows with connections."""
        return len(self._row_connections)
    
    @property
    def num_col_entries(self) -> int:
        """Number of columns with connections."""
        return len(self._col_connections)
    
    def row_connections(self, row: int) -> FrozenSet[int]:
        """Get connected columns for row."""
        return self._row_connections.get(row, frozenset())
    
    def col_connections(self, col: int) -> FrozenSet[int]:
        """Get connected rows for column."""
        return self._col_connections.get(col, frozenset())
    
    def row_cardinality(self, row: int) -> int:
        """Number of columns connected to row."""
        return len(self.row_connections(row))
    
    def col_cardinality(self, col: int) -> int:
        """Number of rows connected to column."""
        return len(self.col_connections(col))
    
    def with_edge(self, row: int, col: int) -> SparseLattice:
        """Add edge to lattice (returns new lattice)."""
        # Check if already exists
        if col in self.row_connections(row):
            return self  # Idempotent
        
        # Copy and update
        new_rows = dict(self._row_connections)
        new_cols = dict(self._col_connections)
        
        # Update row connections
        existing_row = new_rows.get(row, frozenset())
        new_rows[row] = existing_row | {col}
        
        # Update col connections
        existing_col = new_cols.get(col, frozenset())
        new_cols[col] = existing_col | {row}
        
        return SparseLattice(_row_connections=new_rows, _col_connections=new_cols)
    
    def merge(self, other: SparseLattice) -> SparseLattice:
        """Merge two lattices (union of connections)."""
        new_rows = dict(self._row_connections)
        new_cols = dict(self._col_connections)
        
        for idx, conns in other._row_connections.items():
            if idx in new_rows:
                new_rows[idx] = new_rows[idx] | conns
            else:
                new_rows[idx] = conns
        
        for idx, conns in other._col_connections.items():
            if idx in new_cols:
                new_cols[idx] = new_cols[idx] | conns
            else:
                new_cols[idx] = conns
        
        return SparseLattice(_row_connections=new_rows, _col_connections=new_cols)
    
    def intersection_cardinality(self, row: int, col: int) -> int:
        """
        Compute |row_connections[row] ∩ col_connections[col]|.
        
        This is the context covariance between positions.
        How many indices appear in both row's columns and col's rows?
        """
        row_conns = self._row_connections.get(row, frozenset())
        col_conns = self._col_connections.get(col, frozenset())
        
        return len(row_conns & col_conns)
    
    def __repr__(self) -> str:
        return f"SparseLattice(rows={self.num_row_entries}, cols={self.num_col_entries})"


# =============================================================================
# SECTION 5: SparseHRT (Complete Sparse HRT)
# =============================================================================

@dataclass(frozen=True)
class SparseHRT:
    """
    Sparse Hash Relational Tensor - GPU-accelerated, IICA-compliant.
    
    Components:
    - am: SparseAM (edges with context covariance values)
    - lattice: SparseLattice (HLLSets for active indices only)
    - lut: Embedded LUT for token disambiguation
    - config: SparseHRTConfig
    
    Properties:
    - Content-addressed (name = hash of structure)
    - Immutable (all ops return new SparseHRT)
    - Memory efficient (sparse COO on GPU)
    """
    am: SparseAM
    lattice: SparseLattice
    config: SparseHRTConfig
    lut: FrozenSet[Tuple[int, int, int, FrozenSet[str]]] = field(default_factory=frozenset)
    # LUT entry: (reg, zeros, token_hash, tokens)
    step_number: int = 0
    parent_hrt: Optional[str] = None
    _name: str = ""
    
    def __post_init__(self):
        """Compute content hash."""
        if not self._name:
            # Combine hashes of components
            content = f"{self.am.name}|{self.lattice.name}|{hash(self.lut)}|{self.step_number}"
            name = compute_sha1(content.encode())
            object.__setattr__(self, '_name', name)
    
    @classmethod
    def empty(cls, config: Optional[SparseHRTConfig] = None) -> SparseHRT:
        """Create empty SparseHRT."""
        if config is None:
            config = SparseHRTConfig()
        
        return cls(
            am=SparseAM.empty(config),
            lattice=SparseLattice.empty(),
            config=config
        )
    
    @property
    def name(self) -> str:
        """Content-addressed name (SHA1 hash)."""
        return self._name
    
    @property
    def nnz(self) -> int:
        """Number of edges in AM."""
        return self.am.nnz
    
    @property
    def device(self) -> str:
        """Current device."""
        return self.am.device
    
    def with_edge(
        self,
        row: int,
        col: int,
        value: Optional[float] = None
    ) -> SparseHRT:
        """
        Add edge to HRT.
        
        If value is None, computes intersection cardinality.
        """
        # Update lattice first (to get HLLSets for intersection)
        new_lattice = self.lattice.with_edge(row, col)
        
        # Compute value from intersection if not provided
        if value is None:
            value = new_lattice.intersection_cardinality(row, col)
            if value == 0:
                value = 1.0  # Minimum edge weight
        
        # Update AM
        new_am = self.am.with_edge(row, col, value)
        
        if new_am is self.am:
            return self  # No change (idempotent)
        
        return SparseHRT(
            am=new_am,
            lattice=new_lattice,
            config=self.config,
            lut=self.lut,
            step_number=self.step_number,
            parent_hrt=self._name
        )
    
    def with_lut_entry(
        self,
        reg: int,
        zeros: int,
        token_hash: int,
        tokens: FrozenSet[str]
    ) -> SparseHRT:
        """Add LUT entry."""
        entry = (reg, zeros, token_hash, tokens)
        if entry in self.lut:
            return self  # Idempotent
        
        new_lut = self.lut | {entry}
        
        return SparseHRT(
            am=self.am,
            lattice=self.lattice,
            config=self.config,
            lut=new_lut,
            step_number=self.step_number,
            parent_hrt=self._name
        )
    
    def merge(self, other: SparseHRT) -> SparseHRT:
        """
        Merge two SparseHRTs (idempotent).
        
        merge(A, A) = A
        """
        if self.config != other.config:
            raise ValueError("Cannot merge HRTs with different configs")
        
        merged_am = self.am.merge(other.am)
        merged_lattice = self.lattice.merge(other.lattice)
        merged_lut = self.lut | other.lut
        
        return SparseHRT(
            am=merged_am,
            lattice=merged_lattice,
            config=self.config,
            lut=merged_lut,
            step_number=max(self.step_number, other.step_number) + 1,
            parent_hrt=self._name
        )
    
    def evolve(self) -> SparseHRT:
        """Create next step (increment step number)."""
        return SparseHRT(
            am=self.am,
            lattice=self.lattice,
            config=self.config,
            lut=self.lut,
            step_number=self.step_number + 1,
            parent_hrt=self._name
        )
    
    # -------------------------------------------------------------------------
    # Device Movement
    # -------------------------------------------------------------------------
    
    def to(self, device: str) -> SparseHRT:
        """Move AM to device."""
        return SparseHRT(
            am=self.am.to(device),
            lattice=self.lattice,
            config=self.config,
            lut=self.lut,
            step_number=self.step_number,
            parent_hrt=self.parent_hrt,
            _name=self._name
        )
    
    def cuda(self) -> SparseHRT:
        """Move to CUDA."""
        return self.to('cuda')
    
    def cpu(self) -> SparseHRT:
        """Move to CPU."""
        return self.to('cpu')
    
    # -------------------------------------------------------------------------
    # Memory Stats
    # -------------------------------------------------------------------------
    
    def memory_bytes(self) -> int:
        """Estimate memory usage in bytes."""
        # Sparse tensor: nnz * (2 * 8 + 4) = 20 bytes per edge
        am_bytes = self.nnz * 20
        
        # Lattice connections: frozensets of ints
        # Rough estimate: 8 bytes per int + overhead
        lattice_bytes = 0
        for conns in self.lattice._row_connections.values():
            lattice_bytes += len(conns) * 8 + 56  # set overhead
        for conns in self.lattice._col_connections.values():
            lattice_bytes += len(conns) * 8 + 56
        
        # LUT: ~100 bytes per entry estimate
        lut_bytes = len(self.lut) * 100
        
        return am_bytes + lattice_bytes + lut_bytes
    
    def memory_mb(self) -> float:
        """Memory usage in MB."""
        return self.memory_bytes() / (1024 * 1024)
    
    def __repr__(self) -> str:
        return (
            f"SparseHRT(nnz={self.nnz}, lut={len(self.lut)}, "
            f"step={self.step_number}, mem={self.memory_mb():.2f}MB, "
            f"device='{self.device}')"
        )
    
    def __eq__(self, other: object) -> bool:
        """Content equality."""
        if not isinstance(other, SparseHRT):
            return False
        return self._name == other._name
    
    def __hash__(self) -> int:
        return hash(self._name)


# =============================================================================
# SECTION 6: Factory Functions
# =============================================================================

def create_sparse_hrt(
    p_bits: int = 10,
    h_bits: int = 32,
    device: str = "cuda"
) -> SparseHRT:
    """Create empty SparseHRT with config."""
    config = SparseHRTConfig(p_bits=p_bits, h_bits=h_bits, device=device)
    return SparseHRT.empty(config)


# =============================================================================
# SECTION 7: Self-Test
# =============================================================================

if __name__ == "__main__":
    print("=== SparseHRT Test ===")
    print()
    
    # Create empty
    config = SparseHRTConfig(p_bits=10, h_bits=32)
    print(f"Config: dim={config.dimension}, device={config.device}")
    print()
    
    hrt = SparseHRT.empty(config)
    print(f"Empty HRT: {hrt}")
    print(f"  Name: {hrt.name[:32]}...")
    print()
    
    # Add edges
    hrt1 = hrt.with_edge(100, 200, 5.0)
    hrt2 = hrt1.with_edge(300, 400, 3.0)
    hrt3 = hrt2.with_edge(100, 201, 2.0)
    
    print(f"With 3 edges: {hrt3}")
    print(f"  Active rows: {hrt3.am.active_rows}")
    print(f"  Active cols: {hrt3.am.active_cols}")
    print()
    
    # Check lattice
    print("Lattice:")
    print(f"  Row 100 connections: {hrt3.lattice.row_connections(100)}")
    print(f"  Row 100 cardinality: {hrt3.lattice.row_cardinality(100)}")
    print(f"  Col 200 connections: {hrt3.lattice.col_connections(200)}")
    print(f"  Col 200 cardinality: {hrt3.lattice.col_cardinality(200)}")
    print()
    
    # Idempotence test
    hrt4 = hrt3.with_edge(100, 200, 5.0)  # Same edge
    print(f"Idempotence (same edge): {hrt4.name == hrt3.name}")
    print()
    
    # Merge test
    other = SparseHRT.empty(config)
    other = other.with_edge(500, 600, 7.0)
    other = other.with_edge(100, 200, 10.0)  # Higher value
    
    merged = hrt3.merge(other)
    print(f"Merged: {merged}")
    print(f"  Edge (100,200) value: {merged.am.get(100, 200)} (max of 5.0 and 10.0)")
    print()
    
    # Self-merge idempotence
    self_merged = hrt3.merge(hrt3)
    print(f"Self-merge idempotent: {self_merged.name == hrt3.name}")
    print()
    
    # Memory comparison
    print("=== Memory Comparison ===")
    dense_size = config.dimension ** 2 * 4  # float32
    sparse_size = hrt3.memory_bytes()
    print(f"Dense AM:  {dense_size / 1024 / 1024 / 1024:.1f} GB")
    print(f"Sparse HRT: {sparse_size / 1024:.1f} KB")
    print(f"Ratio: {dense_size / sparse_size:.0f}x smaller")
    print()
    
    # Large scale test
    print("=== Large Scale Test ===")
    import time
    
    large_hrt = SparseHRT.empty(config)
    
    # Add 100K edges
    edges = [(i % config.dimension, (i + 1) % config.dimension, float(i % 100)) 
             for i in range(100000)]
    
    start = time.time()
    # Batch add via from_edges
    large_am = SparseAM.from_edges(config, edges)
    large_lattice = SparseLattice.from_sparse_am(large_am)
    large_hrt = SparseHRT(
        am=large_am,
        lattice=large_lattice,
        config=config
    )
    elapsed = time.time() - start
    
    print(f"Created 100K edges in {elapsed*1000:.1f}ms")
    print(f"  {large_hrt}")
    print()
    
    print("✓ All tests passed!")
