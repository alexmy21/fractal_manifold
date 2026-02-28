"""
Sparse HRT 3D - N-Gram Layered Adjacency Matrix

Upgrade from 2D AM[row, col] to 3D AM[n, row, col]:
- n = 0: 1-grams
- n = 1: 2-grams  
- n = 2: 3-grams
- ...

Key Benefits:
1. No n-gram mixing - each layer is separate
2. Self-loops within layer still possible (hash collision)
3. But cross-n-gram relations are explicit
4. BasicHLLSet aggregates across n dimension
5. Disambiguation simplified - n-gram order preserved in structure

Sliding Window Algorithm:
    1-gram → 2-gram → 3-gram → shift → 1-gram → ...
    
Each step adds edge at appropriate n-layer.

IICA Properties:
- Immutable: all operations return new tensor
- Idempotent: merge(A, A) = A
- Content Addressable: name = SHA1(sorted edges)
"""

from __future__ import annotations
from typing import List, Dict, Set, Optional, Tuple, FrozenSet, Any, NamedTuple
from dataclasses import dataclass, field
import hashlib
import sys
import os

# Handle both package import and direct execution
if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from core.hllset import HLLSet, compute_sha1
    from core.constants import P_BITS as KERNEL_P_BITS
else:
    from .hllset import HLLSet, compute_sha1
    from .constants import P_BITS as KERNEL_P_BITS

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

import numpy as np


# =============================================================================
# SECTION 1: Device Management (consolidated from sparse_tensor.py)
# =============================================================================

def get_device(prefer_cuda: bool = True) -> 'torch.device':
    """Get the best available device."""
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch not available")
    if prefer_cuda and torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def get_default_dtype() -> 'torch.dtype':
    """Default dtype for sparse values."""
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch not available")
    return torch.float32


# =============================================================================
# SECTION 2: Configuration with N-Gram Layers
# =============================================================================

@dataclass(frozen=True)
class Sparse3DConfig:
    """
    Configuration for 3D Sparse HRT.
    
    Shape: (max_n, dimension, dimension)
    - max_n: Maximum n-gram order (e.g., 3 for 1,2,3-grams)
    - dimension: 2^P * h_bits + 2
    """
    p_bits: int = 10           # HLL precision (m = 2^p registers)
    h_bits: int = 32           # Hash bit size  
    max_n: int = 3             # Maximum n-gram order (1, 2, 3, ...)
    tau: float = 0.7           # Inclusion tolerance threshold
    rho: float = 0.3           # Exclusion intolerance threshold
    epsilon: float = 0.1       # ε-isomorphism tolerance
    device: str = "cuda"       # Target device
    
    def __post_init__(self):
        """Validate configuration."""
        if not (0 <= self.rho < self.tau <= 1):
            raise ValueError(f"Thresholds must satisfy 0 ≤ ρ < τ ≤ 1")
        if not (0 < self.epsilon < 1):
            raise ValueError(f"Epsilon must satisfy 0 < ε < 1")
        if self.max_n < 1:
            raise ValueError(f"max_n must be >= 1")
        # Normalize device
        if self.device == "cuda" and TORCH_AVAILABLE:
            if not torch.cuda.is_available():
                object.__setattr__(self, 'device', 'cpu')
    
    @property
    def dimension(self) -> int:
        """AM dimension = 2^P * h_bits + 2 (per layer)."""
        return (1 << self.p_bits) * self.h_bits + 2
    
    @property
    def shape(self) -> Tuple[int, int, int]:
        """3D tensor shape: (max_n, dim, dim)."""
        return (self.max_n, self.dimension, self.dimension)
    
    @property
    def num_registers(self) -> int:
        """Number of HLL registers = 2^P."""
        return 1 << self.p_bits
    
    @property
    def max_zeros(self) -> int:
        """Maximum leading zeros = h_bits - p_bits."""
        return self.h_bits - self.p_bits


# =============================================================================
# SECTION 2: 3D Edge Type
# =============================================================================

class Edge3D(NamedTuple):
    """
    3D edge: (n-gram layer, row, col, value).
    
    n: 0-indexed n-gram layer (0=1-gram, 1=2-gram, 2=3-gram)
    row: Row index in AM layer
    col: Column index in AM layer  
    value: Edge weight (typically intersection cardinality)
    """
    n: int
    row: int
    col: int
    value: float


# =============================================================================
# SECTION 3: BasicHLLSet3D (includes n-gram layer)
# =============================================================================

@dataclass(frozen=True)
class BasicHLLSet3D:
    """
    Basic HLLSet with n-gram layer.
    
    Atomic unit: (n, reg, zeros) triple.
    - n: n-gram order (0-indexed: 0=1-gram, 1=2-gram, etc.)
    - reg: Register index
    - zeros: Leading zeros count
    """
    n: int        # N-gram order (0-indexed)
    reg: int      # Register index (0 to 2^P - 1)
    zeros: int    # Leading zeros count (0 to h_bits - p_bits)
    
    def to_index(self, config: Sparse3DConfig) -> int:
        """Convert (reg, zeros) to linear index for AM row/column.
        
        Uses the same formula as UniversalID.to_index for consistency.
        """
        return self.reg * (config.h_bits - config.p_bits + 1) + self.zeros
    
    def to_3d_index(self, config: Sparse3DConfig) -> Tuple[int, int]:
        """Convert to (layer, linear_index) for 3D AM."""
        return (self.n, self.to_index(config))
    
    @classmethod
    def from_hash(
        cls, 
        hash_value: int, 
        n: int,
        p_bits: int, 
        h_bits: int
    ) -> BasicHLLSet3D:
        """
        Create BasicHLLSet3D from hash value.
        
        Args:
            hash_value: Token hash
            n: N-gram order (0-indexed)
            p_bits: HLL precision bits
            h_bits: Total hash bits
        """
        mask = (1 << p_bits) - 1
        reg = hash_value & mask
        
        upper = hash_value >> p_bits
        upper_bits = h_bits - p_bits
        
        # Count leading zeros
        zeros = 0
        for i in range(upper_bits - 1, -1, -1):
            if (upper >> i) & 1:
                break
            zeros += 1
        zeros = min(zeros + 1, upper_bits)
        
        return cls(n=n, reg=reg, zeros=zeros)
    
    def __hash__(self) -> int:
        return hash((self.n, self.reg, self.zeros))


# =============================================================================
# SECTION 4: 3D Sparse Hash Computation
# =============================================================================

def compute_sparse_3d_hash(
    indices: 'torch.Tensor',
    values: 'torch.Tensor',
    shape: Tuple[int, int, int]
) -> str:
    """
    Compute SHA1 hash for 3D sparse tensor.
    
    indices: [3, nnz] - (n, row, col) for each edge
    values: [nnz] - edge weights
    """
    # Move to CPU for hashing
    if indices.is_cuda:
        indices = indices.cpu()
    if values.is_cuda:
        values = values.cpu()
    
    if indices.numel() > 0:
        idx_np = indices.numpy()
        val_np = values.numpy()
        
        # Sort by (n, row, col) for deterministic hash
        sort_order = np.lexsort((idx_np[2], idx_np[1], idx_np[0]))
        idx_sorted = idx_np[:, sort_order]
        val_sorted = val_np[sort_order]
        
        content = (
            np.array(shape, dtype=np.int64).tobytes() +
            idx_sorted.tobytes() +
            val_sorted.tobytes()
        )
    else:
        content = np.array(shape, dtype=np.int64).tobytes()
    
    return hashlib.sha1(content).hexdigest()


# =============================================================================
# SECTION 5: ImmutableSparseTensor3D
# =============================================================================

@dataclass(frozen=True)
class ImmutableSparseTensor3D:
    """
    Immutable 3D sparse tensor using PyTorch COO format.
    
    Shape: (max_n, rows, cols)
    COO Format:
    - indices: [3, nnz] tensor of (n, row, col) triples
    - values: [nnz] tensor of values
    
    IICA compliant.
    """
    _indices: 'torch.Tensor'  # [3, nnz] int64
    _values: 'torch.Tensor'   # [nnz] float32
    _shape: Tuple[int, int, int]  # (max_n, rows, cols)
    _name: str = field(default="", compare=False)
    _device: str = field(default="cpu", compare=False)
    
    def __post_init__(self):
        """Compute content hash if not provided."""
        if not self._name:
            name = compute_sparse_3d_hash(self._indices, self._values, self._shape)
            object.__setattr__(self, '_name', name)
        object.__setattr__(self, '_device', str(self._indices.device))
    
    @classmethod
    def empty(
        cls,
        max_n: int,
        rows: int,
        cols: int,
        device: Optional[str] = None
    ) -> ImmutableSparseTensor3D:
        """Create empty 3D sparse tensor."""
        if device is None:
            device = str(get_device())
        
        indices = torch.empty((3, 0), dtype=torch.int64, device=device)
        values = torch.empty((0,), dtype=get_default_dtype(), device=device)
        
        return cls(
            _indices=indices,
            _values=values,
            _shape=(max_n, rows, cols)
        )
    
    @classmethod
    def from_edges(
        cls,
        max_n: int,
        rows: int,
        cols: int,
        edges: List[Edge3D],
        device: Optional[str] = None
    ) -> ImmutableSparseTensor3D:
        """Create from edge list."""
        if device is None:
            device = str(get_device())
        
        if not edges:
            return cls.empty(max_n, rows, cols, device)
        
        n_indices = []
        row_indices = []
        col_indices = []
        values = []
        
        for edge in edges:
            if 0 <= edge.n < max_n:
                n_indices.append(edge.n)
                row_indices.append(edge.row)
                col_indices.append(edge.col)
                values.append(edge.value)
        
        if not n_indices:
            return cls.empty(max_n, rows, cols, device)
        
        indices = torch.tensor(
            [n_indices, row_indices, col_indices],
            dtype=torch.int64,
            device=device
        )
        vals = torch.tensor(values, dtype=get_default_dtype(), device=device)
        
        return cls(
            _indices=indices,
            _values=vals,
            _shape=(max_n, rows, cols)
        )
    
    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------
    
    @property
    def name(self) -> str:
        """Content-addressed name."""
        return self._name
    
    @property
    def shape(self) -> Tuple[int, int, int]:
        """Tensor dimensions (max_n, rows, cols)."""
        return self._shape
    
    @property
    def max_n(self) -> int:
        """Maximum n-gram order."""
        return self._shape[0]
    
    @property
    def nnz(self) -> int:
        """Number of non-zero entries."""
        return self._values.numel()
    
    @property
    def indices(self) -> 'torch.Tensor':
        """Read-only indices [3, nnz]."""
        return self._indices
    
    @property
    def values(self) -> 'torch.Tensor':
        """Read-only values [nnz]."""
        return self._values
    
    @property
    def device(self) -> str:
        """Current device."""
        return self._device
    
    @property
    def is_cuda(self) -> bool:
        """Check if on GPU."""
        return 'cuda' in self._device
    
    # -------------------------------------------------------------------------
    # Query Operations
    # -------------------------------------------------------------------------
    
    def get(self, n: int, row: int, col: int) -> float:
        """Get value at (n, row, col)."""
        if self.nnz == 0:
            return 0.0
        
        # Find matching indices
        n_match = self._indices[0] == n
        row_match = self._indices[1] == row
        col_match = self._indices[2] == col
        mask = n_match & row_match & col_match
        
        if mask.any():
            return float(self._values[mask][0].item())
        return 0.0
    
    def edges(self) -> List[Edge3D]:
        """Get all edges as list of Edge3D."""
        if self.nnz == 0:
            return []
        
        indices = self._indices.cpu().numpy()
        values = self._values.cpu().numpy()
        
        return [
            Edge3D(int(indices[0, i]), int(indices[1, i]), int(indices[2, i]), float(values[i]))
            for i in range(self.nnz)
        ]
    
    def layer_edges(self, n: int) -> List[Tuple[int, int, float]]:
        """Get edges for specific n-gram layer."""
        if self.nnz == 0:
            return []
        
        mask = self._indices[0] == n
        if not mask.any():
            return []
        
        rows = self._indices[1, mask].cpu().numpy()
        cols = self._indices[2, mask].cpu().numpy()
        vals = self._values[mask].cpu().numpy()
        
        return [
            (int(rows[i]), int(cols[i]), float(vals[i]))
            for i in range(len(rows))
        ]
    
    def active_in_layer(self, n: int) -> Tuple[Set[int], Set[int]]:
        """Get active rows and cols in a specific layer."""
        if self.nnz == 0:
            return set(), set()
        
        mask = self._indices[0] == n
        if not mask.any():
            return set(), set()
        
        rows = set(self._indices[1, mask].cpu().numpy().tolist())
        cols = set(self._indices[2, mask].cpu().numpy().tolist())
        return rows, cols
    
    def active_across_layers(self) -> Tuple[Set[int], Set[int]]:
        """Get active rows and cols across ALL layers (for BasicHLLSet)."""
        if self.nnz == 0:
            return set(), set()
        
        rows = set(self._indices[1].cpu().numpy().tolist())
        cols = set(self._indices[2].cpu().numpy().tolist())
        return rows, cols
    
    # -------------------------------------------------------------------------
    # Immutable Operations
    # -------------------------------------------------------------------------
    
    def with_edge(
        self,
        n: int,
        row: int,
        col: int,
        value: float
    ) -> ImmutableSparseTensor3D:
        """Return new tensor with edge added/updated (max semantics)."""
        if n < 0 or n >= self.max_n:
            raise ValueError(f"n={n} out of range [0, {self.max_n})")
        
        existing = self.get(n, row, col)
        if value <= existing:
            return self  # No change (idempotent for same or lower value)
        
        if self.nnz == 0:
            # First edge
            indices = torch.tensor(
                [[n], [row], [col]],
                dtype=torch.int64,
                device=self._device
            )
            values = torch.tensor([value], dtype=get_default_dtype(), device=self._device)
        else:
            # Find if edge exists
            n_match = self._indices[0] == n
            row_match = self._indices[1] == row
            col_match = self._indices[2] == col
            mask = n_match & row_match & col_match
            
            if mask.any():
                # Update existing
                new_values = self._values.clone()
                new_values[mask] = value
                indices = self._indices
                values = new_values
            else:
                # Add new edge
                new_idx = torch.tensor(
                    [[n], [row], [col]],
                    dtype=torch.int64,
                    device=self._device
                )
                new_val = torch.tensor([value], dtype=get_default_dtype(), device=self._device)
                indices = torch.cat([self._indices, new_idx], dim=1)
                values = torch.cat([self._values, new_val])
        
        return ImmutableSparseTensor3D(
            _indices=indices,
            _values=values,
            _shape=self._shape
        )
    
    def maximum(self, other: ImmutableSparseTensor3D) -> ImmutableSparseTensor3D:
        """Element-wise maximum merge (idempotent)."""
        if self._shape != other._shape:
            raise ValueError(f"Shape mismatch: {self._shape} vs {other._shape}")
        
        if self.nnz == 0:
            return other
        if other.nnz == 0:
            return self
        
        # Combine all edges
        all_indices = torch.cat([self._indices, other._indices], dim=1)
        all_values = torch.cat([self._values, other._values])
        
        # Group by (n, row, col) and take max
        # Convert to tuple keys for grouping
        indices_np = all_indices.cpu().numpy()
        values_np = all_values.cpu().numpy()
        
        edge_dict = {}
        for i in range(indices_np.shape[1]):
            key = (int(indices_np[0, i]), int(indices_np[1, i]), int(indices_np[2, i]))
            edge_dict[key] = max(edge_dict.get(key, 0.0), float(values_np[i]))
        
        # Rebuild tensor
        if not edge_dict:
            return ImmutableSparseTensor3D.empty(
                self.max_n, self._shape[1], self._shape[2], self._device
            )
        
        n_indices = []
        row_indices = []
        col_indices = []
        values = []
        for (n, r, c), v in edge_dict.items():
            n_indices.append(n)
            row_indices.append(r)
            col_indices.append(c)
            values.append(v)
        
        new_indices = torch.tensor(
            [n_indices, row_indices, col_indices],
            dtype=torch.int64,
            device=self._device
        )
        new_values = torch.tensor(values, dtype=get_default_dtype(), device=self._device)
        
        return ImmutableSparseTensor3D(
            _indices=new_indices,
            _values=new_values,
            _shape=self._shape
        )
    
    def to(self, device: str) -> ImmutableSparseTensor3D:
        """Move to different device."""
        if device == self._device:
            return self
        return ImmutableSparseTensor3D(
            _indices=self._indices.to(device),
            _values=self._values.to(device),
            _shape=self._shape
        )
    
    def memory_bytes(self) -> int:
        """Estimate memory usage."""
        # 3 int64 per edge + 1 float32
        return self.nnz * (3 * 8 + 4)
    
    def memory_mb(self) -> float:
        """Memory in megabytes."""
        return self.memory_bytes() / (1024 * 1024)
    
    def __repr__(self) -> str:
        return (
            f"ImmutableSparseTensor3D(shape={self._shape}, nnz={self.nnz}, "
            f"device={self._device}, name={self._name[:16]}...)"
        )


# =============================================================================
# SECTION 6: SparseAM3D
# =============================================================================

@dataclass(frozen=True)
class SparseAM3D:
    """
    3D Sparse Adjacency Matrix.
    
    AM[n, row, col] = context covariance at n-gram layer
    
    n=0: 1-gram relations
    n=1: 2-gram relations
    n=2: 3-gram relations
    
    BasicHLLSets aggregate across all layers.
    """
    tensor: ImmutableSparseTensor3D
    config: Sparse3DConfig
    # Active indices per layer
    _layer_active: Tuple[Tuple[FrozenSet[int], FrozenSet[int]], ...] = field(default_factory=tuple)
    # Active indices across all layers (for BasicHLLSet)
    _all_active_rows: FrozenSet[int] = field(default_factory=frozenset)
    _all_active_cols: FrozenSet[int] = field(default_factory=frozenset)
    
    def __post_init__(self):
        """Cache active indices."""
        if not self._layer_active:
            layer_active = []
            all_rows = set()
            all_cols = set()
            
            for n in range(self.config.max_n):
                rows, cols = self.tensor.active_in_layer(n)
                layer_active.append((frozenset(rows), frozenset(cols)))
                all_rows.update(rows)
                all_cols.update(cols)
            
            object.__setattr__(self, '_layer_active', tuple(layer_active))
            object.__setattr__(self, '_all_active_rows', frozenset(all_rows))
            object.__setattr__(self, '_all_active_cols', frozenset(all_cols))
    
    @classmethod
    def empty(cls, config: Sparse3DConfig) -> SparseAM3D:
        """Create empty 3D sparse AM."""
        tensor = ImmutableSparseTensor3D.empty(
            config.max_n,
            config.dimension,
            config.dimension,
            config.device
        )
        return cls(tensor=tensor, config=config)
    
    @classmethod
    def from_edges(
        cls,
        config: Sparse3DConfig,
        edges: List[Edge3D]
    ) -> SparseAM3D:
        """Create from edge list."""
        tensor = ImmutableSparseTensor3D.from_edges(
            config.max_n,
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
        """Total non-zero edges across all layers."""
        return self.tensor.nnz
    
    @property
    def shape(self) -> Tuple[int, int, int]:
        """3D shape: (max_n, dim, dim)."""
        return self.tensor.shape
    
    @property
    def device(self) -> str:
        """Current device."""
        return self.tensor.device
    
    @property
    def all_active_rows(self) -> FrozenSet[int]:
        """Rows active in ANY layer (for BasicHLLSet)."""
        return self._all_active_rows
    
    @property
    def all_active_cols(self) -> FrozenSet[int]:
        """Columns active in ANY layer (for BasicHLLSet)."""
        return self._all_active_cols
    
    def layer_active(self, n: int) -> Tuple[FrozenSet[int], FrozenSet[int]]:
        """Get (active_rows, active_cols) for specific layer."""
        if 0 <= n < len(self._layer_active):
            return self._layer_active[n]
        return frozenset(), frozenset()
    
    def layer_nnz(self, n: int) -> int:
        """Count edges in specific layer."""
        return len(self.tensor.layer_edges(n))
    
    # -------------------------------------------------------------------------
    # Immutable Operations
    # -------------------------------------------------------------------------
    
    def with_edge(
        self,
        n: int,
        row: int,
        col: int,
        value: float
    ) -> SparseAM3D:
        """Add edge at (n, row, col) with max semantics."""
        new_tensor = self.tensor.with_edge(n, row, col, value)
        if new_tensor is self.tensor:
            return self  # No change
        
        return SparseAM3D(tensor=new_tensor, config=self.config)
    
    def with_ngram_edge(
        self,
        ngram_size: int,  # 1, 2, 3, ...
        row: int,
        col: int,
        value: float
    ) -> SparseAM3D:
        """Add edge for n-gram (1-indexed n-gram size)."""
        n = ngram_size - 1  # Convert to 0-indexed layer
        return self.with_edge(n, row, col, value)
    
    def merge(self, other: SparseAM3D) -> SparseAM3D:
        """Element-wise maximum merge (idempotent)."""
        if self.config != other.config:
            raise ValueError("Cannot merge AMs with different configs")
        
        merged_tensor = self.tensor.maximum(other.tensor)
        return SparseAM3D(tensor=merged_tensor, config=self.config)
    
    # -------------------------------------------------------------------------
    # Query Operations
    # -------------------------------------------------------------------------
    
    def get(self, n: int, row: int, col: int) -> float:
        """Get value at (n, row, col)."""
        return self.tensor.get(n, row, col)
    
    def layer_edges(self, n: int) -> List[Tuple[int, int, float]]:
        """Get all edges in layer n."""
        return self.tensor.layer_edges(n)
    
    def all_edges(self) -> List[Edge3D]:
        """Get all edges across all layers."""
        return self.tensor.edges()
    
    def row_neighbors_in_layer(self, n: int, row: int) -> List[Tuple[int, float]]:
        """Get (col, value) pairs for row in specific layer."""
        edges = self.tensor.layer_edges(n)
        return [(c, v) for (r, c, v) in edges if r == row]
    
    def col_neighbors_in_layer(self, n: int, col: int) -> List[Tuple[int, float]]:
        """Get (row, value) pairs for col in specific layer."""
        edges = self.tensor.layer_edges(n)
        return [(r, v) for (r, c, v) in edges if c == col]
    
    def row_neighbors_all_layers(self, row: int) -> List[Tuple[int, int, float]]:
        """Get (n, col, value) for row across all layers."""
        result = []
        for n in range(self.config.max_n):
            for col, val in self.row_neighbors_in_layer(n, row):
                result.append((n, col, val))
        return result
    
    def memory_mb(self) -> float:
        """Memory in megabytes."""
        return self.tensor.memory_mb()
    
    def __repr__(self) -> str:
        return (
            f"SparseAM3D(shape={self.shape}, nnz={self.nnz}, "
            f"active_rows={len(self._all_active_rows)}, "
            f"active_cols={len(self._all_active_cols)})"
        )


# =============================================================================
# SECTION 7: SparseLattice3D
# =============================================================================

@dataclass(frozen=True)
class SparseLattice3D:
    """
    3D Sparse Lattice for connection tracking.
    
    Tracks connections per layer AND aggregated across layers.
    
    For BasicHLLSet construction:
    - Aggregate row connections across all layers
    - Aggregate col connections across all layers
    """
    # Per-layer connections
    _layer_row_conns: Tuple[Dict[int, FrozenSet[int]], ...]
    _layer_col_conns: Tuple[Dict[int, FrozenSet[int]], ...]
    # Aggregated connections (for BasicHLLSet)
    _all_row_conns: Dict[int, FrozenSet[int]]
    _all_col_conns: Dict[int, FrozenSet[int]]
    config: Sparse3DConfig
    
    @classmethod
    def from_sparse_am(cls, am: SparseAM3D) -> SparseLattice3D:
        """Build lattice from 3D sparse AM."""
        layer_row_conns = []
        layer_col_conns = []
        all_row_conns: Dict[int, Set[int]] = {}
        all_col_conns: Dict[int, Set[int]] = {}
        
        for n in range(am.config.max_n):
            row_conns: Dict[int, Set[int]] = {}
            col_conns: Dict[int, Set[int]] = {}
            
            for r, c, _ in am.tensor.layer_edges(n):
                # Per-layer
                if r not in row_conns:
                    row_conns[r] = set()
                row_conns[r].add(c)
                
                if c not in col_conns:
                    col_conns[c] = set()
                col_conns[c].add(r)
                
                # Aggregated (with layer info encoded)
                # Option 1: Just aggregate indices
                if r not in all_row_conns:
                    all_row_conns[r] = set()
                all_row_conns[r].add(c)
                
                if c not in all_col_conns:
                    all_col_conns[c] = set()
                all_col_conns[c].add(r)
            
            # Freeze per-layer
            layer_row_conns.append({k: frozenset(v) for k, v in row_conns.items()})
            layer_col_conns.append({k: frozenset(v) for k, v in col_conns.items()})
        
        # Freeze aggregated
        frozen_all_row = {k: frozenset(v) for k, v in all_row_conns.items()}
        frozen_all_col = {k: frozenset(v) for k, v in all_col_conns.items()}
        
        return cls(
            _layer_row_conns=tuple(layer_row_conns),
            _layer_col_conns=tuple(layer_col_conns),
            _all_row_conns=frozen_all_row,
            _all_col_conns=frozen_all_col,
            config=am.config
        )
    
    # -------------------------------------------------------------------------
    # Per-Layer Access
    # -------------------------------------------------------------------------
    
    def layer_row_connections(self, n: int, row: int) -> FrozenSet[int]:
        """Columns connected to row in layer n."""
        if 0 <= n < len(self._layer_row_conns):
            return self._layer_row_conns[n].get(row, frozenset())
        return frozenset()
    
    def layer_col_connections(self, n: int, col: int) -> FrozenSet[int]:
        """Rows connected to col in layer n."""
        if 0 <= n < len(self._layer_col_conns):
            return self._layer_col_conns[n].get(col, frozenset())
        return frozenset()
    
    # -------------------------------------------------------------------------
    # Aggregated Access (for BasicHLLSet)
    # -------------------------------------------------------------------------
    
    def all_row_connections(self, row: int) -> FrozenSet[int]:
        """Columns connected to row across ALL layers."""
        return self._all_row_conns.get(row, frozenset())
    
    def all_col_connections(self, col: int) -> FrozenSet[int]:
        """Rows connected to col across ALL layers."""
        return self._all_col_conns.get(col, frozenset())
    
    def row_cardinality(self, row: int) -> int:
        """Total connections for row across all layers."""
        return len(self.all_row_connections(row))
    
    def col_cardinality(self, col: int) -> int:
        """Total connections for col across all layers."""
        return len(self.all_col_connections(col))
    
    def intersection_cardinality(self, row: int, col: int) -> int:
        """
        Context covariance = |row_conns ∩ col_conns|.
        
        Uses aggregated connections across all layers.
        """
        row_conns = self.all_row_connections(row)
        col_conns = self.all_col_connections(col)
        return len(row_conns & col_conns)
    
    def __repr__(self) -> str:
        total_rows = sum(len(d) for d in self._layer_row_conns)
        total_cols = sum(len(d) for d in self._layer_col_conns)
        return f"SparseLattice3D(layers={len(self._layer_row_conns)}, rows={total_rows}, cols={total_cols})"


# =============================================================================
# SECTION 8: SparseHRT3D
# =============================================================================

@dataclass(frozen=True)
class SparseHRT3D:
    """
    3D Sparse HRT with n-gram layered adjacency matrix.
    
    Components:
    - am: SparseAM3D (n-gram layered)
    - lattice: SparseLattice3D (connection tracking)
    - lut: FrozenSet of LUT entries (for disambiguation)
    - config: Sparse3DConfig
    
    IICA compliant.
    """
    am: SparseAM3D
    lattice: SparseLattice3D
    config: Sparse3DConfig
    lut: FrozenSet[Any] = field(default_factory=frozenset)  # LUT entries
    step: int = 0
    parent_name: Optional[str] = None
    _name: str = field(default="", compare=False)
    
    def __post_init__(self):
        """Compute content-addressed name."""
        if not self._name:
            # Hash: AM hash + LUT hash + step
            lut_hash = hashlib.sha1(
                str(sorted(str(e) for e in self.lut)).encode()
            ).hexdigest()[:16]
            
            combined = f"{self.am.name}:{lut_hash}:{self.step}"
            name = hashlib.sha1(combined.encode()).hexdigest()
            object.__setattr__(self, '_name', name)
    
    @classmethod
    def empty(cls, config: Optional[Sparse3DConfig] = None) -> SparseHRT3D:
        """Create empty 3D HRT."""
        if config is None:
            config = Sparse3DConfig()
        
        am = SparseAM3D.empty(config)
        lattice = SparseLattice3D.from_sparse_am(am)
        
        return cls(am=am, lattice=lattice, config=config)
    
    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------
    
    @property
    def name(self) -> str:
        """Content-addressed name."""
        return self._name
    
    @property
    def nnz(self) -> int:
        """Total edges across all layers."""
        return self.am.nnz
    
    @property
    def shape(self) -> Tuple[int, int, int]:
        """3D shape."""
        return self.am.shape
    
    @property
    def device(self) -> str:
        """Current device."""
        return self.am.device
    
    def layer_stats(self) -> Dict[int, int]:
        """Edge count per layer."""
        return {n: self.am.layer_nnz(n) for n in range(self.config.max_n)}
    
    # -------------------------------------------------------------------------
    # Immutable Operations
    # -------------------------------------------------------------------------
    
    def with_edge(
        self,
        n: int,
        row: int,
        col: int,
        value: float
    ) -> SparseHRT3D:
        """Add edge at (n, row, col)."""
        new_am = self.am.with_edge(n, row, col, value)
        if new_am is self.am:
            return self
        
        new_lattice = SparseLattice3D.from_sparse_am(new_am)
        
        return SparseHRT3D(
            am=new_am,
            lattice=new_lattice,
            config=self.config,
            lut=self.lut,
            step=self.step,
            parent_name=self.parent_name
        )
    
    def with_ngram_edge(
        self,
        ngram_size: int,  # 1, 2, 3, ...
        row: int,
        col: int,
        value: float
    ) -> SparseHRT3D:
        """Add edge for specific n-gram size (1-indexed)."""
        return self.with_edge(ngram_size - 1, row, col, value)
    
    def merge(self, other: SparseHRT3D) -> SparseHRT3D:
        """Merge two HRTs (idempotent)."""
        if self.config != other.config:
            raise ValueError("Cannot merge HRTs with different configs")
        
        merged_am = self.am.merge(other.am)
        merged_lattice = SparseLattice3D.from_sparse_am(merged_am)
        merged_lut = self.lut | other.lut
        
        return SparseHRT3D(
            am=merged_am,
            lattice=merged_lattice,
            config=self.config,
            lut=merged_lut,
            step=max(self.step, other.step) + 1,
            parent_name=self._name
        )
    
    def evolve(self) -> SparseHRT3D:
        """Create next evolution step."""
        return SparseHRT3D(
            am=self.am,
            lattice=self.lattice,
            config=self.config,
            lut=self.lut,
            step=self.step + 1,
            parent_name=self._name
        )
    
    # -------------------------------------------------------------------------
    # BasicHLLSet Construction (Row and Column)
    # -------------------------------------------------------------------------
    
    def basic_hllsets_for_row(self, row: int) -> List[BasicHLLSet3D]:
        """
        Get BasicHLLSet3D entries for a row.
        
        Row's BasicHLLSet = set of (n, reg, zeros) from all connected columns.
        Aggregates across all n-gram layers.
        """
        result = []
        for n in range(self.config.max_n):
            cols = self.lattice.layer_row_connections(n, row)
            for col in cols:
                # Convert column index back to (reg, zeros)
                idx = col - 1
                if idx >= 0:
                    reg = idx // self.config.max_zeros
                    zeros = (idx % self.config.max_zeros) + 1
                    result.append(BasicHLLSet3D(n=n, reg=reg, zeros=zeros))
        return result
    
    def basic_hllsets_for_col(self, col: int) -> List[BasicHLLSet3D]:
        """
        Get BasicHLLSet3D entries for a column.
        
        Column's BasicHLLSet = set of (n, reg, zeros) from all connected rows.
        Aggregates across all n-gram layers.
        """
        result = []
        for n in range(self.config.max_n):
            rows = self.lattice.layer_col_connections(n, col)
            for row in rows:
                # Convert row index back to (reg, zeros)
                idx = row - 1
                if idx >= 0:
                    reg = idx // self.config.max_zeros
                    zeros = (idx % self.config.max_zeros) + 1
                    result.append(BasicHLLSet3D(n=n, reg=reg, zeros=zeros))
        return result
    
    def build_hllset_for_row(self, row: int) -> HLLSet:
        """
        Build aggregated HLLSet for a row.
        
        Combines (reg, zeros) from all connected columns across all layers.
        This is the BasicHLLSet for the row.
        """
        hll = HLLSet(p_bits=self.config.p_bits)
        
        for basic in self.basic_hllsets_for_row(row):
            # Reconstruct a hash-like token from (n, reg, zeros)
            # Include n in the synthetic hash to differentiate layers
            synthetic_hash = (basic.n << 30) | (basic.reg << (self.config.h_bits - self.config.p_bits)) | (basic.zeros - 1)
            hll = HLLSet.add(hll, f"_synthetic_{synthetic_hash}")
        
        return hll
    
    def build_hllset_for_col(self, col: int) -> HLLSet:
        """
        Build aggregated HLLSet for a column.
        
        Combines (reg, zeros) from all connected rows across all layers.
        This is the BasicHLLSet for the column.
        """
        hll = HLLSet(p_bits=self.config.p_bits)
        
        for basic in self.basic_hllsets_for_col(col):
            # Reconstruct a hash-like token from (n, reg, zeros)
            synthetic_hash = (basic.n << 30) | (basic.reg << (self.config.h_bits - self.config.p_bits)) | (basic.zeros - 1)
            hll = HLLSet.add(hll, f"_synthetic_{synthetic_hash}")
        
        return hll
    
    def compute_edge_weight(self, row: int, col: int) -> float:
        """
        Compute edge weight as intersection cardinality of BasicHLLSets.
        
        AM[n, row, col] = |BasicHLLSet(row) ∩ BasicHLLSet(col)|
        
        This is the context covariance: how many positions are shared
        between the row's context and the column's context.
        """
        row_hll = self.build_hllset_for_row(row)
        col_hll = self.build_hllset_for_col(col)
        
        inter = HLLSet.intersect(row_hll, col_hll)
        return inter.cardinality()
    
    def with_computed_edge(
        self,
        n: int,
        row: int,
        col: int
    ) -> SparseHRT3D:
        """
        Add edge with automatically computed weight.
        
        Weight = |BasicHLLSet(row) ∩ BasicHLLSet(col)|
        
        Use this after building the basic structure to fill in
        context covariance weights.
        """
        weight = self.compute_edge_weight(row, col)
        if weight == 0:
            return self  # No edge for zero intersection
        return self.with_edge(n, row, col, weight)
    
    # -------------------------------------------------------------------------
    # SHEAF-BASED RETRIEVAL: Sub-Lattice → Sub-AM → Token Clouds → Candidates
    # -------------------------------------------------------------------------
    
    def extract_sub_lattice(
        self,
        query_basics: List[BasicHLLSet3D]
    ) -> Tuple[FrozenSet[int], SparseLattice3D]:
        """
        STEP 1: Extract sub-lattice from query BasicHLLSets.
        
        Given a decomposed query (list of BasicHLLSet3D), find all
        active rows in the lattice that could match.
        
        Returns:
            (active_rows, sub_lattice restricted to those rows)
        
        This is the PROJECTION onto the query's support.
        """
        # Convert query basics to row indices
        active_rows: Set[int] = set()
        for basic in query_basics:
            idx = basic.to_index(self.config)
            if idx in self.am.all_active_rows:
                active_rows.add(idx)
        
        # The sub-lattice is just the restriction of connections
        # to these active rows - we return the full lattice but
        # the caller should only query the active rows
        return frozenset(active_rows), self.lattice
    
    def extract_sub_am(
        self,
        active_rows: FrozenSet[int]
    ) -> Dict[int, List[Edge3D]]:
        """
        STEP 2: Extract sub-tensor from 3D AM for active rows.
        
        Returns edges grouped by n-gram layer:
            {n: [Edge3D, ...], ...}
        
        Each layer is INDEPENDENT → can be processed in PARALLEL.
        """
        sub_am: Dict[int, List[Edge3D]] = {n: [] for n in range(self.config.max_n)}
        
        for n in range(self.config.max_n):
            for row in active_rows:
                for col, val in self.am.row_neighbors_in_layer(n, row):
                    sub_am[n].append(Edge3D(n=n, row=row, col=col, value=val))
        
        return sub_am
    
    def project_to_clouds(
        self,
        sub_am: Dict[int, List[Edge3D]],
        lut_lookup: Optional[Dict[int, Set[str]]] = None
    ) -> Dict[int, Set[int]]:
        """
        STEP 3: Project sub-AM to token clouds (sheaves).
        
        For each n-gram layer, collect all column indices (potential tokens).
        
        Args:
            sub_am: Edges grouped by layer
            lut_lookup: Optional col_index → {tokens} mapping for disambiguation
        
        Returns:
            {n: set of column indices (token candidates)}
        
        Each cloud is a SHEAF over the sub-lattice!
        Higher n → more context → smaller cloud (restriction map).
        """
        clouds: Dict[int, Set[int]] = {}
        
        for n in range(self.config.max_n):
            layer_edges = sub_am.get(n, [])
            # Collect unique columns with their weights
            cloud: Set[int] = set()
            for edge in layer_edges:
                cloud.add(edge.col)
            clouds[n] = cloud
        
        return clouds
    
    def intersect_clouds(
        self,
        clouds: Dict[int, Set[int]]
    ) -> Set[int]:
        """
        STEP 4: Intersect n-gram clouds → global section.
        
        Tokens appearing in ALL clouds (1-gram, 2-gram, 3-gram)
        are the strongest candidates.
        
        Returns:
            Set of column indices (tokens in all n-gram contexts)
        
        This is the GLOBAL SECTION of the sheaf!
        """
        if not clouds:
            return set()
        
        # Start with union of all non-empty clouds
        non_empty = [c for c in clouds.values() if c]
        if not non_empty:
            return set()
        
        # Intersection
        result = non_empty[0].copy()
        for cloud in non_empty[1:]:
            result &= cloud
        
        return result
    
    def rank_candidates(
        self,
        candidates: Set[int],
        sub_am: Dict[int, List[Edge3D]]
    ) -> List[Tuple[int, float]]:
        """
        Rank candidates by aggregated edge weight from sub-AM.
        
        rank(col) = Σₙ Σ_edges AM[n, row, col]
        
        Returns:
            List of (col_index, score) sorted descending
        """
        scores: Dict[int, float] = {col: 0.0 for col in candidates}
        
        for n, edges in sub_am.items():
            for edge in edges:
                if edge.col in candidates:
                    scores[edge.col] += edge.value
        
        ranked = sorted(scores.items(), key=lambda x: -x[1])
        return ranked
    
    def retrieve(
        self,
        query_basics: List[BasicHLLSet3D],
        top_k: Optional[int] = None,
        require_all_layers: bool = False
    ) -> List[Tuple[int, float, Dict[int, float]]]:
        """
        Full retrieval pipeline using sheaf-based projection.
        
        Pipeline:
        1. Extract sub-lattice from query BasicHLLSets
        2. Extract sub-AM tensor for active rows  
        3. Project sub-AM to token clouds per layer (parallel)
        4. Intersect clouds → candidates with ordering
        
        Args:
            query_basics: Decomposed query as List[BasicHLLSet3D]
            top_k: Return only top K results
            require_all_layers: If True, only return tokens in ALL layer clouds
        
        Returns:
            List of (col_index, total_score, {layer: layer_score})
            Sorted by total_score descending
        
        ┌─────────────────────────────────────────────────────────────────┐
        │  Query HLLSet                                                   │
        │       ↓ decompose                                               │
        │  List[BasicHLLSet3D]                                            │
        │       ↓ extract_sub_lattice (projection p)                      │
        │  Active rows in W                                               │
        │       ↓ extract_sub_am                                          │
        │  Sub-tensor AM[n, active_rows, :]                               │
        │       ↓ project_to_clouds (parallel by n)                       │
        │  {Layer 0: Cloud₀, Layer 1: Cloud₁, Layer 2: Cloud₂}            │
        │       ↓ intersect_clouds (global section)                       │
        │  Candidate tokens                                               │
        │       ↓ rank_candidates (by sub-AM weights)                     │
        │  Ordered results                                                │
        └─────────────────────────────────────────────────────────────────┘
        """
        # Step 1: Extract sub-lattice
        active_rows, sub_lattice = self.extract_sub_lattice(query_basics)
        
        if not active_rows:
            return []
        
        # Step 2: Extract sub-AM
        sub_am = self.extract_sub_am(active_rows)
        
        # Step 3: Project to clouds (PARALLEL per layer)
        clouds = self.project_to_clouds(sub_am)
        
        # Step 4: Determine candidates
        if require_all_layers:
            # Intersection: must appear in ALL non-empty layers
            candidates = self.intersect_clouds(clouds)
        else:
            # Union: appear in ANY layer
            candidates: Set[int] = set()
            for cloud in clouds.values():
                candidates |= cloud
        
        if not candidates:
            return []
        
        # Step 5: Compute per-layer scores
        detailed_scores: Dict[int, Dict[int, float]] = {col: {} for col in candidates}
        total_scores: Dict[int, float] = {col: 0.0 for col in candidates}
        
        for n, edges in sub_am.items():
            for edge in edges:
                if edge.col in candidates:
                    detailed_scores[edge.col][n] = detailed_scores[edge.col].get(n, 0.0) + edge.value
                    total_scores[edge.col] += edge.value
        
        # Step 6: Rank and return
        results = [
            (col, total_scores[col], detailed_scores[col])
            for col in candidates
        ]
        results.sort(key=lambda x: -x[1])
        
        if top_k is not None:
            results = results[:top_k]
        
        return results
    
    def retrieve_from_hllset(
        self,
        query_hll: HLLSet,
        top_k: Optional[int] = None,
        require_all_layers: bool = False
    ) -> List[Tuple[int, float, Dict[int, float]]]:
        """
        Retrieve from a query HLLSet.
        
        First decomposes HLLSet into BasicHLLSet3D (distributing across layers),
        then runs the full retrieval pipeline.
        
        Args:
            query_hll: Query HLLSet
            top_k: Return only top K results
            require_all_layers: If True, only return tokens in ALL layer clouds
        
        Returns:
            List of (col_index, total_score, {layer: layer_score})
        """
        # Decompose HLLSet to BasicHLLSets
        # For now, distribute query elements across all layers equally
        query_basics: List[BasicHLLSet3D] = []
        
        # Extract (reg, zeros) from the HLLSet's data
        for reg, zeros in enumerate(query_hll.data):
            if zeros > 0:  # Only active registers
                # Distribute across all n-gram layers
                for n in range(self.config.max_n):
                    query_basics.append(BasicHLLSet3D(n=n, reg=reg, zeros=zeros))
        
        return self.retrieve(query_basics, top_k, require_all_layers)
    
    def memory_mb(self) -> float:
        """Memory in megabytes."""
        return self.am.memory_mb()
    
    def __repr__(self) -> str:
        return (
            f"SparseHRT3D(shape={self.shape}, nnz={self.nnz}, "
            f"step={self.step}, layers={self.layer_stats()})"
        )


# =============================================================================
# SECTION 9: Factory Functions
# =============================================================================

def create_sparse_hrt_3d(
    p_bits: int = 10,
    h_bits: int = 32,
    max_n: int = 3,
    device: Optional[str] = None
) -> SparseHRT3D:
    """Create empty 3D sparse HRT."""
    if device is None:
        device = str(get_device())
    
    config = Sparse3DConfig(
        p_bits=p_bits,
        h_bits=h_bits,
        max_n=max_n,
        device=device
    )
    
    return SparseHRT3D.empty(config)


# =============================================================================
# SECTION 10: Test
# =============================================================================

if __name__ == "__main__":
    print("=== SparseHRT 3D Test ===\n")
    
    # Create config
    config = Sparse3DConfig(p_bits=10, h_bits=32, max_n=3)
    print(f"Config: p_bits={config.p_bits}, h_bits={config.h_bits}, max_n={config.max_n}")
    print(f"Dimension: {config.dimension}")
    print(f"Shape: {config.shape}")
    print(f"Device: {config.device}")
    print()
    
    # Create empty HRT
    hrt = create_sparse_hrt_3d()
    print(f"Empty HRT: {hrt}")
    print()
    
    # Add edges at different n-gram layers
    print("=== Adding Edges ===")
    hrt1 = hrt.with_ngram_edge(1, 100, 200, 1.0)   # 1-gram
    hrt2 = hrt1.with_ngram_edge(2, 100, 201, 2.0)  # 2-gram
    hrt3 = hrt2.with_ngram_edge(3, 100, 202, 3.0)  # 3-gram
    hrt4 = hrt3.with_ngram_edge(1, 300, 400, 1.0)  # Another 1-gram
    
    print(f"After 4 edges: {hrt4}")
    print(f"Layer stats: {hrt4.layer_stats()}")
    print()
    
    # Query per-layer
    print("=== Per-Layer Queries ===")
    for n in range(3):
        edges = hrt4.am.layer_edges(n)
        print(f"Layer {n} ({n+1}-grams): {len(edges)} edges")
        for r, c, v in edges:
            print(f"  ({r}, {c}) = {v}")
    print()
    
    # Lattice
    print("=== Lattice ===")
    print(f"Row 100 connections (all layers): {hrt4.lattice.all_row_connections(100)}")
    print(f"Col 200 connections (all layers): {hrt4.lattice.all_col_connections(200)}")
    print(f"Row 100 per-layer:")
    for n in range(3):
        print(f"  Layer {n}: {hrt4.lattice.layer_row_connections(n, 100)}")
    print()
    
    # BasicHLLSets for BOTH rows and columns
    print("=== BasicHLLSets for Row 100 ===")
    row_basics = hrt4.basic_hllsets_for_row(100)
    for b in row_basics:
        print(f"  {b}")
    print()
    
    print("=== BasicHLLSets for Col 200 ===")
    col_basics = hrt4.basic_hllsets_for_col(200)
    for b in col_basics:
        print(f"  {b}")
    print()
    
    # Build HLLSets
    print("=== HLLSets from BasicHLLSets ===")
    row_hll = hrt4.build_hllset_for_row(100)
    col_hll = hrt4.build_hllset_for_col(200)
    print(f"Row 100 HLLSet: cardinality={row_hll.cardinality():.1f}")
    print(f"Col 200 HLLSet: cardinality={col_hll.cardinality():.1f}")
    
    # Intersection for edge weight
    inter = HLLSet.intersect(row_hll, col_hll)
    print(f"Intersection cardinality: {inter.cardinality():.1f}")
    print()
    
    # Compute edge weight
    print("=== Edge Weight Computation ===")
    # Add more edges to create overlap
    hrt5 = hrt4.with_ngram_edge(1, 200, 100, 1.0)  # Reverse edge
    hrt6 = hrt5.with_ngram_edge(1, 200, 300, 1.0)  # Col 200 also connects to 300
    
    # Now compute weight between row 100 and col 200
    weight = hrt6.compute_edge_weight(100, 200)
    print(f"Edge weight (100, 200) = |BasicHLLSet(row) ∩ BasicHLLSet(col)| = {weight:.1f}")
    print()
    
    # Memory comparison
    print("=== Memory ===")
    dim = config.dimension
    dense_3d_gb = 3 * dim * dim * 4 / (1024**3)  # 3 layers
    print(f"Dense 3D AM: {dense_3d_gb:.1f} GB")
    print(f"Sparse 3D AM: {hrt4.memory_mb():.4f} MB")
    print()
    
    # Large scale test
    print("=== Large Scale Test (100K edges) ===")
    import time
    
    edges = []
    for i in range(100_000):
        n = i % 3  # Distribute across layers
        row = i % config.dimension
        col = (i + 1) % config.dimension
        edges.append(Edge3D(n, row, col, float(i % 100 + 1)))
    
    start = time.time()
    large_am = SparseAM3D.from_edges(config, edges)
    am_time = time.time() - start
    print(f"Created AM with {large_am.nnz:,} edges in {am_time*1000:.1f}ms")
    print(f"Memory: {large_am.memory_mb():.2f} MB")
    print(f"Layer stats: {', '.join(f'L{n}={large_am.layer_nnz(n):,}' for n in range(3))}")
    print()
    
    print("✓ SparseHRT 3D tests passed!")
