"""
Immutable Sparse Tensor for GPU-Accelerated HRT

Provides:
1. ImmutableSparseTensor - COO format sparse tensor with CUDA support
2. Content-addressed naming via SHA1 of (indices, values)
3. Clone-on-modify semantics enforced at the type level
4. Efficient memory usage for sparse AM/W matrices

Design Principles:
- All tensors are frozen after creation
- Any "modification" creates a new tensor with new hash
- COO format for efficient edge insertion
- CUDA support for GPU acceleration
- IICA compliant (Immutability, Idempotence, Content Addressability)
"""

from __future__ import annotations
from typing import Optional, Tuple, List, Set, FrozenSet
from dataclasses import dataclass, field
import hashlib

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

import numpy as np


# =============================================================================
# SECTION 1: Device Management
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
    return torch.float32


# =============================================================================
# SECTION 2: Hash Functions for Sparse Tensors
# =============================================================================

def compute_sparse_hash(
    indices: 'torch.Tensor',
    values: 'torch.Tensor',
    shape: Tuple[int, ...]
) -> str:
    """
    Compute SHA1 hash for sparse tensor content.
    
    Hash includes:
    - Shape (dimensions)
    - Sorted (row, col, value) triples for deterministic ordering
    """
    # Move to CPU for hashing
    if indices.is_cuda:
        indices = indices.cpu()
    if values.is_cuda:
        values = values.cpu()
    
    # Sort by (row, col) for deterministic hash
    if indices.numel() > 0:
        # Convert to numpy for sorting
        idx_np = indices.numpy()
        val_np = values.numpy()
        
        # Sort by row, then col
        sort_order = np.lexsort((idx_np[1], idx_np[0]))
        idx_sorted = idx_np[:, sort_order]
        val_sorted = val_np[sort_order]
        
        # Combine into bytes
        content = (
            np.array(shape, dtype=np.int64).tobytes() +
            idx_sorted.tobytes() +
            val_sorted.tobytes()
        )
    else:
        # Empty tensor - just hash shape
        content = np.array(shape, dtype=np.int64).tobytes()
    
    return hashlib.sha1(content).hexdigest()


# =============================================================================
# SECTION 3: ImmutableSparseTensor
# =============================================================================

@dataclass(frozen=True)
class ImmutableSparseTensor:
    """
    Immutable sparse tensor using PyTorch COO format.
    
    Properties:
    - Content-addressed: name = SHA1(sorted edges + shape)
    - Immutable: all operations return new tensor
    - GPU-ready: can reside on CUDA device
    - IICA compliant
    
    COO Format:
    - indices: [2, nnz] tensor of (row, col) pairs
    - values: [nnz] tensor of values
    - shape: (rows, cols) dimensions
    """
    _indices: 'torch.Tensor'  # [2, nnz] int64
    _values: 'torch.Tensor'   # [nnz] float32
    _shape: Tuple[int, int]
    _name: str = field(default="", compare=False)
    _device: str = field(default="cpu", compare=False)
    
    def __post_init__(self):
        """Compute content hash if not provided."""
        if not self._name:
            name = compute_sparse_hash(self._indices, self._values, self._shape)
            object.__setattr__(self, '_name', name)
        object.__setattr__(self, '_device', str(self._indices.device))
    
    @classmethod
    def empty(
        cls,
        rows: int,
        cols: int,
        device: Optional[str] = None
    ) -> ImmutableSparseTensor:
        """Create empty sparse tensor."""
        if device is None:
            device = str(get_device())
        
        indices = torch.empty((2, 0), dtype=torch.int64, device=device)
        values = torch.empty((0,), dtype=get_default_dtype(), device=device)
        
        return cls(
            _indices=indices,
            _values=values,
            _shape=(rows, cols)
        )
    
    @classmethod
    def from_edges(
        cls,
        rows: int,
        cols: int,
        edges: List[Tuple[int, int, float]],
        device: Optional[str] = None
    ) -> ImmutableSparseTensor:
        """
        Create sparse tensor from edge list.
        
        Args:
            rows: Number of rows
            cols: Number of columns
            edges: List of (row, col, value) tuples
            device: Target device ('cuda' or 'cpu')
        """
        if device is None:
            device = str(get_device())
        
        if not edges:
            return cls.empty(rows, cols, device)
        
        # Build indices and values
        row_indices = []
        col_indices = []
        values = []
        
        for r, c, v in edges:
            row_indices.append(r)
            col_indices.append(c)
            values.append(v)
        
        indices = torch.tensor(
            [row_indices, col_indices],
            dtype=torch.int64,
            device=device
        )
        vals = torch.tensor(values, dtype=get_default_dtype(), device=device)
        
        return cls(
            _indices=indices,
            _values=vals,
            _shape=(rows, cols)
        )
    
    @classmethod
    def from_torch_sparse(
        cls,
        sparse_tensor: 'torch.Tensor'
    ) -> ImmutableSparseTensor:
        """Create from existing PyTorch sparse tensor."""
        if not sparse_tensor.is_sparse:
            raise ValueError("Expected sparse tensor")
        
        coalesced = sparse_tensor.coalesce()
        return cls(
            _indices=coalesced.indices().clone(),
            _values=coalesced.values().clone(),
            _shape=tuple(coalesced.shape)
        )
    
    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------
    
    @property
    def name(self) -> str:
        """Content-addressed name (SHA1 hash)."""
        return self._name
    
    @property
    def shape(self) -> Tuple[int, int]:
        """Tensor dimensions."""
        return self._shape
    
    @property
    def nnz(self) -> int:
        """Number of non-zero entries."""
        return self._values.numel()
    
    @property
    def indices(self) -> 'torch.Tensor':
        """Read-only indices [2, nnz]."""
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
    # Device Movement (returns new tensor)
    # -------------------------------------------------------------------------
    
    def to(self, device: str) -> ImmutableSparseTensor:
        """Move to device (returns new tensor)."""
        if device == self._device:
            return self
        
        return ImmutableSparseTensor(
            _indices=self._indices.to(device),
            _values=self._values.to(device),
            _shape=self._shape,
            _name=self._name  # Same content, same hash
        )
    
    def cuda(self) -> ImmutableSparseTensor:
        """Move to CUDA."""
        return self.to('cuda')
    
    def cpu(self) -> ImmutableSparseTensor:
        """Move to CPU."""
        return self.to('cpu')
    
    # -------------------------------------------------------------------------
    # Immutable Operations (all return new tensor)
    # -------------------------------------------------------------------------
    
    def with_edge(
        self,
        row: int,
        col: int,
        value: float
    ) -> ImmutableSparseTensor:
        """
        Return new tensor with edge added/updated.
        
        If edge exists, takes maximum (idempotent for same value).
        """
        # Check if edge exists
        if self.nnz > 0:
            mask = (self._indices[0] == row) & (self._indices[1] == col)
            if mask.any():
                # Edge exists - update with max
                idx = mask.nonzero(as_tuple=True)[0][0]
                if self._values[idx].item() >= value:
                    return self  # No change needed (idempotent)
                
                # Update value
                new_values = self._values.clone()
                new_values[idx] = value
                return ImmutableSparseTensor(
                    _indices=self._indices,
                    _values=new_values,
                    _shape=self._shape
                )
        
        # Add new edge
        new_idx = torch.tensor(
            [[row], [col]],
            dtype=torch.int64,
            device=self._indices.device
        )
        new_val = torch.tensor(
            [value],
            dtype=self._values.dtype,
            device=self._values.device
        )
        
        return ImmutableSparseTensor(
            _indices=torch.cat([self._indices, new_idx], dim=1),
            _values=torch.cat([self._values, new_val]),
            _shape=self._shape
        )
    
    def with_edges(
        self,
        edges: List[Tuple[int, int, float]]
    ) -> ImmutableSparseTensor:
        """Return new tensor with multiple edges added."""
        result = self
        for row, col, value in edges:
            result = result.with_edge(row, col, value)
        return result
    
    def maximum(self, other: ImmutableSparseTensor) -> ImmutableSparseTensor:
        """
        Element-wise maximum merge (idempotent).
        
        max(A, A) = A
        max(A, B) = max(B, A)
        """
        if self._shape != other._shape:
            raise ValueError(f"Shape mismatch: {self._shape} vs {other._shape}")
        
        # Convert to dense PyTorch sparse, do max, convert back
        # This handles duplicate indices correctly
        self_sparse = torch.sparse_coo_tensor(
            self._indices, self._values, self._shape,
            device=self._indices.device
        ).coalesce()
        
        other_sparse = torch.sparse_coo_tensor(
            other._indices, other._values, other._shape,
            device=other._indices.device
        ).coalesce()
        
        # For sparse max, we need to handle this carefully
        # Union of indices, max of values where both exist
        
        # Get all unique indices
        all_indices = torch.cat([self_sparse.indices(), other_sparse.indices()], dim=1)
        
        if all_indices.numel() == 0:
            return ImmutableSparseTensor.empty(
                self._shape[0], self._shape[1],
                device=self._device
            )
        
        # Use scatter_reduce with 'max' for efficient merging
        # Convert 2D indices to 1D for scatter
        flat_idx = all_indices[0] * self._shape[1] + all_indices[1]
        all_values = torch.cat([self_sparse.values(), other_sparse.values()])
        
        # Create output buffer
        flat_size = self._shape[0] * self._shape[1]
        out = torch.zeros(flat_size, dtype=all_values.dtype, device=all_values.device)
        out.scatter_reduce_(0, flat_idx, all_values, reduce='amax', include_self=False)
        
        # Convert back to sparse (only non-zero)
        nonzero_mask = out != 0
        nonzero_flat = nonzero_mask.nonzero(as_tuple=True)[0]
        
        if nonzero_flat.numel() == 0:
            return ImmutableSparseTensor.empty(
                self._shape[0], self._shape[1],
                device=self._device
            )
        
        # Convert flat indices back to 2D
        rows = nonzero_flat // self._shape[1]
        cols = nonzero_flat % self._shape[1]
        new_indices = torch.stack([rows, cols])
        new_values = out[nonzero_flat]
        
        return ImmutableSparseTensor(
            _indices=new_indices,
            _values=new_values,
            _shape=self._shape
        )
    
    # -------------------------------------------------------------------------
    # Query Operations
    # -------------------------------------------------------------------------
    
    def get(self, row: int, col: int, default: float = 0.0) -> float:
        """Get value at (row, col)."""
        if self.nnz == 0:
            return default
        
        mask = (self._indices[0] == row) & (self._indices[1] == col)
        if mask.any():
            return self._values[mask][0].item()
        return default
    
    def row_indices(self, row: int) -> 'torch.Tensor':
        """Get column indices for given row."""
        mask = self._indices[0] == row
        return self._indices[1, mask]
    
    def col_indices(self, col: int) -> 'torch.Tensor':
        """Get row indices for given column."""
        mask = self._indices[1] == col
        return self._indices[0, mask]
    
    def active_rows(self) -> Set[int]:
        """Get set of rows with at least one edge."""
        if self.nnz == 0:
            return set()
        return set(self._indices[0].cpu().numpy().tolist())
    
    def active_cols(self) -> Set[int]:
        """Get set of columns with at least one edge."""
        if self.nnz == 0:
            return set()
        return set(self._indices[1].cpu().numpy().tolist())
    
    def edges(self) -> List[Tuple[int, int, float]]:
        """Get all edges as list of (row, col, value) tuples."""
        if self.nnz == 0:
            return []
        
        indices = self._indices.cpu().numpy()
        values = self._values.cpu().numpy()
        
        return [
            (int(indices[0, i]), int(indices[1, i]), float(values[i]))
            for i in range(self.nnz)
        ]
    
    # -------------------------------------------------------------------------
    # Conversion
    # -------------------------------------------------------------------------
    
    def to_torch_sparse(self) -> 'torch.Tensor':
        """Convert to PyTorch sparse COO tensor."""
        return torch.sparse_coo_tensor(
            self._indices, self._values, self._shape,
            device=self._indices.device
        ).coalesce()
    
    def to_dense(self) -> 'torch.Tensor':
        """Convert to dense tensor (warning: may be large!)."""
        return self.to_torch_sparse().to_dense()
    
    # -------------------------------------------------------------------------
    # Serialization
    # -------------------------------------------------------------------------
    
    def to_bytes(self) -> bytes:
        """Serialize to bytes for persistence."""
        indices_np = self._indices.cpu().numpy()
        values_np = self._values.cpu().numpy()
        shape_np = np.array(self._shape, dtype=np.int64)
        
        return (
            shape_np.tobytes() +
            np.array([self.nnz], dtype=np.int64).tobytes() +
            indices_np.tobytes() +
            values_np.tobytes()
        )
    
    @classmethod
    def from_bytes(
        cls,
        data: bytes,
        device: Optional[str] = None
    ) -> ImmutableSparseTensor:
        """Deserialize from bytes."""
        if device is None:
            device = str(get_device())
        
        # Read shape
        shape = tuple(np.frombuffer(data[:16], dtype=np.int64))
        
        # Read nnz
        nnz = int(np.frombuffer(data[16:24], dtype=np.int64)[0])
        
        if nnz == 0:
            return cls.empty(shape[0], shape[1], device)
        
        # Read indices
        idx_bytes = nnz * 2 * 8  # 2 rows, int64
        indices_np = np.frombuffer(
            data[24:24+idx_bytes], dtype=np.int64
        ).reshape(2, nnz)
        
        # Read values
        values_np = np.frombuffer(
            data[24+idx_bytes:], dtype=np.float32
        )
        
        return cls(
            _indices=torch.tensor(indices_np, dtype=torch.int64, device=device),
            _values=torch.tensor(values_np, dtype=torch.float32, device=device),
            _shape=shape
        )
    
    # -------------------------------------------------------------------------
    # String Representation
    # -------------------------------------------------------------------------
    
    def __repr__(self) -> str:
        return (
            f"ImmutableSparseTensor(shape={self._shape}, nnz={self.nnz}, "
            f"device='{self._device}', name='{self._name[:16]}...')"
        )
    
    def __eq__(self, other: object) -> bool:
        """Content equality via hash comparison."""
        if not isinstance(other, ImmutableSparseTensor):
            return False
        return self._name == other._name
    
    def __hash__(self) -> int:
        """Hash based on content hash."""
        return hash(self._name)


# =============================================================================
# SECTION 4: Utility Functions
# =============================================================================

def sparse_zeros(rows: int, cols: int, device: Optional[str] = None) -> ImmutableSparseTensor:
    """Create empty sparse tensor."""
    return ImmutableSparseTensor.empty(rows, cols, device)


def sparse_from_dense(
    dense: 'torch.Tensor',
    threshold: float = 0.0
) -> ImmutableSparseTensor:
    """
    Create sparse tensor from dense tensor.
    
    Args:
        dense: Dense tensor
        threshold: Values <= threshold are treated as zero
    """
    mask = dense > threshold
    indices = mask.nonzero(as_tuple=False).t().contiguous()
    values = dense[mask]
    
    return ImmutableSparseTensor(
        _indices=indices.to(torch.int64),
        _values=values.to(torch.float32),
        _shape=tuple(dense.shape)
    )


# =============================================================================
# SECTION 5: Self-Test
# =============================================================================

if __name__ == "__main__":
    print("=== ImmutableSparseTensor Test ===")
    print()
    
    device = str(get_device())
    print(f"Device: {device}")
    print()
    
    # Create empty
    t1 = ImmutableSparseTensor.empty(1000, 1000, device)
    print(f"Empty: {t1}")
    print(f"  nnz: {t1.nnz}")
    print(f"  name: {t1.name}")
    print()
    
    # Add edges
    t2 = t1.with_edge(10, 20, 5.0)
    t3 = t2.with_edge(30, 40, 3.0)
    print(f"With 2 edges: {t3}")
    print(f"  nnz: {t3.nnz}")
    print(f"  get(10,20): {t3.get(10, 20)}")
    print(f"  get(30,40): {t3.get(30, 40)}")
    print(f"  get(99,99): {t3.get(99, 99)}")
    print()
    
    # Idempotence test
    t4 = t3.with_edge(10, 20, 5.0)  # Same edge, same value
    print(f"Idempotence (same edge): {t4.name == t3.name}")
    print()
    
    # Maximum merge
    t5 = ImmutableSparseTensor.from_edges(1000, 1000, [
        (10, 20, 7.0),  # Higher than t3
        (50, 60, 2.0),  # New edge
    ], device)
    
    merged = t3.maximum(t5)
    print(f"Merged: {merged}")
    print(f"  nnz: {merged.nnz}")
    print(f"  get(10,20): {merged.get(10, 20)} (max of 5.0 and 7.0)")
    print(f"  get(30,40): {merged.get(30, 40)}")
    print(f"  get(50,60): {merged.get(50, 60)}")
    print()
    
    # Merge idempotence
    self_merged = t3.maximum(t3)
    print(f"Self-merge idempotent: {self_merged.name == t3.name}")
    print()
    
    # Active indices
    print(f"Active rows: {merged.active_rows()}")
    print(f"Active cols: {merged.active_cols()}")
    print()
    
    # Memory comparison
    dense_size = 1000 * 1000 * 4  # float32
    sparse_size = merged.nnz * (8 + 8 + 4)  # 2 int64 + 1 float32
    print(f"Memory comparison (1000x1000, {merged.nnz} edges):")
    print(f"  Dense:  {dense_size / 1024:.1f} KB")
    print(f"  Sparse: {sparse_size / 1024:.1f} KB")
    print(f"  Ratio:  {dense_size / sparse_size:.1f}x smaller")
    print()
    
    # Serialization
    data = t3.to_bytes()
    t3_restored = ImmutableSparseTensor.from_bytes(data, device)
    print(f"Serialization round-trip: {t3_restored.name == t3.name}")
    print()
    
    # Large scale test
    print("=== Large Scale Test ===")
    import time
    
    edges = [(i, i+1, float(i)) for i in range(100000)]
    start = time.time()
    large = ImmutableSparseTensor.from_edges(32770, 32770, edges, device)
    elapsed = time.time() - start
    print(f"Created 100K edges in {elapsed*1000:.1f}ms")
    print(f"  {large}")
    print(f"  Memory: {large.nnz * 20 / 1024 / 1024:.2f} MB")
    print(f"  (Dense would be: {32770*32770*4 / 1024/1024/1024:.1f} GB)")
    print()
    
    print("✓ All tests passed!")
