"""
Sparse Matrix Representations for Algebra Operations

This module provides lightweight sparse matrix wrappers for algebraic operations.

Architecture:
- SparseMatrix: Pure-Python dict-of-dicts 2D matrix (for algebra ops)
- Sparse3DMatrix: Layered 2D matrices (for algebra ops)

These are SEPARATE from the core HRT structures:
- core/sparse_hrt_3d.py: Canonical HRT implementation (SparseAM3D, SparseHRT3D)
- core/sparse_tensor.py: PyTorch/CUDA tensor backend (ImmutableSparseTensor)

The algebra wrappers here are used for operations.py (project, merge, compose)
where we need simple dict-based access without the HRT overhead.

Conversion:
- Sparse3DMatrix.from_am(): Convert SparseAM3D → Sparse3DMatrix for algebra
- Sparse3DMatrix.to_edges(): Convert back to Edge3D list for HRT
"""

from __future__ import annotations
from typing import Dict, List, Set, Tuple, Iterator, TYPE_CHECKING
from dataclasses import dataclass

if TYPE_CHECKING:
    from ..sparse_hrt_3d import SparseAM3D, Sparse3DConfig, Edge3D


@dataclass
class SparseMatrix:
    """
    Simple sparse 2D matrix (single layer).
    
    Uses dict-of-dicts representation for O(1) access.
    Suitable for algebraic operations like project, transpose, merge.
    """
    data: Dict[int, Dict[int, float]]
    shape: Tuple[int, int]
    
    def get(self, row: int, col: int) -> float:
        """Get value at (row, col), default 0."""
        return self.data.get(row, {}).get(col, 0.0)
    
    def set(self, row: int, col: int, value: float):
        """Set value at (row, col)."""
        if row not in self.data:
            self.data[row] = {}
        self.data[row][col] = value
    
    def nonzero_entries(self) -> Iterator[Tuple[int, int, float]]:
        """Iterate over (row, col, value) for non-zero entries."""
        for row, cols in self.data.items():
            for col, val in cols.items():
                if val != 0:
                    yield row, col, val
    
    def row_indices(self) -> Set[int]:
        """Get all row indices with data."""
        return set(self.data.keys())
    
    def col_indices(self) -> Set[int]:
        """Get all column indices with data."""
        cols = set()
        for row_data in self.data.values():
            cols.update(row_data.keys())
        return cols
    
    def to_dict(self) -> Dict[int, Dict[int, float]]:
        """Return the underlying dict representation."""
        return self.data
    
    @property
    def nnz(self) -> int:
        """Number of non-zero entries."""
        return sum(len(cols) for cols in self.data.values())
    
    @classmethod
    def empty(cls, shape: Tuple[int, int]) -> 'SparseMatrix':
        """Create empty sparse matrix."""
        return cls(data={}, shape=shape)
    
    @classmethod
    def from_edges(cls, edges: List[Tuple[int, int, float]], 
                   shape: Tuple[int, int]) -> 'SparseMatrix':
        """Create from list of (row, col, value) tuples."""
        data: Dict[int, Dict[int, float]] = {}
        for row, col, val in edges:
            if row not in data:
                data[row] = {}
            data[row][col] = val
        return cls(data=data, shape=shape)
    
    @classmethod
    def from_dict(cls, data: Dict[int, Dict[int, float]], 
                  shape: Tuple[int, int]) -> 'SparseMatrix':
        """Create from dict-of-dicts."""
        return cls(data=data, shape=shape)


@dataclass
class Sparse3DMatrix:
    """
    Sparse 3D matrix (multiple layers).
    
    Each layer is a SparseMatrix. Used for algebra operations
    on n-gram layered structures.
    """
    layers: Tuple[SparseMatrix, ...]
    shape: Tuple[int, int, int]  # (n_layers, rows, cols)
    
    def get(self, layer: int, row: int, col: int) -> float:
        """Get value at (layer, row, col)."""
        if layer < 0 or layer >= len(self.layers):
            return 0.0
        return self.layers[layer].get(row, col)
    
    def layer(self, n: int) -> SparseMatrix:
        """Get layer n."""
        return self.layers[n]
    
    @property
    def n_layers(self) -> int:
        """Number of layers."""
        return len(self.layers)
    
    @property
    def nnz(self) -> int:
        """Total non-zero entries across all layers."""
        return sum(layer.nnz for layer in self.layers)
    
    @classmethod
    def empty(cls, n_layers: int, shape_2d: Tuple[int, int]) -> 'Sparse3DMatrix':
        """Create empty 3D sparse matrix."""
        layers = tuple(SparseMatrix.empty(shape_2d) for _ in range(n_layers))
        return cls(layers=layers, shape=(n_layers, shape_2d[0], shape_2d[1]))
    
    @classmethod
    def from_layer_edges(
        cls, 
        layer_edges: Dict[int, List[Tuple[int, int, float]]], 
        n_layers: int,
        shape_2d: Tuple[int, int]
    ) -> 'Sparse3DMatrix':
        """Create from dict of layer -> edges list."""
        layers = []
        for n in range(n_layers):
            edges = layer_edges.get(n, [])
            layers.append(SparseMatrix.from_edges(edges, shape_2d))
        return cls(layers=tuple(layers), shape=(n_layers, shape_2d[0], shape_2d[1]))
    
    @classmethod
    def from_am(cls, am: 'SparseAM3D', config: 'Sparse3DConfig') -> 'Sparse3DMatrix':
        """
        Convert SparseAM3D to Sparse3DMatrix for algebra operations.
        
        This bridges the HRT implementation with the algebra layer.
        """
        layer_edges: Dict[int, List[Tuple[int, int, float]]] = {}
        for n in range(config.max_n):
            edges = []
            for row, col, val in am.tensor.layer_edges(n):
                edges.append((row, col, val))
            layer_edges[n] = edges
        
        dim = config.dimension
        return cls.from_layer_edges(layer_edges, config.max_n, (dim, dim))
    
    def to_edges(self) -> List['Edge3D']:
        """
        Convert back to Edge3D list for HRT operations.
        
        Requires import at runtime to avoid circular dependency.
        """
        from ..sparse_hrt_3d import Edge3D
        edges = []
        for n, layer in enumerate(self.layers):
            for row, col, val in layer.nonzero_entries():
                edges.append(Edge3D(n=n, row=row, col=col, value=val))
        return edges


# Re-export canonical HRT classes for convenience
# These are the CANONICAL implementations - use these for HRT operations
def get_canonical_classes():
    """
    Get canonical sparse HRT classes.
    
    Use these for HRT construction and manipulation.
    Use SparseMatrix/Sparse3DMatrix for algebra operations.
    """
    from ..sparse_hrt_3d import (
        Sparse3DConfig,
        Edge3D,
        BasicHLLSet3D,
        ImmutableSparseTensor3D,
        SparseAM3D,
        SparseLattice3D,
        SparseHRT3D,
    )
    return {
        'Sparse3DConfig': Sparse3DConfig,
        'Edge3D': Edge3D,
        'BasicHLLSet3D': BasicHLLSet3D,
        'ImmutableSparseTensor3D': ImmutableSparseTensor3D,
        'SparseAM3D': SparseAM3D,
        'SparseLattice3D': SparseLattice3D,
        'SparseHRT3D': SparseHRT3D,
    }


__all__ = [
    # Algebra wrappers (for operations.py)
    'SparseMatrix',
    'Sparse3DMatrix',
    # Canonical class accessor
    'get_canonical_classes',
]
