"""
Generic Immutable Tensor for System Evolution

Provides:
1. ImmutableTensor - base class for all tensor data in the system
2. Content-addressed naming via SHA1 of tensor content
3. Clone-on-modify semantics enforced at the type level
4. PyTorch backend with numpy interoperability

Design Principles:
- All tensors are frozen after creation
- Any "modification" creates a new tensor with new hash
- Parent pointers maintain evolution lineage
"""

from __future__ import annotations
from typing import Optional, Tuple, Union, List, Dict, Any, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import hashlib
import json
import time

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    # Mock torch for type hints
    class MockTensor:
        pass
    torch = MockTensor()

import numpy as np


# =============================================================================
# SECTION 1: Hash Functions
# =============================================================================

def compute_element_hash(content: Any, bits: int = 64) -> int:
    """
    Compute hash for elements (tokens).
    Returns integer hash within specified bit width.
    """
    if isinstance(content, (int, float)):
        h = hash(content)
    elif isinstance(content, str):
        h = hash(content)
    elif isinstance(content, bytes):
        h = int.from_bytes(hashlib.sha256(content).digest()[:8], 'big')
    else:
        h = hash(str(content))
    
    # Mask to specified bit width
    mask = (1 << bits) - 1
    return h & mask


def compute_aggregate_hash(content: Union['torch.Tensor', np.ndarray, bytes, str]) -> str:
    """
    Compute SHA1 hash for aggregates (structures, tensors).
    """
    if TORCH_AVAILABLE and isinstance(content, torch.Tensor):
        # Convert to bytes via numpy (efficient, zero-copy when possible)
        if content.is_cuda:
            content = content.cpu()
        byte_data = content.numpy().tobytes()
    elif isinstance(content, np.ndarray):
        byte_data = content.tobytes()
    elif isinstance(content, str):
        byte_data = content.encode('utf-8')
    elif isinstance(content, bytes):
        byte_data = content
    else:
        byte_data = str(content).encode('utf-8')
    
    return hashlib.sha1(byte_data).hexdigest()


def compute_structural_hash(*components: str) -> str:
    """
    Compute hash from multiple string components.
    Used for naming structures composed of multiple hashed elements.
    """
    combined = ":".join(sorted(components))
    return hashlib.sha1(combined.encode('utf-8')).hexdigest()


# =============================================================================
# SECTION 2: Immutable Tensor Base
# =============================================================================

@dataclass(frozen=True)
class ImmutableTensor:
    """
    Immutable tensor with content-addressed naming.
    
    Core invariant: tensor data never changes after creation.
    All "modifications" return new ImmutableTensor instances.
    
    Attributes:
        data: PyTorch tensor (frozen)
        name: SHA1 hash of tensor content (computed automatically)
        parent: Optional hash of tensor this was derived from
        timestamp: Creation time
        metadata: Additional non-hash data (not part of identity)
    """
    data: 'torch.Tensor'
    name: str = field(init=False)
    parent: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    metadata: Tuple[Tuple[str, Any], ...] = field(default_factory=tuple, compare=False)
    
    def __post_init__(self):
        # Compute content hash
        if TORCH_AVAILABLE:
            tensor_hash = compute_aggregate_hash(self.data)
        else:
            tensor_hash = compute_aggregate_hash(str(self.data))
        object.__setattr__(self, 'name', tensor_hash)
    
    # -------------------------------------------------------------------------
    # Factory Methods
    # -------------------------------------------------------------------------
    
    @classmethod
    def zeros(cls, *shape: int, dtype = None, 
              parent: Optional[str] = None) -> ImmutableTensor:
        """Create zero-initialized tensor."""
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available")
        if dtype is None:
            dtype = torch.float32
        data = torch.zeros(*shape, dtype=dtype)
        return cls(data=data, parent=parent)
    
    @classmethod
    def from_numpy(cls, array: np.ndarray, parent: Optional[str] = None) -> ImmutableTensor:
        """Create from numpy array."""
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available")
        data = torch.from_numpy(array).clone()  # Clone to ensure no shared memory
        return cls(data=data, parent=parent)
    
    @classmethod
    def from_tensor(cls, tensor: 'torch.Tensor', parent: Optional[str] = None) -> ImmutableTensor:
        """Create from existing tensor (clones to ensure immutability)."""
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available")
        return cls(data=tensor.clone(), parent=parent)
    
    # -------------------------------------------------------------------------
    # Immutable Operations (all return new instances)
    # -------------------------------------------------------------------------
    
    def clone(self) -> ImmutableTensor:
        """Create identical copy with new timestamp."""
        return ImmutableTensor(
            data=self.data.clone(),
            parent=self.name,
            metadata=self.metadata
        )
    
    def with_value(self, indices: Tuple[int, ...], value: float) -> ImmutableTensor:
        """Return new tensor with value set at indices."""
        new_data = self.data.clone()
        new_data[indices] = value
        return ImmutableTensor(
            data=new_data,
            parent=self.name
        )
    
    def with_values(self, mask: 'torch.Tensor', values: 'torch.Tensor') -> ImmutableTensor:
        """Return new tensor with values set where mask is True."""
        new_data = self.data.clone()
        new_data[mask] = values
        return ImmutableTensor(
            data=new_data,
            parent=self.name
        )
    
    def maximum(self, other: ImmutableTensor) -> ImmutableTensor:
        """Element-wise maximum (for HRT merge)."""
        new_data = torch.maximum(self.data, other.data)
        return ImmutableTensor(
            data=new_data,
            parent=compute_structural_hash(self.name, other.name)
        )
    
    def add(self, other: ImmutableTensor) -> ImmutableTensor:
        """Element-wise addition."""
        new_data = self.data + other.data
        return ImmutableTensor(
            data=new_data,
            parent=compute_structural_hash(self.name, other.name)
        )
    
    def multiply(self, scalar: float) -> ImmutableTensor:
        """Scalar multiplication."""
        new_data = self.data * scalar
        return ImmutableTensor(
            data=new_data,
            parent=self.name
        )
    
    def select_rows(self, row_indices: List[int]) -> ImmutableTensor:
        """Select subset of rows."""
        new_data = self.data[row_indices]
        return ImmutableTensor(
            data=new_data,
            parent=self.name
        )
    
    def select_cols(self, col_indices: List[int]) -> ImmutableTensor:
        """Select subset of columns."""
        new_data = self.data[:, col_indices]
        return ImmutableTensor(
            data=new_data,
            parent=self.name
        )
    
    def project_rows(self, col_indices: List[int]) -> torch.Tensor:
        """
        Project columns to rows (sum over specified columns).
        Returns raw tensor (not wrapped) for efficiency.
        """
        if not col_indices:
            return torch.zeros(self.data.shape[0])
        return torch.sum(self.data[:, col_indices], dim=1)
    
    def project_cols(self, row_indices: List[int]) -> torch.Tensor:
        """
        Project rows to columns (sum over specified rows).
        Returns raw tensor (not wrapped) for efficiency.
        """
        if not row_indices:
            return torch.zeros(self.data.shape[1])
        return torch.sum(self.data[row_indices, :], dim=0)
    
    # -------------------------------------------------------------------------
    # Queries
    # -------------------------------------------------------------------------
    
    def to_numpy(self) -> np.ndarray:
        """Convert to numpy array."""
        return self.data.numpy()
    
    def nonzero_entries(self) -> List[Tuple[int, ...]]:
        """Get indices of all non-zero entries."""
        indices = torch.nonzero(self.data, as_tuple=False)
        return [tuple(idx.tolist()) for idx in indices]
    
    def nonzero_with_values(self) -> List[Tuple[int, ...], float]:
        """Get (indices, value) for all non-zero entries."""
        indices = torch.nonzero(self.data, as_tuple=False)
        result = []
        for idx in indices:
            value = float(self.data[tuple(idx.tolist())])
            result.append((tuple(idx.tolist()), value))
        return result
    
    @property
    def shape(self) -> Tuple[int, ...]:
        return tuple(self.data.shape)
    
    @property
    def dtype(self) -> torch.dtype:
        return self.data.dtype
    
    def __repr__(self) -> str:
        return f"ImmutableTensor({self.name[:16]}..., shape={self.shape})"
    
    def __hash__(self) -> int:
        return hash(self.name)
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, ImmutableTensor):
            return False
        return self.name == other.name


# =============================================================================
# SECTION 3: Tensor Evolution Triple (The Three States)
# =============================================================================

@dataclass(frozen=True)
class TensorEvolutionTriple:
    """
    The three-state model for evolution:
    - in_process: New data being ingested (will become current)
    - current: Active state in memory (will go to history)
    - history_hash: Reference to committed state in Git/persistent store
    
    Evolution step transitions:
    1. in_process + current → new_current (merge)
    2. current → history (commit)
    3. new in_process emerges (empty)
    """
    in_process: Optional[ImmutableTensor]
    current: ImmutableTensor
    history_hash: str  # Hash of last committed state
    step_number: int = 0
    
    def __repr__(self) -> str:
        return (f"TensorEvolutionTriple(step={self.step_number}, "
                f"in_process={'exists' if self.in_process else 'None'}, "
                f"current={self.current.name[:8]}..., "
                f"history={self.history_hash[:8]}...)")


class TensorEvolution:
    """
    Manages evolution of immutable tensors through the three-state cycle.
    """
    
    def __init__(self, initial_tensor: ImmutableTensor):
        self.triple = TensorEvolutionTriple(
            in_process=None,
            current=initial_tensor,
            history_hash=initial_tensor.name,  # Self-referential for genesis
            step_number=0
        )
        self._history: List[str] = [initial_tensor.name]
    
    def ingest(self, new_data: ImmutableTensor) -> TensorEvolutionTriple:
        """
        Set new data as in_process.
        Only valid if in_process is currently None.
        """
        if self.triple.in_process is not None:
            raise RuntimeError("Cannot ingest: in_process already exists. Must evolve first.")
        
        self.triple = TensorEvolutionTriple(
            in_process=new_data,
            current=self.triple.current,
            history_hash=self.triple.history_hash,
            step_number=self.triple.step_number
        )
        return self.triple
    
    def evolve(self, merge_fn: Callable[[ImmutableTensor, ImmutableTensor], ImmutableTensor],
               commit_fn: Callable[[ImmutableTensor], str]) -> TensorEvolutionTriple:
        """
        Execute evolution step:
        1. Merge in_process with current → new_current
        2. Commit current to history (get new hash)
        3. Reset in_process to None
        """
        if self.triple.in_process is None:
            raise RuntimeError("Cannot evolve: no in_process data. Must ingest first.")
        
        # Step 1: Merge
        new_current = merge_fn(self.triple.in_process, self.triple.current)
        
        # Step 2: Commit current to history
        new_history_hash = commit_fn(self.triple.current)
        
        # Step 3: Update triple
        self.triple = TensorEvolutionTriple(
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


# =============================================================================
# SECTION 4: Example Usage
# =============================================================================

def main():
    """Demonstrate immutable tensor evolution."""
    print("="*70)
    print("IMMUTABLE TENSOR - Generic Foundation for System Evolution")
    print("="*70)
    
    if not TORCH_AVAILABLE:
        print("PyTorch not available - demo skipped")
        return
    
    # Create initial tensor (genesis state)
    print("\n1. Genesis State")
    print("-" * 40)
    
    genesis = ImmutableTensor.zeros(5, 5)
    print(f"Genesis: {genesis}")
    print(f"Name: {genesis.name}")
    
    # Create evolution manager
    evolution = TensorEvolution(genesis)
    print(f"Initial triple: {evolution.triple}")
    
    # Ingest new data
    print("\n2. Ingest New Data")
    print("-" * 40)
    
    new_data = ImmutableTensor.zeros(5, 5)
    new_data = new_data.with_value((0, 0), 1.0)
    new_data = new_data.with_value((1, 1), 2.0)
    print(f"New data: {new_data}")
    
    evolution.ingest(new_data)
    print(f"After ingest: {evolution.triple}")
    
    # Evolve
    print("\n3. Evolution Step")
    print("-" * 40)
    
    def merge_fn(in_proc, curr):
        """Element-wise maximum merge."""
        return in_proc.maximum(curr)
    
    def commit_fn(tensor):
        """Mock commit - just returns hash."""
        print(f"  Committing tensor {tensor.name[:16]}... to history")
        return tensor.name
    
    evolution.evolve(merge_fn, commit_fn)
    print(f"After evolve: {evolution.triple}")
    print(f"Lineage: {[h[:8] for h in evolution.get_lineage()]}")
    
    # Another cycle
    print("\n4. Another Evolution Cycle")
    print("-" * 40)
    
    more_data = ImmutableTensor.zeros(5, 5)
    more_data = more_data.with_value((2, 2), 3.0)
    
    evolution.ingest(more_data)
    evolution.evolve(merge_fn, commit_fn)
    print(f"After second evolve: {evolution.triple}")
    print(f"Lineage: {[h[:8] for h in evolution.get_lineage()]}")
    
    # Demonstrate immutability
    print("\n5. Immutability Guarantee")
    print("-" * 40)
    
    original = evolution.triple.current
    modified = original.with_value((3, 3), 5.0)
    
    print(f"Original: {original}")
    print(f"Modified: {modified}")
    print(f"Same object? {original is modified}")
    print(f"Same hash? {original.name == modified.name}")
    print(f"Original unchanged? {original.name == evolution.triple.current.name}")
    
    print("\n" + "="*70)
    print("Immutable Tensor Ready for HRT Integration")
    print("="*70)
    
    return evolution


if __name__ == "__main__":
    main()
