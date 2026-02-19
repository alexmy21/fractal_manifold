"""
HLLSet - Immutable C/Cython Backend with Batch Processing

Design principles:
- HLLSet instances are fully immutable
- All operations return new instances
- Batch processing is the primary mode for token ingestion
- Multi-batch processing can be parallelized (thread-safe C backend)
- No in-place modifications

Batch Processing Pattern:
    # Single batch
    hll = HLLSet.from_batch(['token1', 'token2', ...])
    
    # Multi-batch with parallel processing
    batches = [batch1, batch2, batch3]
    hll_combined = HLLSet.from_batches(batches, parallel=True)
    
    # Accumulating pattern
    hll1 = HLLSet.from_batch(batch1)
    hll2 = HLLSet.from_batch(batch2)
    hll_combined = hll1.union(hll2)  # Immutable merge
"""

from __future__ import annotations
from typing import Set, Tuple, Union, List, Optional, Iterable
import hashlib
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import os

from .constants import P_BITS, SHARED_SEED

# Import C backend
try:
    from .hll_core import HLLCore
    C_BACKEND_AVAILABLE = True
except ImportError:
    C_BACKEND_AVAILABLE = False
    raise ImportError(
        "C backend (hll_core) not available. "
        "Please build the Cython extension with: python setup.py build_ext --inplace"
    )


def compute_sha1(data: Union[str, bytes, np.ndarray]) -> str:
    """Compute SHA1 hash of data."""
    if isinstance(data, str):
        data = data.encode('utf-8')
    elif isinstance(data, np.ndarray):
        data = data.tobytes()
    return hashlib.sha1(data).hexdigest()


class HLLSet:
    """
    HLLSet with C/Cython backend.
    
    Treat instances as immutable. Methods return new instances.
    """
    
    def __init__(self, p_bits: int = P_BITS, _core: Optional[HLLCore] = None):
        """
        Create HLLSet.
        
        Args:
            p_bits: Precision bits
            _core: Existing C HLLCore (internal use)
        """
        self.p_bits = p_bits
        self._core = _core if _core is not None else HLLCore(self.p_bits)
        self._name: Optional[str] = None
        
        # Compute name from content
        self._compute_name()
    
    def _compute_name(self):
        """Compute content-addressed name from registers."""
        registers = self.dump_numpy()
        self._name = compute_sha1(registers)
    
    # -------------------------------------------------------------------------
    # Class Methods - Primary API (Immutable Batch Processing)
    # -------------------------------------------------------------------------
    
    @classmethod
    def from_batch(cls, tokens: Union[List[str], Set[str], Iterable[str]], 
                   p_bits: int = P_BITS, seed: int = SHARED_SEED) -> HLLSet:
        """
        Create HLLSet from a batch of tokens (PRIMARY FACTORY METHOD).
        
        This is the recommended way to create HLLSets. All tokens in the batch
        are processed together into a new immutable HLLSet instance.
        
        Args:
            tokens: Batch of tokens (list, set, or iterable)
            p_bits: Precision bits for HLL
            seed: Hash seed for consistency
            
        Returns:
            New immutable HLLSet containing all tokens
            
        Example:
            >>> hll = HLLSet.from_batch(['token1', 'token2', 'token3'])
            >>> print(hll.cardinality())
        """
        if not isinstance(tokens, list):
            tokens = list(tokens)
        
        hll = cls(p_bits=p_bits)
        
        if tokens:
            hll._core.add_batch(tokens, seed)
            hll._compute_name()
        
        return hll
    
    @staticmethod
    def compute_reg_zeros_batch(tokens: Union[List[str], Set[str], Iterable[str]],
                                p_bits: int = P_BITS, seed: int = SHARED_SEED) -> List[Tuple[int, int]]:
        """
        Compute (reg, zeros) pairs for tokens WITHOUT creating HLLSet.
        
        This is a utility method for adjacency matrix construction to avoid
        duplicate hash calculations. When building AM, we need both:
        1. HLLSet for cardinality estimation (set operations)
        2. (reg, zeros) pairs for compact identifiers in AM
        
        Instead of:
            hll = HLLSet.from_batch(tokens)  # Calculates hashes
            pairs = [compute_reg_zeros(t) for t in tokens]  # RECALCULATES hashes!
        
        Use:
            pairs = HLLSet.compute_reg_zeros_batch(tokens)  # Calculate once
            hll = HLLSet.from_batch(tokens)  # Reuse cached calculation
        
        Args:
            tokens: Batch of tokens (list, set, or iterable)
            p_bits: Precision bits (must match HLLSet creation)
            seed: Hash seed (must match HLLSet creation)
        
        Returns:
            List of (reg, zeros) tuples, one per token
        
        Example:
            >>> tokens = ['hello', 'world']
            >>> pairs = HLLSet.compute_reg_zeros_batch(tokens)
            >>> print(pairs)  # [(512, 3), (789, 1)]
        """
        if not isinstance(tokens, list):
            tokens = list(tokens)
        
        if not tokens:
            return []
        
        # Use C backend to compute efficiently
        core = HLLCore(p_bits)
        return core.compute_reg_zeros_batch(tokens, seed)
    
    @classmethod
    def from_batches(cls, batches: List[Union[List[str], Set[str]]], 
                     p_bits: int = P_BITS, seed: int = SHARED_SEED,
                     parallel: bool = False, max_workers: Optional[int] = None) -> HLLSet:
        """
        Create HLLSet from multiple batches with optional parallel processing.
        
        Each batch is processed independently (can be parallelized), then all
        results are merged via union operation. This is efficient for large
        datasets that can be split into chunks.
        
        The C backend is thread-safe and supports true parallel processing!
        
        Args:
            batches: List of token batches
            p_bits: Precision bits for HLL
            seed: Hash seed (must be same for all batches)
            parallel: If True, process batches in parallel
            max_workers: Number of parallel workers (None = CPU count)
            
        Returns:
            New immutable HLLSet containing union of all batches
            
        Example:
            >>> batches = [['a', 'b'], ['c', 'd'], ['e', 'f']]
            >>> hll = HLLSet.from_batches(batches, parallel=True)
        """
        if not batches:
            return cls(p_bits=p_bits)
        
        if parallel:
            # TRUE parallel processing with C backend!
            max_workers = max_workers or os.cpu_count()
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                hlls = list(executor.map(
                    lambda b: cls.from_batch(b, p_bits=p_bits, seed=seed),
                    batches
                ))
        else:
            # Sequential processing
            hlls = [cls.from_batch(b, p_bits=p_bits, seed=seed) for b in batches]
        
        # Merge all HLLSets via union
        return cls.merge(hlls)
    
    @classmethod
    def merge(cls, hlls: List[HLLSet]) -> HLLSet:
        """
        Merge multiple HLLSets into one via union operation.
        
        This is the recommended way to combine multiple HLLSets. All input
        HLLSets must have the same p_bits.
        
        Args:
            hlls: List of HLLSet instances to merge
            
        Returns:
            New HLLSet containing union of all inputs
            
        Example:
            >>> hll1 = HLLSet.from_batch(['a', 'b'])
            >>> hll2 = HLLSet.from_batch(['c', 'd'])
            >>> merged = HLLSet.merge([hll1, hll2])
        """
        if not hlls:
            return cls()
        
        if len(hlls) == 1:
            return hlls[0]
        
        # Start with first HLL and union with rest
        result = hlls[0]
        for hll in hlls[1:]:
            result = result.union(hll)
        
        return result
    
    @classmethod
    def absorb(cls, tokens: Set[str], p_bits: int = P_BITS, seed: int = SHARED_SEED) -> HLLSet:
        """
        Create HLLSet from tokens (legacy method, use from_batch instead).
        
        Kept for backward compatibility. Prefer from_batch() for new code.
        """
        return cls.from_batch(tokens, p_bits=p_bits, seed=seed)
    
    @classmethod
    def add(cls, base: HLLSet, tokens: Union[str, List[str]], seed: int = SHARED_SEED) -> HLLSet:
        """
        Add tokens to an HLLSet, return new HLLSet (legacy method).
        
        Note: For better performance with large datasets, use from_batch()
        or from_batches() instead.
        
        Usage:
            h1 = HLLSet.from_batch(['a', 'b'])
            h2 = HLLSet.add(h1, 'c')  # h2 contains a,b,c; h1 unchanged
            h3 = HLLSet.add(h1, ['d', 'e'])  # batch add
        
        For accumulating batches, prefer:
            h1 = HLLSet.from_batch(batch1)
            h2 = HLLSet.from_batch(batch2)
            h_combined = h1.union(h2)
        """
        if isinstance(tokens, str):
            tokens = [tokens]
        
        if not tokens:
            return base  # No change needed
        
        # Create new HLLSet from tokens, then union with base
        tokens_hll = cls.from_batch(tokens, p_bits=base.p_bits, seed=seed)
        return base.union(tokens_hll)
    
    @classmethod
    def append(cls, base: HLLSet, tokens: Union[str, List[str]], seed: int = SHARED_SEED) -> HLLSet:
        """
        Append tokens to an HLLSet (alias for add).
        
        Same as add() - provided for API consistency.
        """
        return cls.add(base, tokens, seed)
    
    # -------------------------------------------------------------------------
    # Instance Methods - Return new instances
    # -------------------------------------------------------------------------
    
    def union(self, other: HLLSet) -> HLLSet:
        """Union with another HLLSet (returns new instance)."""
        if self.p_bits != other.p_bits:
            raise ValueError("Cannot union HLLs with different p_bits")
        
        result_core = self._core.union(other._core)
        return HLLSet(p_bits=self.p_bits, _core=result_core)
    
    def intersect(self, other: HLLSet) -> HLLSet:
        """
        Intersection with another HLLSet (returns new composable instance).
        
        Uses bitwise AND on register bitmaps: result[i] = A[i] & B[i].
        Each uint32 register is a bitmap of observed trailing-zero counts;
        AND keeps only observations present in BOTH sets.
        
        This is a TRUE set intersection sketch, fully composable:
          - A ∩ A = A  (idempotent)
          - A ∩ B = B ∩ A  (commutative)
          - (A ∩ B) ∩ C = A ∩ (B ∩ C)  (associative)
          - A ∩ (B ∪ C) = (A ∩ B) ∪ (A ∩ C)  (distributive)
        
        Matches Julia HllSets.jl: z.counts[i] = x.counts[i] & y.counts[i]
        """
        if self.p_bits != other.p_bits:
            raise ValueError("Cannot intersect HLLs with different p_bits")
        
        result_core = self._core.intersect(other._core)
        return HLLSet(p_bits=self.p_bits, _core=result_core)
    
    def diff(self, other: HLLSet) -> HLLSet:
        """
        Difference A - B (returns new composable instance).
        
        Uses bitwise AND-NOT on register bitmaps: result[i] = A[i] & ~B[i].
        Keeps only the observations in A that are NOT also in B.
        
        Matches Julia HllSets.jl set_comp:
            z.counts[i] = x.counts[i] & ~y.counts[i]
        
        Fully composable — result is a valid HLLSet.
        """
        if self.p_bits != other.p_bits:
            raise ValueError("Cannot diff HLLs with different p_bits")
        
        result_core = self._core.difference(other._core)
        return HLLSet(p_bits=self.p_bits, _core=result_core)
    
    def symmetric_difference(self, other: HLLSet) -> HLLSet:
        """
        Symmetric difference A △ B (returns new composable instance).
        
        Uses bitwise XOR on register bitmaps: result[i] = A[i] ^ B[i].
        Keeps only observations in exactly one of the two sets.
        
        Matches Julia HllSets.jl set_xor:
            z.counts[i] = xor(x.counts[i], y.counts[i])
        
        Fully composable — result is a valid HLLSet.
        """
        if self.p_bits != other.p_bits:
            raise ValueError("Cannot XOR HLLs with different p_bits")
        
        result_core = self._core.symmetric_difference(other._core)
        return HLLSet(p_bits=self.p_bits, _core=result_core)
    
    def xor(self, other: HLLSet) -> HLLSet:
        """Alias for symmetric_difference()."""
        return self.symmetric_difference(other)
    
    # -------------------------------------------------------------------------
    # Queries
    # -------------------------------------------------------------------------
    
    def cardinality(self) -> float:
        """Estimated cardinality."""
        return self._core.cardinality()
    
    def similarity(self, other: HLLSet) -> float:
        """Compute Jaccard similarity with another HLLSet (0.0 to 1.0)."""
        return self._core.jaccard_similarity(other._core)
    
    def cosine(self, other: HLLSet) -> float:
        """Cosine similarity."""
        return self._core.cosine_similarity(other._core)
    
    def dump_numpy(self) -> np.ndarray:
        """Get register vector as numpy array."""
        return self._core.get_registers()
    
    def dump_roaring(self) -> bytes:
        """
        Get registers as compressed Roaring bitmap.
        
        Returns serialized Roaring bitmap with 10-50x compression ratio
        for typical sparse HLLSets.
        
        Returns:
            bytes: Serialized Roaring bitmap
        """
        return self._core.get_registers_roaring()
    
    @classmethod
    def from_roaring(cls, compressed_data: bytes, p_bits: int = P_BITS) -> HLLSet:
        """
        Create HLLSet from compressed Roaring bitmap.
        
        Args:
            compressed_data: Serialized Roaring bitmap
            p_bits: Precision bits (must match original HLLSet)
            
        Returns:
            New HLLSet instance
        """
        hll = cls(p_bits=p_bits)
        hll._core.set_registers_roaring(compressed_data)
        hll._compute_name()
        return hll
    
    def get_compression_stats(self) -> dict:
        """
        Get compression statistics for this HLLSet.
        
        Returns:
            dict with keys:
                - original_size: Uncompressed size in bytes
                - compressed_size: Roaring bitmap size in bytes
                - compression_ratio: original / compressed
                - non_zero_registers: Count of non-zero registers
        """
        return self._core.get_compression_stats()
    
    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------
    
    @property
    def name(self) -> str:
        """Content-addressed name."""
        return self._name if self._name is not None else ""
    
    @property
    def short_name(self) -> str:
        """Short name for display."""
        return self._name[:8] if self._name else "unknown"
    
    @property
    def backend(self) -> str:
        """Return which backend is being used."""
        return "C/Cython"
    
    # -------------------------------------------------------------------------
    # Python Protocols
    # -------------------------------------------------------------------------
    
    def __hash__(self) -> int:
        return hash(self.name)
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, HLLSet):
            return False
        return self.name == other.name
    
    def __repr__(self) -> str:
        return f"HLLSet({self.short_name}..., |A|≈{self.cardinality():.1f}, backend={self.backend})"


# Export
__all__ = ['HLLSet', 'compute_sha1', 'C_BACKEND_AVAILABLE']
