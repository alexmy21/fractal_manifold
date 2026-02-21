# core/ingest_utils.py
"""
Ingest Utils: Helper functions for computing hashes and (reg, zeros) pairs.

This module helps decouple ManifoldOS from HLLSet specifics by providing
the hash-to-register logic in a standalone way.
"""

from typing import Tuple, List, Union
import hashlib

from .constants import P_BITS

def murmur_hash_wrapper(data: bytes, seed: int = 0) -> int:
    """
    Wrapper for MurmurHash64A (Python implementation).
    NOTE: For production, we should use the C-extension version if available,
    but this provides a reference implementation.
    
    In the final system, ManifoldOS should likely call into HLLCore for this
    to ensure 100% consistency with HLLSet storage.
    """
    # Placeholder: In a real scenario, we expose murmur_hash64 from hll_core
    # For now, we rely on HLLSet.compute_hash_batch equivalent
    pass

def compute_reg_zeros(hash_val: int, p_bits: int) -> Tuple[int, int]:
    """
    Compute register index and trailing zeros from a 64-bit hash.
    
    Args:
        hash_val: 64-bit integer hash
        p_bits: Precision bits (defines register count 2^p)
    
    Returns:
        (register_index, trailing_zeros_count)
    """
    # Bottom P bits -> register index
    reg_idx = hash_val & ((1 << p_bits) - 1)
    
    # Remaining bits -> count trailing zeros
    remaining = hash_val >> p_bits
    
    if remaining == 0:
        zeros = 64 - p_bits
    else:
        zeros = 0
        while (remaining & 1) == 0:
            zeros += 1
            remaining >>= 1
            
    return (reg_idx, zeros)
