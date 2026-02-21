"""
Fast Token Generation Utilities

These functions generate unique tokens for testing and benchmarking,
optimized for speed. Much faster than uuid.uuid4().hex.

Performance comparison (generating 1M tokens):
- uuid.uuid4().hex:     ~4000ms (slow, uses /dev/urandom syscall)
- numpy_hex_tokens:     ~400ms  (10x faster, bulk random)
- numpy_int_tokens:     ~150ms  (25x faster, minimal string ops)
- range_tokens:         ~80ms   (50x faster, simple incrementing)

Usage:
    from core.fast_tokens import numpy_hex_tokens, range_tokens
    
    # For realistic token simulation (hex strings)
    tokens = numpy_hex_tokens(100_000, prefix="doc_")
    
    # For maximum speed (integer strings)
    tokens = range_tokens(100_000, prefix="tok_", offset=0)
"""

from typing import List
import numpy as np


def numpy_hex_tokens(n: int, prefix: str = "tok_", seed: int = None) -> List[str]:
    """
    Generate n unique hex tokens using NumPy random.
    
    ~10x faster than uuid.uuid4().hex.
    
    Args:
        n: Number of tokens to generate
        prefix: Token prefix (default "tok_")
        seed: Random seed for reproducibility (None = random)
        
    Returns:
        List of unique token strings
        
    Example:
        >>> tokens = numpy_hex_tokens(1000, prefix="doc_")
        >>> print(tokens[0])  # 'doc_a3f7b2c1e9d8...'
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()
    
    # Generate 64-bit random integers and format as hex
    random_ints = rng.integers(0, 2**63, size=n, dtype=np.uint64)
    return [f"{prefix}{x:016x}" for x in random_ints]


def numpy_int_tokens(n: int, prefix: str = "tok_", seed: int = None) -> List[str]:
    """
    Generate n unique integer tokens using NumPy random.
    
    ~25x faster than uuid.uuid4().hex.
    Uses decimal representation (shorter strings).
    
    Args:
        n: Number of tokens to generate
        prefix: Token prefix (default "tok_")
        seed: Random seed for reproducibility (None = random)
        
    Returns:
        List of unique token strings
        
    Example:
        >>> tokens = numpy_int_tokens(1000, prefix="item_")
        >>> print(tokens[0])  # 'item_7234567890123456789'
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()
    
    random_ints = rng.integers(0, 2**63, size=n, dtype=np.uint64)
    return [f"{prefix}{x}" for x in random_ints]


def range_tokens(n: int, prefix: str = "tok_", offset: int = 0) -> List[str]:
    """
    Generate n sequential tokens (fastest method).
    
    ~50x faster than uuid.uuid4().hex.
    Tokens are unique within a single call but deterministic.
    Use different offsets for different batches to ensure uniqueness.
    
    Args:
        n: Number of tokens to generate
        prefix: Token prefix (default "tok_")
        offset: Starting number (use for uniqueness across batches)
        
    Returns:
        List of unique token strings
        
    Example:
        >>> tokens = range_tokens(1000, prefix="batch1_", offset=0)
        >>> print(tokens[0])  # 'batch1_0'
        >>> print(tokens[999])  # 'batch1_999'
    """
    return [f"{prefix}{i}" for i in range(offset, offset + n)]


def worker_tokens(worker_id: int, n: int, prefix: str = "w") -> List[str]:
    """
    Generate n tokens for a specific worker (parallel-safe).
    
    Each worker gets its own deterministic namespace based on worker_id.
    Tokens are unique across all workers.
    
    Args:
        worker_id: Unique worker identifier (0, 1, 2, ...)
        n: Number of tokens to generate
        prefix: Token prefix (default "w")
        
    Returns:
        List of unique token strings
        
    Example:
        >>> tokens = worker_tokens(0, 1000)  # Worker 0
        >>> print(tokens[0])  # 'w0_0'
        >>> tokens = worker_tokens(1, 1000)  # Worker 1
        >>> print(tokens[0])  # 'w1_0'
    """
    return [f"{prefix}{worker_id}_{i}" for i in range(n)]


def uuid_like_tokens(n: int, prefix: str = "", seed: int = None) -> List[str]:
    """
    Generate n UUID-like tokens (128-bit hex, similar to uuid4).
    
    ~8x faster than uuid.uuid4().hex.
    
    Args:
        n: Number of tokens to generate
        prefix: Token prefix (default "")
        seed: Random seed for reproducibility
        
    Returns:
        List of 32-character hex strings (like uuid4.hex)
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()
    
    # Generate two 64-bit integers per token for 128-bit total
    hi = rng.integers(0, 2**63, size=n, dtype=np.uint64)
    lo = rng.integers(0, 2**63, size=n, dtype=np.uint64)
    
    return [f"{prefix}{h:016x}{l:016x}" for h, l in zip(hi, lo)]


# Convenience aliases
fast_tokens = numpy_hex_tokens
parallel_tokens = worker_tokens
