"""
Universal Identifier - (reg, zeros) addressing

The (reg, zeros) Universal Identifier
=====================================

All structures (HLLSet, AM, W, Sheaf sections) use the same addressing:

    content → hash → (reg, zeros) → index

This invariant enables:
- Content-addressability: same content → same position everywhere
- Compatibility: sub-structures built anywhere use same addressing
- Idempotent merge: no index translation needed
- Sheaf gluing: sections at same (reg, zeros) across layers form global sections
"""

from __future__ import annotations
from typing import Optional, Tuple
from dataclasses import dataclass

from ..hllset import (
    HashConfig,
    DEFAULT_HASH_CONFIG,
)
from ..sparse_hrt_3d import Sparse3DConfig

from .identifier_schemes import IdentifierScheme


@dataclass(frozen=True)
class UniversalID:
    """
    Universal identifier used across all structures.
    
    Computed from content hash using HLLSet's centralized hash:
        hash → reg (which register) + zeros (leading zeros)
    
    This is the "glue" that connects HLLSet, AM, W, and Sheaf sections.
    """
    reg: int      # Register index (0 to 2^p_bits - 1)
    zeros: int    # Leading zeros count (0 to h_bits - p_bits)
    layer: int    # Layer/n-gram level (0, 1, 2, ...)
    
    @classmethod
    def from_hash(cls, h: int, layer: int, p_bits: int, h_bits: int) -> 'UniversalID':
        """Compute (reg, zeros) from hash value."""
        reg = h & ((1 << p_bits) - 1)
        remaining = h >> p_bits
        zeros = 0
        max_zeros = h_bits - p_bits
        while zeros < max_zeros and (remaining & 1) == 0:
            zeros += 1
            remaining >>= 1
        return cls(reg=reg, zeros=zeros, layer=layer)
    
    @classmethod
    def from_content(cls, content: str, layer: int, 
                     config: Optional[HashConfig] = None) -> 'UniversalID':
        """
        Compute (reg, zeros) from content string.
        
        Uses HLLSet's centralized hash configuration.
        """
        cfg = config or DEFAULT_HASH_CONFIG
        reg, zeros = cfg.hash_to_reg_zeros(content)
        return cls(reg=reg, zeros=zeros, layer=layer)
    
    def to_index(self, config: Sparse3DConfig) -> int:
        """Convert to linear index for matrix addressing."""
        return self.reg * (config.h_bits - config.p_bits + 1) + self.zeros
    
    def __repr__(self) -> str:
        return f"UID(r={self.reg}, z={self.zeros}, L{self.layer})"


def content_to_index(
    content: str, 
    layer: int, 
    config: Sparse3DConfig,
    identifier_scheme: Optional[IdentifierScheme] = None
) -> int:
    """
    Convert content to matrix index.
    
    Uses the provided identifier scheme, or default hash scheme.
    This is the main entry point for content → index conversion.
    """
    if identifier_scheme:
        return identifier_scheme.to_index(content, layer)
    
    # Default: use UniversalID (hash-based)
    uid = UniversalID.from_content(content, layer)
    return uid.to_index(config)


__all__ = [
    'UniversalID',
    'content_to_index',
]
