"""
Lookup Table (LUT) - Token Recovery

Uses (reg, zeros) addressing by default, but supports pluggable
identifier schemes for different languages/sign systems.

- Hash scheme (default): content → hash → (reg, zeros) → index
- Vocabulary scheme: sign → vocabulary lookup → index (Chinese, etc.)
"""

from __future__ import annotations
from typing import Dict, Set, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict

from ..sparse_hrt_3d import Sparse3DConfig

from .identifier_schemes import IdentifierScheme
from .universal_id import content_to_index


# Special boundary tokens
START = ("<START>",)
END = ("<END>",)


@dataclass
class LookupTable:
    """
    Lookup Table for n-token recovery.
    
    Uses (reg, zeros) addressing by default, but supports pluggable
    identifier schemes for different languages/sign systems.
    
    - Hash scheme (default): content → hash → (reg, zeros) → index
    - Vocabulary scheme: sign → vocabulary lookup → index (Chinese, etc.)
    """
    config: Sparse3DConfig
    index_to_ntokens: Dict[int, Set[Tuple[int, Tuple[str, ...]]]] = field(default_factory=lambda: defaultdict(set))
    ntoken_to_index: Dict[Tuple[str, ...], int] = field(default_factory=dict)
    identifier_scheme: Optional[IdentifierScheme] = None  # None = use default hash scheme
    
    def add_ntoken(self, ntoken: Tuple[str, ...]) -> int:
        """Add n-token, return its index."""
        if ntoken in (START, END):
            layer = 0
        else:
            layer = len(ntoken) - 1
        
        content = " ".join(ntoken)
        idx = content_to_index(content, layer, self.config, self.identifier_scheme)
        
        self.index_to_ntokens[idx].add((layer, ntoken))
        self.ntoken_to_index[ntoken] = idx
        
        return idx
    
    def get_ntoken_index(self, ntoken: Tuple[str, ...]) -> Optional[int]:
        """Get index for n-token."""
        return self.ntoken_to_index.get(ntoken)
    
    def get_ntokens_at_index(self, idx: int) -> Set[Tuple[int, Tuple[str, ...]]]:
        """Get all (layer, ntoken) pairs at index."""
        return self.index_to_ntokens.get(idx, set())
    
    def get_1tokens_at_index(self, idx: int) -> Set[str]:
        """Get only 1-tokens (single words) at index."""
        result = set()
        for layer, nt in self.index_to_ntokens.get(idx, set()):
            if layer == 0 and nt not in (START, END):
                result.add(nt[0])
        return result


__all__ = [
    'START',
    'END',
    'LookupTable',
]
