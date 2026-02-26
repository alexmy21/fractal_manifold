"""
Manifold Algebra - Unified Processing Model

The (reg, zeros) Universal Identifier
=====================================

All structures (HLLSet, AM, W, Sheaf sections) use the same addressing:

    content → hash → (reg, zeros) → index

This invariant enables:
- Content-addressability: same content → same position everywhere
- Compatibility: sub-structures built anywhere use same addressing
- Idempotent merge: no index translation needed
- Sheaf gluing: sections at same (reg, zeros) across layers form global sections

Unified Pipeline
================

Every interaction (ingestion OR query) follows the same pipeline:

    INPUT → HLLSet → New HRT → Extend with Context → Merge → New Current

Properties:
- Sub-structure isolation: work on separate instance
- Idempotent merge: same input → same result
- Eventual consistency: parallel changes converge
- CRDT-like: commutative, associative, idempotent

Manifold Algebra Operations
===========================

Structure-agnostic operations that preserve (reg, zeros) addressing:

Projection (π):
    π_n(M)        - Extract layer n
    π_R(M)        - Extract rows R
    π_C(M)        - Extract columns C

Transform:
    T(M)          - Transpose
    N(M)          - Normalize rows
    S_α(M)        - Scale by α

Composition:
    M₁ + M₂       - Merge (add)
    M₁ ∘ M₂       - Chain (multiply)

Path:
    reach(M, S, k) - k-hop reachability
    M*            - Transitive closure

Lift/Lower:
    ↑_n(M)        - Lift 2D to layer n
    ↓(M)          - Lower 3D to 2D
"""

from __future__ import annotations
from typing import (
    List, Dict, Set, Optional, Tuple, FrozenSet, 
    Callable, Iterator, Any, Union, NamedTuple, Iterable
)
from dataclasses import dataclass, field
from functools import reduce
from collections import defaultdict
import numpy as np

# Internal imports - structure layer
from .sparse_hrt_3d import (
    SparseHRT3D,
    Sparse3DConfig,
    SparseAM3D,
    SparseLattice3D,
    ImmutableSparseTensor3D,
    BasicHLLSet3D,
    Edge3D,
)

# Internal imports - foundation layer (via hllset for core types)
from .hllset import (
    HLLSet, 
    compute_sha1,
    HashConfig,
    HashType,
    DEFAULT_HASH_CONFIG,
    P_BITS,
    SHARED_SEED,
)

# Default hash bits (from HLLSet's centralized config)
DEFAULT_H_BITS = DEFAULT_HASH_CONFIG.h_bits


# ═══════════════════════════════════════════════════════════════════════════
# IDENTIFIER SCHEME: Pluggable Content → Index Mapping
# ═══════════════════════════════════════════════════════════════════════════

from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable


@runtime_checkable
class IdentifierScheme(Protocol):
    """
    Protocol for content → index mapping.
    
    Different languages/sign systems use different identification schemes:
    
    - HashScheme (default): content → hash → (reg, zeros) → index
      For inflected languages where vocabulary is open-ended.
      
    - VocabularyScheme: sign → vocabulary lookup → index
      For uninflected languages with fixed sign systems (Chinese, etc.)
      where each sign is unambiguous and unmodifiable.
    
    The AM and W structures don't care HOW indices are computed,
    only that they're consistent within a system.
    """
    
    def to_index(self, content: str, layer: int = 0) -> int:
        """Convert content to matrix index."""
        ...
    
    def index_range(self) -> int:
        """Maximum possible index (for matrix sizing)."""
        ...
    
    @property
    def scheme_type(self) -> str:
        """Identifier for serialization."""
        ...


@dataclass
class HashIdentifierScheme:
    """
    Hash-based identification: content → hash → (reg, zeros) → index
    
    Default scheme for inflected languages where vocabulary is open-ended.
    Uses HLLSet-compatible (reg, zeros) addressing.
    
    HASH CONFIGURATION: Uses HLLSet's centralized hash settings.
    The hash function is delegated to HLLSet, ensuring consistency
    across all modules.
    
    Properties:
    - Covers all bits in HLLSet representation
    - Simple and consistent
    - Handles typos, variations, novel words naturally
    - Same content ALWAYS → same index
    """
    p_bits: int = DEFAULT_HASH_CONFIG.p_bits
    h_bits: int = DEFAULT_HASH_CONFIG.h_bits
    _hash_seed: int = DEFAULT_HASH_CONFIG.seed
    
    def __post_init__(self):
        # Create a HashConfig for this scheme
        self._config = HashConfig(
            hash_type=DEFAULT_HASH_CONFIG.hash_type,
            p_bits=self.p_bits,
            seed=self._hash_seed,
            h_bits=self.h_bits
        )
    
    def to_index(self, content: str, layer: int = 0) -> int:
        """Compute index from content hash using HLLSet's hash."""
        reg, zeros = self._config.hash_to_reg_zeros(content)
        return reg * (self.h_bits - self.p_bits + 1) + zeros
    
    def index_range(self) -> int:
        """Max index = m * (h_bits - p_bits + 1)."""
        return (1 << self.p_bits) * (self.h_bits - self.p_bits + 1)
    
    @property
    def scheme_type(self) -> str:
        return "hash"
    
    def to_reg_zeros(self, content: str) -> Tuple[int, int]:
        """Get (reg, zeros) for HLLSet compatibility."""
        return self._config.hash_to_reg_zeros(content)
    
    @property
    def config(self) -> HashConfig:
        """Get the hash configuration."""
        return self._config


@dataclass
class VocabularyIdentifierScheme:
    """
    Vocabulary-based identification: sign → lookup → index
    
    For uninflected languages with fixed sign systems where each sign
    is unambiguous and unmodifiable (Chinese hieroglyphs, musical notes, etc.)
    
    Properties:
    - Direct mapping: no hash collisions
    - Limited vocabulary (~80K for Chinese)
    - Each sign is a complete semantic unit
    - Hieroglyphs combine to form sentences/compounds
    
    Usage:
        # Load Chinese vocabulary (character → index)
        vocab = load_chinese_vocab()  # {"你": 0, "好": 1, ...}
        scheme = VocabularyIdentifierScheme(vocab)
        
        # Index lookup
        idx = scheme.to_index("你")  # 0
    """
    vocabulary: Dict[str, int] = field(default_factory=dict)
    unknown_index: int = -1  # Index for unknown signs
    
    def __post_init__(self):
        if self.vocabulary:
            self._max_idx = max(self.vocabulary.values()) + 1
        else:
            self._max_idx = 1
    
    def to_index(self, content: str, layer: int = 0) -> int:
        """Lookup sign in vocabulary.
        
        For single characters: direct vocabulary lookup
        For n-grams (space-separated): combine constituent indices
        """
        # Single character: direct lookup
        if len(content) == 1:
            return self.vocabulary.get(content, self.unknown_index)
        
        # N-gram (space-separated tokens): combine constituent indices
        # Format: "学 习" for 2-gram, "学 习 是" for 3-gram
        parts = content.split()
        if len(parts) > 1:
            # All constituents must be known
            indices = []
            for part in parts:
                idx = self.vocabulary.get(part, self.unknown_index)
                if idx == self.unknown_index:
                    return self.unknown_index
                indices.append(idx)
            
            # Combine indices using a deterministic formula
            # Use Cantor pairing extended to n-tuples
            result = indices[0]
            for idx in indices[1:]:
                # Cantor pairing: (a + b)(a + b + 1)/2 + b
                result = ((result + idx) * (result + idx + 1)) // 2 + idx
            
            # Offset by vocabulary size * layer to avoid collisions with 1-grams
            return self._max_idx * (layer + 1) + result
        
        # Fallback: try direct lookup
        return self.vocabulary.get(content, self.unknown_index)
    
    def index_range(self) -> int:
        """Max index = vocabulary size."""
        return self._max_idx
    
    @property
    def scheme_type(self) -> str:
        return "vocabulary"
    
    def add_sign(self, sign: str, index: Optional[int] = None) -> int:
        """Add a sign to vocabulary, return its index."""
        if sign in self.vocabulary:
            return self.vocabulary[sign]
        idx = index if index is not None else self._max_idx
        self.vocabulary[sign] = idx
        self._max_idx = max(self._max_idx, idx + 1)
        return idx
    
    @classmethod
    def from_file(cls, path: str, encoding: str = 'utf-8') -> 'VocabularyIdentifierScheme':
        """Load vocabulary from file (one sign per line, or sign<tab>index)."""
        vocab = {}
        with open(path, 'r', encoding=encoding) as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                if '\t' in line:
                    sign, idx_str = line.split('\t', 1)
                    vocab[sign] = int(idx_str)
                else:
                    vocab[line] = i
        return cls(vocabulary=vocab)


@dataclass
class HashVocabularyScheme:
    """
    Precise hash-indexed vocabulary: sign → hash → index (full hash)
    
    For compact vocabularies (Chinese ~80K) where you want EXACT addressing:
    - Full 32-bit hash as index (no (reg, zeros) compression)
    - Each sign gets unique row/column in AM/W
    - Union across row/column = context for THAT SPECIFIC SIGN
    - No collision from compression (only from hash collision, ~0 for 80K)
    
    Compare to default HashIdentifierScheme:
    - HashIdentifierScheme: token → hash → (reg, zeros) → index  [~32K indices]
    - HashVocabularyScheme: sign → hash → index                  [~4B indices]
    
    The default scheme compresses via (reg, zeros) which is fine for:
    - Open vocabularies (unbounded tokens)
    - HLLSet operations (native addressing)
    - Probabilistic retrieval (some collision acceptable)
    
    Use HashVocabularyScheme when you need:
    - Exact addressing for compact, known vocabulary
    - Precise context unions (neighbors of 你 = exactly tokens seen with 你)
    - Reverse lookup (index → sign)
    - Vocabulary validation (reject unknown signs)
    
    Note: Still stores (reg, zeros) per sign for HLLSet compatibility.
    
    HASH CONFIGURATION: Uses HLLSet's centralized hash settings.
    The hash function is delegated to HLLSet, ensuring consistency
    across all modules.
    
    Usage:
        scheme = HashVocabularyScheme()
        scheme.add_sign("你")  # Full hash = exact index
        scheme.add_sign("好")
        
        idx = scheme.to_index("你")  # 32-bit hash as index
        reg, zeros = scheme.to_reg_zeros("你")  # For HLLSet ops
        sign = scheme.get_sign(idx)  # Reverse lookup
    """
    p_bits: int = DEFAULT_HASH_CONFIG.p_bits
    h_bits: int = DEFAULT_HASH_CONFIG.h_bits
    _hash_seed: int = DEFAULT_HASH_CONFIG.seed
    
    # sign → full hash index
    known_signs: Dict[str, int] = field(default_factory=dict)
    # index → sign (for reverse lookup)
    index_to_sign: Dict[int, str] = field(default_factory=dict)
    # sign → (reg, zeros) for HLLSet compatibility
    sign_to_reg_zeros: Dict[str, Tuple[int, int]] = field(default_factory=dict)
    
    # How to handle unknown signs
    allow_unknown: bool = False
    unknown_index: int = -1
    
    def __post_init__(self):
        # Create a HashConfig for this scheme
        self._config = HashConfig(
            hash_type=DEFAULT_HASH_CONFIG.hash_type,
            p_bits=self.p_bits,
            seed=self._hash_seed,
            h_bits=self.h_bits
        )
    
    def _compute_full_hash(self, content: str) -> int:
        """Compute full hash value (32-bit) using HLLSet's seeded hash."""
        return self._config.hash_with_seed(content)
    
    def to_index(self, content: str, layer: int = 0) -> int:
        """Get index for sign (full hash, exact addressing)."""
        if content in self.known_signs:
            return self.known_signs[content]
        elif self.allow_unknown:
            return self._compute_full_hash(content)
        else:
            return self.unknown_index
    
    def to_his_index(self, content: str) -> int:
        """
        Get HashIdentifierScheme-compatible index.
        
        Degrades HVS → HIS: hash → (reg, zeros) → index
        
        This enables mixing HVS and HIS data:
        - HVS uses full hash for precise AM/W addressing
        - HIS uses (reg, zeros) for HLLSet-native operations
        - Same sign → same (reg, zeros) in both schemes
        
        Returns:
            HIS-compatible index: reg * (h_bits - p_bits + 1) + zeros
        """
        reg, zeros = self.to_reg_zeros(content)
        return reg * (self.h_bits - self.p_bits + 1) + zeros
    
    @property
    def config(self) -> HashConfig:
        """Get the hash configuration."""
        return self._config
    
    def to_reg_zeros(self, content: str) -> Tuple[int, int]:
        """Get (reg, zeros) for HLLSet compatibility using centralized hash."""
        if content in self.sign_to_reg_zeros:
            return self.sign_to_reg_zeros[content]
        # Use centralized hash_to_reg_zeros from HashConfig
        return self._config.hash_to_reg_zeros(content)
    
    def add_sign(self, sign: str) -> int:
        """Add sign to vocabulary, return its full hash index."""
        h = self._compute_full_hash(sign)
        # Use centralized hash_to_reg_zeros from HashConfig
        reg_zeros = self._config.hash_to_reg_zeros(sign)
        
        self.known_signs[sign] = h
        self.index_to_sign[h] = sign
        self.sign_to_reg_zeros[sign] = reg_zeros
        return h
    
    def is_known(self, sign: str) -> bool:
        """Check if sign is in vocabulary."""
        return sign in self.known_signs
    
    def get_sign(self, index: int) -> Optional[str]:
        """Reverse lookup: index → sign."""
        return self.index_to_sign.get(index)
    
    def get_reg_zeros(self, sign: str) -> Optional[Tuple[int, int]]:
        """Get stored (reg, zeros) for a known sign."""
        return self.sign_to_reg_zeros.get(sign)
    
    def index_range(self) -> int:
        """Max index value: 2^32 (full hash range)."""
        return 2 ** 32
    
    def his_index_range(self) -> int:
        """Max index in HIS-compatible mode: m * (h_bits - p_bits + 1)."""
        return (1 << self.p_bits) * (self.h_bits - self.p_bits + 1)
    
    # ═══════════════════════════════════════════════════════════════════════
    # HVS ↔ HIS INTEROPERABILITY
    # ═══════════════════════════════════════════════════════════════════════
    
    def to_his_scheme(self) -> 'HashIdentifierScheme':
        """
        Create a HashIdentifierScheme with same parameters.
        
        Both schemes will produce identical (reg, zeros) for same content.
        """
        return HashIdentifierScheme(p_bits=self.p_bits, h_bits=self.h_bits)
    
    def hvs_to_his_index_map(self) -> Dict[int, int]:
        """
        Build mapping: HVS index → HIS index for all known signs.
        
        This is the projection map: precise → compressed.
        Multiple HVS indices may map to same HIS index (lossy).
        
        Returns:
            Dict mapping full hash indices to (reg,zeros) encoded indices
        """
        return {
            hvs_idx: self.to_his_index(sign)
            for sign, hvs_idx in self.known_signs.items()
        }
    
    def his_to_hvs_index_map(self) -> Dict[int, Set[int]]:
        """
        Build reverse mapping: HIS index → set of HVS indices.
        
        Since HIS is lossy, one HIS index may correspond to multiple HVS indices.
        
        Returns:
            Dict mapping (reg,zeros) encoded index to set of full hash indices
        """
        result: Dict[int, Set[int]] = defaultdict(set)
        for sign, hvs_idx in self.known_signs.items():
            his_idx = self.to_his_index(sign)
            result[his_idx].add(hvs_idx)
        return dict(result)
    
    def signs_at_his_index(self, his_index: int) -> List[str]:
        """
        Get all signs that map to a given HIS index.
        
        Since (reg, zeros) is lossy, multiple signs may share same HIS index.
        This is the disambiguation set for that HIS index within this vocabulary.
        
        Args:
            his_index: HIS-compatible index (reg * range + zeros)
            
        Returns:
            List of signs that hash to this (reg, zeros)
        """
        return [
            sign for sign in self.known_signs
            if self.to_his_index(sign) == his_index
        ]
    
    def project_am_to_his(
        self,
        am_edges: List[Tuple[int, int, int, float]]
    ) -> List[Tuple[int, int, int, float]]:
        """
        Project AM edges from HVS indices to HIS indices.
        
        HVS AM has precise row/column per sign.
        HIS AM has compressed (reg, zeros) addressing.
        
        This merges edges whose endpoints map to same (reg, zeros).
        Values are summed (or max'd depending on semantics).
        
        Args:
            am_edges: List of (layer, row, col, value) with HVS indices
            
        Returns:
            List of (layer, row, col, value) with HIS indices
        """
        hvs_to_his = self.hvs_to_his_index_map()
        
        # Aggregate edges by HIS coordinates
        his_edges: Dict[Tuple[int, int, int], float] = defaultdict(float)
        
        for layer, row, col, val in am_edges:
            his_row = hvs_to_his.get(row, row)  # Pass through if not in vocab
            his_col = hvs_to_his.get(col, col)
            his_edges[(layer, his_row, his_col)] += val
        
        return [(l, r, c, v) for (l, r, c), v in his_edges.items()]
    
    @property
    def scheme_type(self) -> str:
        return "hash_vocabulary"
    
    @classmethod
    def from_file(
        cls,
        path: str,
        encoding: str = 'utf-8',
        p_bits: int = P_BITS,
        h_bits: int = DEFAULT_H_BITS
    ) -> 'HashVocabularyScheme':
        """Load vocabulary from file (one sign per line)."""
        scheme = cls(p_bits=p_bits, h_bits=h_bits)
        with open(path, 'r', encoding=encoding) as f:
            for line in f:
                sign = line.strip()
                if sign:
                    scheme.add_sign(sign)
        return scheme
    
    @classmethod
    def from_signs(
        cls,
        signs: Iterable[str],
        p_bits: int = P_BITS,
        h_bits: int = DEFAULT_H_BITS
    ) -> 'HashVocabularyScheme':
        """Build vocabulary from iterable of signs."""
        scheme = cls(p_bits=p_bits, h_bits=h_bits)
        for sign in signs:
            scheme.add_sign(sign)
        return scheme
    
    def collision_report(self) -> Dict[str, Any]:
        """Analyze index collisions in vocabulary.
        
        For 80K vocabulary in 2^32 space: ~0 collisions expected.
        """
        index_counts: Dict[int, List[str]] = defaultdict(list)
        for sign, idx in self.known_signs.items():
            index_counts[idx].append(sign)
        
        collisions = {idx: signs for idx, signs in index_counts.items() if len(signs) > 1}
        
        return {
            'vocabulary_size': len(self.known_signs),
            'unique_indices': len(index_counts),
            'collisions': len(collisions),
            'collision_pairs': collisions,
            'index_range': self.index_range(),
            'density': len(self.known_signs) / self.index_range(),
        }
    
    # ═══════════════════════════════════════════════════════════════════════
    # VOCABULARY FINGERPRINT: HLLSet representation of vocabulary
    # ═══════════════════════════════════════════════════════════════════════
    
    def to_registers(self) -> np.ndarray:
        """
        Get vocabulary fingerprint as raw HLL registers.
        
        Returns numpy array of register values, NOT an HLLSet object.
        For actual HLLSet object, use to_hllset().
        
        Returns:
            np.ndarray of shape (2^p_bits,) with dtype uint32 (bitmap format)
        """
        n_registers = 1 << self.p_bits
        registers = np.zeros(n_registers, dtype=np.uint32)
        
        for sign, (reg, zeros) in self.sign_to_reg_zeros.items():
            # Bitmap encoding: set bit at position (zeros) 
            registers[reg] |= (1 << zeros)
        
        return registers
    
    def to_hllset(self) -> HLLSet:
        """
        Get vocabulary as HLLSet object.
        
        Enables:
        - O(1) vocabulary comparison via tau similarity
        - Cross-installation vocabulary matching
        - Social/domain profiling by vocabulary overlap
        - Lazy vocabulary buildup with instant fingerprint
        - Full HLLSet operations (union, intersection, etc.)
        
        Returns:
            HLLSet object with vocabulary fingerprint
        """
        # Efficient: create empty HLLSet and set registers directly
        # (avoids re-hashing all vocabulary signs)
        hll = HLLSet(p_bits=self.p_bits)
        hll._core.set_registers(self.to_registers())
        hll._compute_name()
        return hll
    
    def vocabulary_cardinality(self) -> int:
        """Estimate vocabulary size from HLLSet (for verification)."""
        registers = self.to_registers()
        return estimate_cardinality(registers)
    
    def tau(self, other: 'HashVocabularyScheme') -> float:
        """
        O(1) similarity between two vocabularies.
        
        Uses HLLSet tau measure:
        - 1.0 = identical vocabularies (or one contains the other)
        - 0.0 = completely disjoint
        
        Use cases:
        - Compare vocabularies across installations
        - Detect domain/culture/specialty by vocabulary overlap
        - Find similar users by their vocabulary fingerprints
        
        Args:
            other: Another HashVocabularyScheme to compare
            
        Returns:
            Similarity score 0.0 to 1.0
        """
        hll_self = self.to_hllset()
        hll_other = other.to_hllset()
        
        # Use HLLSet intersection
        intersection = hll_self.intersect(hll_other)
        inter_card = intersection.cardinality()
        
        # tau = |A ∩ B| / min(|A|, |B|)
        self_card = len(self.known_signs)
        other_card = len(other.known_signs)
        min_card = min(self_card, other_card)
        
        if min_card == 0:
            return 0.0
        return inter_card / min_card
    
    def jaccard(self, other: 'HashVocabularyScheme') -> float:
        """
        Jaccard similarity between vocabularies.
        
        |A ∩ B| / |A ∪ B|
        """
        hll_self = self.to_hllset()
        hll_other = other.to_hllset()
        
        # Use HLLSet similarity (which is Jaccard)
        return hll_self.similarity(hll_other)
    
    def contains_vocabulary(self, other: 'HashVocabularyScheme') -> bool:
        """
        Check if this vocabulary contains all signs from other.
        
        Uses exact set membership check on known signs.
        """
        for sign in other.known_signs:
            if sign not in self.known_signs:
                return False
        return True
    
    def vocabulary_diff(self, other: 'HashVocabularyScheme') -> Set[str]:
        """Signs in self but not in other."""
        return set(self.known_signs.keys()) - set(other.known_signs.keys())
    
    def vocabulary_intersection(self, other: 'HashVocabularyScheme') -> Set[str]:
        """Signs in both vocabularies."""
        return set(self.known_signs.keys()) & set(other.known_signs.keys())
    
    def merge_vocabulary(self, other: 'HashVocabularyScheme') -> 'HashVocabularyScheme':
        """
        Merge two vocabularies, return new scheme with union of signs.
        
        Useful for combining vocabularies from different sources/installations.
        """
        merged = HashVocabularyScheme(
            p_bits=self.p_bits,
            h_bits=self.h_bits,
            _hash_seed=self._hash_seed
        )
        for sign in self.known_signs:
            merged.add_sign(sign)
        for sign in other.known_signs:
            merged.add_sign(sign)
        return merged


# Default scheme
DEFAULT_IDENTIFIER_SCHEME = HashIdentifierScheme()


# ═══════════════════════════════════════════════════════════════════════════
# UNIVERSAL IDENTIFIER: (reg, zeros)
# ═══════════════════════════════════════════════════════════════════════════

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
        """Convert to matrix index."""
        return self.reg * (config.h_bits - config.p_bits + 1) + self.zeros
    
    def __repr__(self) -> str:
        return f"UID(reg={self.reg}, zeros={self.zeros}, L{self.layer})"


def content_to_index(
    content: str, 
    layer: int, 
    config: Sparse3DConfig,
    scheme: Optional[IdentifierScheme] = None
) -> int:
    """
    Unified content → index mapping.
    
    This is the fundamental addressing function used everywhere.
    
    Args:
        content: Content string (token, n-gram, hieroglyph, etc.)
        layer: N-gram layer (0=1-gram, 1=2-gram, etc.)
        config: Sparse3DConfig
        scheme: Optional IdentifierScheme (defaults to HashIdentifierScheme)
    
    Returns:
        Matrix index for use in AM/W row/column
    
    The scheme determines HOW content maps to indices:
        - HashIdentifierScheme: content → hash → (reg, zeros) → index
        - VocabularyIdentifierScheme: content → vocabulary lookup → index
    """
    if scheme is not None:
        return scheme.to_index(content, layer)
    # Default: use hash-based scheme (uses DEFAULT_HASH_CONFIG)
    uid = UniversalID.from_content(content, layer)
    return uid.to_index(config)


# ═══════════════════════════════════════════════════════════════════════════
# SPARSE MATRIX (2D)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class SparseMatrix:
    """
    Immutable sparse matrix.
    
    Entries stored as frozenset of (row, col, value) tuples.
    """
    entries: FrozenSet[Tuple[int, int, float]]
    shape: Tuple[int, int]
    
    @classmethod
    def empty(cls, dim: int) -> 'SparseMatrix':
        return cls(entries=frozenset(), shape=(dim, dim))
    
    @classmethod
    def from_edges(cls, edges: List[Tuple[int, int, float]], dim: int) -> 'SparseMatrix':
        return cls(entries=frozenset(edges), shape=(dim, dim))
    
    @classmethod
    def from_dict(cls, d: Dict[int, Dict[int, float]], dim: int) -> 'SparseMatrix':
        entries = frozenset(
            (row, col, val)
            for row, cols in d.items()
            for col, val in cols.items()
        )
        return cls(entries=entries, shape=(dim, dim))
    
    def to_dict(self) -> Dict[int, Dict[int, float]]:
        d: Dict[int, Dict[int, float]] = {}
        for row, col, val in self.entries:
            if row not in d:
                d[row] = {}
            d[row][col] = val
        return d
    
    def __iter__(self) -> Iterator[Tuple[int, int, float]]:
        return iter(self.entries)
    
    @property
    def nnz(self) -> int:
        return len(self.entries)


# ═══════════════════════════════════════════════════════════════════════════
# SPARSE 3D MATRIX (Layered)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class Sparse3DMatrix:
    """
    Immutable 3D sparse matrix (layered).
    
    Each layer is a SparseMatrix. Layers correspond to n-gram sizes.
    """
    layers: Tuple[SparseMatrix, ...]
    shape: Tuple[int, int, int]  # (n_layers, dim, dim)
    
    @classmethod
    def empty(cls, n_layers: int, dim: int) -> 'Sparse3DMatrix':
        layers = tuple(SparseMatrix.empty(dim) for _ in range(n_layers))
        return cls(layers=layers, shape=(n_layers, dim, dim))
    
    @classmethod
    def from_am(cls, am: SparseAM3D, config: Sparse3DConfig) -> 'Sparse3DMatrix':
        layer_matrices = []
        for n in range(config.max_n):
            edges = list(am.tensor.layer_edges(n))
            matrix = SparseMatrix.from_edges(edges, config.dimension)
            layer_matrices.append(matrix)
        return cls(layers=tuple(layer_matrices), shape=(config.max_n, config.dimension, config.dimension))
    
    @classmethod
    def from_w(cls, W: Dict[int, Dict[int, Dict[int, float]]], config: Sparse3DConfig) -> 'Sparse3DMatrix':
        layer_matrices = []
        for n in range(config.max_n):
            if n in W:
                matrix = SparseMatrix.from_dict(W[n], config.dimension)
            else:
                matrix = SparseMatrix.empty(config.dimension)
            layer_matrices.append(matrix)
        return cls(layers=tuple(layer_matrices), shape=(config.max_n, config.dimension, config.dimension))
    
    def to_edges(self) -> List[Edge3D]:
        edges = []
        for n, layer in enumerate(self.layers):
            for row, col, val in layer:
                edges.append(Edge3D(n=n, row=row, col=col, value=val))
        return edges
    
    @property
    def nnz(self) -> int:
        return sum(layer.nnz for layer in self.layers)


# ═══════════════════════════════════════════════════════════════════════════
# PROJECTION OPERATIONS (π)
# ═══════════════════════════════════════════════════════════════════════════

def project_layer(M: Sparse3DMatrix, layer: int) -> SparseMatrix:
    """π_n: Extract single layer."""
    if 0 <= layer < len(M.layers):
        return M.layers[layer]
    return SparseMatrix.empty(M.shape[1])


def project_rows(M: SparseMatrix, rows: Set[int]) -> SparseMatrix:
    """π_R: Extract subset of rows."""
    new_entries = frozenset(
        (row, col, val) for row, col, val in M.entries if row in rows
    )
    return SparseMatrix(entries=new_entries, shape=M.shape)


def project_cols(M: SparseMatrix, cols: Set[int]) -> SparseMatrix:
    """π_C: Extract subset of columns."""
    new_entries = frozenset(
        (row, col, val) for row, col, val in M.entries if col in cols
    )
    return SparseMatrix(entries=new_entries, shape=M.shape)


def project_submatrix(M: SparseMatrix, rows: Set[int], cols: Set[int]) -> SparseMatrix:
    """π_{R,C}: Extract submatrix."""
    new_entries = frozenset(
        (row, col, val) for row, col, val in M.entries 
        if row in rows and col in cols
    )
    return SparseMatrix(entries=new_entries, shape=M.shape)


# ═══════════════════════════════════════════════════════════════════════════
# TRANSFORM OPERATIONS
# ═══════════════════════════════════════════════════════════════════════════

def transpose(M: SparseMatrix) -> SparseMatrix:
    """T: Transpose (swap rows/cols)."""
    new_entries = frozenset(
        (col, row, val) for row, col, val in M.entries
    )
    return SparseMatrix(entries=new_entries, shape=(M.shape[1], M.shape[0]))


def transpose_3d(M: Sparse3DMatrix) -> Sparse3DMatrix:
    """T: Transpose all layers."""
    new_layers = tuple(transpose(layer) for layer in M.layers)
    return Sparse3DMatrix(layers=new_layers, shape=M.shape)


def normalize_rows(M: SparseMatrix) -> SparseMatrix:
    """N: Row-normalize (sum to 1). Converts AM → W."""
    row_sums: Dict[int, float] = {}
    for row, col, val in M.entries:
        row_sums[row] = row_sums.get(row, 0.0) + val
    
    new_entries = frozenset(
        (row, col, val / row_sums[row]) if row_sums.get(row, 0) > 0 else (row, col, 0.0)
        for row, col, val in M.entries
    )
    return SparseMatrix(entries=new_entries, shape=M.shape)


def normalize_3d(M: Sparse3DMatrix) -> Sparse3DMatrix:
    """N: Normalize all layers."""
    return Sparse3DMatrix(
        layers=tuple(normalize_rows(layer) for layer in M.layers),
        shape=M.shape
    )


def scale(M: SparseMatrix, factor: float) -> SparseMatrix:
    """S_α: Scale all values."""
    new_entries = frozenset(
        (row, col, val * factor) for row, col, val in M.entries
    )
    return SparseMatrix(entries=new_entries, shape=M.shape)


# ═══════════════════════════════════════════════════════════════════════════
# FILTER OPERATIONS (σ)
# ═══════════════════════════════════════════════════════════════════════════

def filter_threshold(M: SparseMatrix, min_val: float = 0.0) -> SparseMatrix:
    """σ_θ: Keep entries ≥ threshold."""
    new_entries = frozenset(
        (row, col, val) for row, col, val in M.entries if val >= min_val
    )
    return SparseMatrix(entries=new_entries, shape=M.shape)


def filter_predicate(M: SparseMatrix, pred: Callable[[int, int, float], bool]) -> SparseMatrix:
    """σ_P: Keep entries where predicate is True."""
    new_entries = frozenset(
        (row, col, val) for row, col, val in M.entries if pred(row, col, val)
    )
    return SparseMatrix(entries=new_entries, shape=M.shape)


# ═══════════════════════════════════════════════════════════════════════════
# COMPOSITION OPERATIONS
# ═══════════════════════════════════════════════════════════════════════════

def merge_add(M1: SparseMatrix, M2: SparseMatrix) -> SparseMatrix:
    """+ : Element-wise addition."""
    combined: Dict[Tuple[int, int], float] = {}
    for row, col, val in M1.entries:
        combined[(row, col)] = combined.get((row, col), 0.0) + val
    for row, col, val in M2.entries:
        combined[(row, col)] = combined.get((row, col), 0.0) + val
    
    new_entries = frozenset(
        (row, col, val) for (row, col), val in combined.items()
    )
    return SparseMatrix(entries=new_entries, shape=M1.shape)


def merge_max(M1: SparseMatrix, M2: SparseMatrix) -> SparseMatrix:
    """∨ : Element-wise maximum."""
    combined: Dict[Tuple[int, int], float] = {}
    for row, col, val in M1.entries:
        key = (row, col)
        combined[key] = max(combined.get(key, 0.0), val)
    for row, col, val in M2.entries:
        key = (row, col)
        combined[key] = max(combined.get(key, 0.0), val)
    
    return SparseMatrix(
        entries=frozenset((r, c, v) for (r, c), v in combined.items()),
        shape=M1.shape
    )


def compose_chain(M1: SparseMatrix, M2: SparseMatrix) -> SparseMatrix:
    """∘ : Matrix multiplication (path composition)."""
    m2_by_row: Dict[int, List[Tuple[int, float]]] = {}
    for row, col, val in M2.entries:
        if row not in m2_by_row:
            m2_by_row[row] = []
        m2_by_row[row].append((col, val))
    
    result: Dict[Tuple[int, int], float] = {}
    for i, j, v1 in M1.entries:
        if j in m2_by_row:
            for k, v2 in m2_by_row[j]:
                key = (i, k)
                result[key] = result.get(key, 0.0) + v1 * v2
    
    return SparseMatrix(
        entries=frozenset((r, c, v) for (r, c), v in result.items()),
        shape=M1.shape
    )


def merge_3d_add(M1: Sparse3DMatrix, M2: Sparse3DMatrix) -> Sparse3DMatrix:
    """+ : Merge 3D matrices with addition."""
    new_layers = tuple(
        merge_add(l1, l2) for l1, l2 in zip(M1.layers, M2.layers)
    )
    return Sparse3DMatrix(layers=new_layers, shape=M1.shape)


# ═══════════════════════════════════════════════════════════════════════════
# PATH OPERATIONS
# ═══════════════════════════════════════════════════════════════════════════

def reachable_from(M: SparseMatrix, sources: Set[int], hops: int = 1) -> Set[int]:
    """Reach_k: Find nodes reachable in k hops from sources."""
    current = sources
    for _ in range(hops):
        next_set = set()
        for row, col, _ in M.entries:
            if row in current:
                next_set.add(col)
        current = next_set
    return current


def path_closure(M: SparseMatrix, max_hops: int = 10) -> SparseMatrix:
    """M*: Transitive closure (all paths up to max_hops)."""
    result = M
    current = M
    for _ in range(max_hops - 1):
        current = compose_chain(current, M)
        if current.nnz == 0:
            break
        result = merge_add(result, current)
    return result


# ═══════════════════════════════════════════════════════════════════════════
# LIFT/LOWER OPERATIONS
# ═══════════════════════════════════════════════════════════════════════════

def lift_to_layer(M: SparseMatrix, target_layer: int, n_layers: int) -> Sparse3DMatrix:
    """↑_n: Lift 2D matrix to layer n of 3D matrix."""
    layers = []
    for n in range(n_layers):
        if n == target_layer:
            layers.append(M)
        else:
            layers.append(SparseMatrix.empty(M.shape[0]))
    return Sparse3DMatrix(layers=tuple(layers), shape=(n_layers, M.shape[0], M.shape[1]))


def lower_aggregate(M: Sparse3DMatrix, agg: str = 'sum') -> SparseMatrix:
    """↓: Lower 3D to 2D by aggregating layers."""
    if agg == 'sum':
        result = M.layers[0]
        for layer in M.layers[1:]:
            result = merge_add(result, layer)
        return result
    elif agg == 'max':
        result = M.layers[0]
        for layer in M.layers[1:]:
            result = merge_max(result, layer)
        return result
    else:
        raise ValueError(f"Unknown aggregation: {agg}")


# ═══════════════════════════════════════════════════════════════════════════
# CROSS-STRUCTURE OPERATIONS (AM ↔ W)
# ═══════════════════════════════════════════════════════════════════════════

def am_to_w(AM: Sparse3DMatrix) -> Sparse3DMatrix:
    """AM → W: Convert adjacency counts to transition probabilities."""
    return normalize_3d(AM)


def w_to_am(W: Sparse3DMatrix, scale_factor: float = 1.0) -> Sparse3DMatrix:
    """W → AM: Convert probabilities back to counts (approximate)."""
    new_layers = tuple(scale(layer, scale_factor) for layer in W.layers)
    return Sparse3DMatrix(layers=new_layers, shape=W.shape)


# ═══════════════════════════════════════════════════════════════════════════
# LOOKUP TABLE (LUT) - Token Recovery
# ═══════════════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════════════
# UNIFIED PROCESSING PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class ProcessingResult:
    """
    Result of unified processing.
    
    All intermediate states preserved for inspection/debugging.
    """
    input_hllset: HLLSet
    input_basics: Tuple[BasicHLLSet3D, ...]
    input_edges: Tuple[Edge3D, ...]  # Edges from input tokens
    sub_hrt: SparseHRT3D
    context_edges: Tuple[Edge3D, ...]  # Edges from W context extension
    merged_hrt: SparseHRT3D


def tokenize(text: str) -> List[str]:
    """Simple tokenization."""
    return text.lower().strip().split()


def generate_ntokens(tokens: List[str], max_n: int = 3) -> List[Tuple[str, ...]]:
    """
    Generate n-tokens with START/END boundaries.
    
    Pattern: (START) → (a) → (a,b) → (a,b,c) → (b) → ... → (END)
    """
    ntokens = [START]
    for i in range(len(tokens)):
        for n in range(1, min(max_n + 1, len(tokens) - i + 1)):
            ntokens.append(tuple(tokens[i:i + n]))
    ntokens.append(END)
    return ntokens


def input_to_hllset(
    input_data: str,
    config: Sparse3DConfig,
    lut: LookupTable,
    max_n: int = 3
) -> Tuple[HLLSet, List[BasicHLLSet3D], List[Edge3D]]:
    """
    STEP 1: Convert input to HLLSet.
    
    Same for ingestion AND query.
    """
    tokens = tokenize(input_data)
    ntokens = generate_ntokens(tokens, max_n)
    
    hll = HLLSet(p_bits=config.p_bits)
    basics: List[BasicHLLSet3D] = []
    
    for ntoken in ntokens:
        lut.add_ntoken(ntoken)
        ntoken_text = " ".join(ntoken)
        hll = HLLSet.add(hll, ntoken_text)
        
        # Use UniversalID to compute consistent indices with LUT
        layer = 0 if ntoken in (START, END) else len(ntoken) - 1
        uid = UniversalID.from_content(ntoken_text, layer)
        basic = BasicHLLSet3D(n=layer, reg=uid.reg, zeros=uid.zeros)
        basics.append(basic)
    
    # Generate edges
    edges = []
    for i in range(len(ntokens) - 1):
        row_idx = lut.get_ntoken_index(ntokens[i])
        col_idx = lut.get_ntoken_index(ntokens[i + 1])
        
        col_ntoken = ntokens[i + 1]
        layer = 0 if col_ntoken in (START, END) else len(col_ntoken) - 1
        
        if row_idx is not None and col_idx is not None and layer < config.max_n:
            edges.append(Edge3D(n=layer, row=row_idx, col=col_idx, value=1.0))
    
    return hll, basics, edges


def build_sub_hrt(edges: List[Edge3D], config: Sparse3DConfig) -> SparseHRT3D:
    """
    STEP 2: Build isolated HRT from edges.
    """
    if not edges:
        am = SparseAM3D.from_edges(config, [])
        lattice = SparseLattice3D.from_sparse_am(am)
        return SparseHRT3D(am=am, lattice=lattice, config=config, lut=frozenset(), step=0)
    
    edge_dict: Dict[Tuple[int, int, int], float] = {}
    for edge in edges:
        key = (edge.n, edge.row, edge.col)
        edge_dict[key] = edge_dict.get(key, 0.0) + edge.value
    
    aggregated = [Edge3D(n=k[0], row=k[1], col=k[2], value=v) for k, v in edge_dict.items()]
    am = SparseAM3D.from_edges(config, aggregated)
    lattice = SparseLattice3D.from_sparse_am(am)
    
    return SparseHRT3D(am=am, lattice=lattice, config=config, lut=frozenset(), step=0)


def extend_with_context(
    sub_hrt: SparseHRT3D,
    current_W: Dict[int, Dict[int, Dict[int, float]]],
    input_basics: List[BasicHLLSet3D],
    config: Sparse3DConfig
) -> Tuple[SparseHRT3D, List[Edge3D]]:
    """
    STEP 3: Extend sub-HRT with context from current W.
    """
    input_indices = {b.to_index(config) for b in input_basics}
    context_edges: List[Edge3D] = []
    
    for n in range(config.max_n):
        if n not in current_W:
            continue
        for row_idx in input_indices:
            if row_idx in current_W[n]:
                for col_idx, prob in current_W[n][row_idx].items():
                    context_edges.append(Edge3D(n=n, row=row_idx, col=col_idx, value=prob))
    
    if not context_edges:
        return sub_hrt, []
    
    new_am = sub_hrt.am
    for edge in context_edges:
        new_am = new_am.with_edge(edge.n, edge.row, edge.col, edge.value)
    
    new_lattice = SparseLattice3D.from_sparse_am(new_am)
    return SparseHRT3D(
        am=new_am, lattice=new_lattice, config=config,
        lut=sub_hrt.lut, step=sub_hrt.step
    ), context_edges


def extend_with_intersected_context(
    sub_hrt: SparseHRT3D,
    current_W: Dict[int, Dict[int, Dict[int, float]]],
    input_basics: List[BasicHLLSet3D],
    config: Sparse3DConfig
) -> Tuple[SparseHRT3D, List[Edge3D]]:
    """
    STEP 3 (IMPROVED): Extend sub-HRT with INTERSECTED context from current W.
    
    Extended context = row_union(query) ∩ col_union(query)
    
    Where:
        row_union(query) = {col | ∃n: W[n][query_idx][col] > 0}
        col_union(query) = {row | ∃n: W[n][row][query_idx] > 0}
    
    The intersection narrows context to indices that appear in BOTH
    row and column relationships with the query, reducing noise from
    indices that only appear in one direction.
    """
    input_indices = {b.to_index(config) for b in input_basics}
    
    # Collect row-related indices: where query appears as row
    row_related: Set[int] = set()
    # Collect col-related indices: where query appears as col
    col_related: Set[int] = set()
    
    for n in range(config.max_n):
        if n not in current_W:
            continue
        
        for row_idx in input_indices:
            # Query as row → collect all columns
            if row_idx in current_W[n]:
                row_related.update(current_W[n][row_idx].keys())
        
        # Query as column → collect all rows
        for row, cols in current_W[n].items():
            for col in cols.keys():
                if col in input_indices:
                    col_related.add(row)
    
    # Intersected context: indices that appear in BOTH directions
    intersected_context = row_related & col_related
    
    if not intersected_context:
        return sub_hrt, []
    
    # Build context edges only for intersected indices
    context_edges: List[Edge3D] = []
    
    for n in range(config.max_n):
        if n not in current_W:
            continue
        for row_idx in input_indices:
            if row_idx in current_W[n]:
                for col_idx, prob in current_W[n][row_idx].items():
                    if col_idx in intersected_context:
                        context_edges.append(Edge3D(n=n, row=row_idx, col=col_idx, value=prob))
    
    if not context_edges:
        return sub_hrt, []
    
    new_am = sub_hrt.am
    for edge in context_edges:
        new_am = new_am.with_edge(edge.n, edge.row, edge.col, edge.value)
    
    new_lattice = SparseLattice3D.from_sparse_am(new_am)
    return SparseHRT3D(
        am=new_am, lattice=new_lattice, config=config,
        lut=sub_hrt.lut, step=sub_hrt.step
    ), context_edges


def merge_hrt(
    current_hrt: SparseHRT3D,
    sub_hrt: SparseHRT3D,
    config: Sparse3DConfig
) -> SparseHRT3D:
    """
    STEP 4: Merge sub-HRT into current (idempotent).
    
    Properties: commutative, associative.
    """
    all_edges: Dict[Tuple[int, int, int], float] = {}
    
    for n in range(config.max_n):
        for row, col, val in current_hrt.am.tensor.layer_edges(n):
            key = (n, row, col)
            all_edges[key] = all_edges.get(key, 0.0) + val
        for row, col, val in sub_hrt.am.tensor.layer_edges(n):
            key = (n, row, col)
            all_edges[key] = all_edges.get(key, 0.0) + val
    
    merged_edges = [Edge3D(n=k[0], row=k[1], col=k[2], value=v) for k, v in all_edges.items()]
    am = SparseAM3D.from_edges(config, merged_edges)
    lattice = SparseLattice3D.from_sparse_am(am)
    
    return SparseHRT3D(
        am=am, lattice=lattice, config=config,
        lut=frozenset(), step=max(current_hrt.step, sub_hrt.step) + 1
    )


def unified_process(
    input_data: str,
    current_hrt: SparseHRT3D,
    current_W: Dict[int, Dict[int, Dict[int, float]]],
    config: Sparse3DConfig,
    lut: LookupTable,
    max_n: int = 3
) -> ProcessingResult:
    """
    UNIFIED PROCESSING PIPELINE
    
    Same for ingestion AND query:
    INPUT → HLLSet → Sub-HRT → Extend → Merge
    """
    input_hll, input_basics, input_edges = input_to_hllset(input_data, config, lut, max_n)
    sub_hrt = build_sub_hrt(input_edges, config)
    extended_hrt, context_edges = extend_with_context(sub_hrt, current_W, input_basics, config)
    merged_hrt = merge_hrt(current_hrt, extended_hrt, config)
    
    return ProcessingResult(
        input_hllset=input_hll,
        input_basics=tuple(input_basics),
        input_edges=tuple(input_edges),
        sub_hrt=extended_hrt,
        context_edges=tuple(context_edges),
        merged_hrt=merged_hrt
    )


# ═══════════════════════════════════════════════════════════════════════════
# W MATRIX BUILDER
# ═══════════════════════════════════════════════════════════════════════════

def build_w_from_am(am: SparseAM3D, config: Sparse3DConfig) -> Dict[int, Dict[int, Dict[int, float]]]:
    """
    Build W (transition probabilities) from AM.
    
    W[n][row][col] = AM[n, row, col] / Σ_c AM[n, row, c]
    """
    W: Dict[int, Dict[int, Dict[int, float]]] = {}
    
    for n in range(config.max_n):
        W[n] = {}
        edges = am.tensor.layer_edges(n)
        
        row_sums: Dict[int, float] = {}
        row_edges: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
        
        for row, col, val in edges:
            row_sums[row] = row_sums.get(row, 0.0) + val
            row_edges[row].append((col, val))
        
        for row, edges_list in row_edges.items():
            W[n][row] = {}
            row_sum = row_sums[row]
            for col, val in edges_list:
                W[n][row][col] = val / row_sum if row_sum > 0 else 0.0
    
    return W


# ═══════════════════════════════════════════════════════════════════════════
# LAYER HLLSETS (for Cascading Disambiguation)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class LayerHLLSets:
    """
    Layer-specific HLLSets for cascading disambiguation.
    
    Maintains 3 HLLSets (L0, L1, L2) corresponding to n-gram layers,
    plus START_HLLSet for tokens following START symbol.
    
    These are cheap to maintain (just 3 additional HLLSets) and enable
    O(1) layer classification via intersection.
    """
    L0: HLLSet  # Layer 0 (1-grams)
    L1: HLLSet  # Layer 1 (2-grams)
    L2: HLLSet  # Layer 2 (3-grams)
    START: HLLSet  # START followers
    p_bits: int = 10
    
    @classmethod
    def empty(cls, p_bits: int = 10) -> 'LayerHLLSets':
        """Create empty layer HLLSets."""
        return cls(
            L0=HLLSet(p_bits=p_bits),
            L1=HLLSet(p_bits=p_bits),
            L2=HLLSet(p_bits=p_bits),
            START=HLLSet(p_bits=p_bits),
            p_bits=p_bits
        )
    
    @classmethod
    def from_am(cls, am: SparseAM3D, p_bits: int = 10) -> 'LayerHLLSets':
        """Build from existing SparseAM3D."""
        result = cls.empty(p_bits)
        
        for n in range(min(3, am.config.max_n)):
            rows, cols = am.layer_active(n)
            for idx in rows:
                result.add_to_layer(n, idx)
            for idx in cols:
                result.add_to_layer(n, idx)
        
        return result
    
    def add_to_layer(self, layer: int, idx: int):
        """Add index to appropriate layer HLLSet."""
        # Convert idx to string token for HLLSet API
        token = str(idx)
        if layer == 0:
            self.L0 = HLLSet.add(self.L0, token)
        elif layer == 1:
            self.L1 = HLLSet.add(self.L1, token)
        elif layer == 2:
            self.L2 = HLLSet.add(self.L2, token)
    
    def mark_start(self, idx: int):
        """Mark index as START follower."""
        token = str(idx)
        self.START = HLLSet.add(self.START, token)
    
    def get_layer(self, layer: int) -> HLLSet:
        """Get HLLSet for layer."""
        return [self.L0, self.L1, self.L2][layer]
    
    def merge(self, other: 'LayerHLLSets') -> 'LayerHLLSets':
        """Merge (union) two LayerHLLSets."""
        return LayerHLLSets(
            L0=self.L0.union(other.L0),
            L1=self.L1.union(other.L1),
            L2=self.L2.union(other.L2),
            START=self.START.union(other.START),
            p_bits=self.p_bits
        )
    
    def extract_known(self, query_hll: HLLSet) -> HLLSet:
        """
        Extract known tokens from query via layer intersections.
        
        Known = (Q ∩ L0) ∪ (Q ∩ L1) ∪ (Q ∩ L2)
        
        O(1) operation - no lookup table scan needed.
        
        Note: query_hll must have same p_bits as LayerHLLSets.
        """
        if query_hll.p_bits != self.p_bits:
            raise ValueError(
                f"Query HLLSet p_bits ({query_hll.p_bits}) != LayerHLLSets p_bits ({self.p_bits}). "
                f"Use HLLSet.from_batch(tokens, p_bits={self.p_bits}) to create compatible query."
            )
        known = query_hll.intersect(self.L0)
        known = known.union(query_hll.intersect(self.L1))
        known = known.union(query_hll.intersect(self.L2))
        return known
    
    def extract_unknown(self, query_hll: HLLSet) -> HLLSet:
        """
        Extract unknown tokens from query.
        
        Unknown = Q - Known = Q - ((Q ∩ L0) ∪ (Q ∩ L1) ∪ (Q ∩ L2))
        
        If Unknown.cardinality() > 0, query contains unseen tokens.
        O(1) operation for detecting novel content.
        """
        known = self.extract_known(query_hll)
        return query_hll.diff(known)
    
    def classify_query(self, query_hll: HLLSet) -> Dict[str, float]:
        """
        O(1) layer classification of query HLLSet.
        
        Returns similarity scores for each layer + unknown detection.
        """
        known = self.extract_known(query_hll)
        unknown = query_hll.diff(known)
        
        return {
            "L0_sim": query_hll.similarity(self.L0),
            "L1_sim": query_hll.similarity(self.L1),
            "L2_sim": query_hll.similarity(self.L2),
            "START_sim": query_hll.similarity(self.START),
            "known_ratio": known.cardinality() / max(query_hll.cardinality(), 1),
            "unknown_count": int(unknown.cardinality()),
        }
    
    def summary(self) -> Dict[str, int]:
        """Get cardinality summary."""
        return {
            "L0": int(self.L0.cardinality()),
            "L1": int(self.L1.cardinality()),
            "L2": int(self.L2.cardinality()),
            "START": int(self.START.cardinality()),
        }


@dataclass
class DisambiguationResult:
    """Result of disambiguating one index."""
    index: int
    layer: int
    constituent_indices: Set[int]
    
    def __repr__(self) -> str:
        return f"Disamb(idx={self.index}, L{self.layer}, n={len(self.constituent_indices)})"


def update_layer_hllsets(
    edges: List[Edge3D],
    layer_hllsets: LayerHLLSets,
    start_indices: Optional[Set[int]] = None
) -> LayerHLLSets:
    """
    Update layer HLLSets from new edges.
    
    Call during build_sub_hrt or merge to keep in sync.
    """
    for edge in edges:
        layer_hllsets.add_to_layer(edge.n, edge.row)
        layer_hllsets.add_to_layer(edge.n, edge.col)
    
    if start_indices:
        for idx in start_indices:
            layer_hllsets.mark_start(idx)
    
    return layer_hllsets


def cascading_disambiguate(
    query_indices: Set[int],
    am: SparseAM3D,
    layer_hllsets: LayerHLLSets,
    W: Dict[int, Dict[int, Dict[int, float]]],
    lut: LookupTable
) -> List[DisambiguationResult]:
    """
    Full Cascading Disambiguation Algorithm (5 steps).
    
    Given query indices, reconstruct token sequences by:
    1. Slice by layer (H_0, H_1, H_2 via intersection with LayerHLLSets)
    2. Find START candidates (H_0 ∩ START_HLLSet)
    3. Follow W transitions (W[s] ∩ H_1, W[2-gram] ∩ H_2)
    4. Decompose n-grams to constituent hashes
    5. Remove processed, repeat until H empty
    
    Returns list of DisambiguationResult ordered by discovery.
    """
    results = []
    processed = set()
    remaining = set(query_indices)
    
    # STEP 1: Classify by layer
    # Use AM's layer_active for accurate layer membership
    layer0_active = am.layer_active(0)[0] | am.layer_active(0)[1]
    layer1_active = (am.layer_active(1)[0] | am.layer_active(1)[1]) if am.config.max_n > 1 else set()
    layer2_active = (am.layer_active(2)[0] | am.layer_active(2)[1]) if am.config.max_n > 2 else set()
    
    H_0 = remaining & layer0_active  # 1-grams in query
    H_1 = remaining & layer1_active  # 2-grams in query
    H_2 = remaining & layer2_active  # 3-grams in query
    
    # STEP 2: Find START candidates
    # Look up START index from LUT
    start_idx = lut.get_ntoken_index(START)
    start_followers = set()
    
    if start_idx is not None and 0 in W:
        # Get all indices that follow START
        if start_idx in W[0]:
            start_followers = set(W[0][start_idx].keys())
    
    start_candidates = H_0 & start_followers
    
    # STEP 3: Follow W transitions from START candidates
    for start_token in start_candidates:
        if start_token in processed:
            continue
        
        # Try to build sequence: start_token → 2-gram → 3-gram
        sequence_results = _follow_transitions(
            start_token, W, H_0, H_1, H_2, am, processed
        )
        
        for result in sequence_results:
            results.append(result)
            processed.add(result.index)
            processed.update(result.constituent_indices)
            remaining.discard(result.index)
            remaining -= result.constituent_indices
    
    # STEP 4: Process remaining 3-grams (not reached via START)
    remaining_H2 = H_2 - processed
    for idx in remaining_H2:
        if idx in processed:
            continue
        
        # Decompose 3-gram to constituents
        constituents = _get_constituents(idx, 2, am)
        results.append(DisambiguationResult(idx, 2, constituents))
        processed.add(idx)
        processed.update(constituents)
        remaining.discard(idx)
        remaining -= constituents
    
    # Process remaining 2-grams
    remaining_H1 = H_1 - processed
    for idx in remaining_H1:
        if idx in processed:
            continue
        
        constituents = _get_constituents(idx, 1, am)
        results.append(DisambiguationResult(idx, 1, constituents))
        processed.add(idx)
        processed.update(constituents)
        remaining.discard(idx)
        remaining -= constituents
    
    # STEP 5: Process standalone 1-grams
    remaining_H0 = H_0 - processed
    for idx in remaining_H0:
        if idx in processed:
            continue
        results.append(DisambiguationResult(idx, 0, {idx}))
        processed.add(idx)
        remaining.discard(idx)
    
    return results


def _follow_transitions(
    start_token: int,
    W: Dict[int, Dict[int, Dict[int, float]]],
    H_0: Set[int],
    H_1: Set[int],
    H_2: Set[int],
    am: SparseAM3D,
    processed: Set[int]
) -> List[DisambiguationResult]:
    """
    Follow W transitions from a start token.
    
    start_token → 2-grams → 3-grams
    
    Returns results for the discovered sequence.
    """
    results = []
    
    # Start token is a 1-gram
    if start_token in processed:
        return results
    
    results.append(DisambiguationResult(start_token, 0, {start_token}))
    
    # Look for 2-grams following this token
    if 0 not in W or start_token not in W[0]:
        return results
    
    followers_1 = set(W[0][start_token].keys()) & H_0
    
    for follower in followers_1:
        if follower in processed:
            continue
        
        # Check if (start_token, follower) forms a 2-gram in H_1
        # The 2-gram index would be found via the AM connections
        two_gram_candidates = set()
        if 1 in W:
            for tg_idx in W.get(1, {}).get(start_token, {}).keys():
                if tg_idx in H_1:
                    two_gram_candidates.add(tg_idx)
        
        for tg_idx in two_gram_candidates:
            if tg_idx in processed:
                continue
            
            constituents_2g = _get_constituents(tg_idx, 1, am)
            results.append(DisambiguationResult(tg_idx, 1, constituents_2g))
            
            # Look for 3-grams
            if 2 in W and tg_idx in W[2]:
                for three_gram_idx in W[2][tg_idx].keys():
                    if three_gram_idx in H_2 and three_gram_idx not in processed:
                        constituents_3g = _get_constituents(three_gram_idx, 2, am)
                        results.append(DisambiguationResult(three_gram_idx, 2, constituents_3g))
    
    return results


def _get_constituents(idx: int, layer: int, am: SparseAM3D) -> Set[int]:
    """
    Get constituent indices for an n-gram.
    
    For layer 2 (3-gram): returns 2 constituent indices
    For layer 1 (2-gram): returns 1 constituent index
    For layer 0 (1-gram): returns self
    """
    if layer == 0:
        return {idx}
    
    constituents = set()
    for row, col, _ in am.layer_edges(layer):
        if row == idx:
            constituents.add(col)
        if col == idx:
            constituents.add(row)
    
    # If no constituents found via AM, return self
    return constituents if constituents else {idx}


def resolve_disambiguation(
    results: List[DisambiguationResult],
    lut: LookupTable
) -> Dict[int, List[str]]:
    """
    Resolve DisambiguationResults to tokens using LUT.
    
    Returns {index: [tokens]}
    """
    resolved = {}
    
    for r in results:
        tokens = []
        for idx in r.constituent_indices:
            ntoken = lut.index_to_ntokens.get(idx)
            if ntoken:
                if isinstance(ntoken, tuple):
                    tokens.extend(ntoken)
                else:
                    tokens.append(str(ntoken))
            else:
                tokens.append(f"<{idx}>")
        resolved[r.index] = tokens
    
    return resolved


# ═══════════════════════════════════════════════════════════════════════════
# COMMIT STORE - Track Processing History
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class Commit:
    """
    A timestamped commit representing a processing step.
    
    Each ingestion/query creates a commit, enabling:
    - Time-travel: access any historical state
    - Provenance: trace where data came from
    - Incremental update: resume from any point
    """
    id: str              # SHA-1 of content
    timestamp: float     # UNIX timestamp
    source: str          # File path or query ID
    perceptron: str      # Which perceptron processed it
    hrt: SparseHRT3D     # State after this commit
    W: Dict[int, Dict[int, Dict[int, float]]]  # W matrix after commit
    parent_id: Optional[str] = None  # Previous commit
    
    @classmethod
    def create(
        cls,
        hrt: SparseHRT3D,
        W: Dict[int, Dict[int, Dict[int, float]]],
        source: str,
        perceptron: str,
        parent_id: Optional[str] = None
    ) -> 'Commit':
        """Create a new commit."""
        import time
        timestamp = time.time()
        content = f"{source}:{timestamp}:{hrt.nnz}"
        commit_id = compute_sha1(content)
        
        return cls(
            id=commit_id,
            timestamp=timestamp,
            source=source,
            perceptron=perceptron,
            hrt=hrt,
            W=W,
            parent_id=parent_id
        )


class CommitStore:
    """
    Store for tracking commits (processing history).
    
    Enables:
    - Linear history of all processing
    - Rollback to any point
    - Branching for experiments
    """
    
    def __init__(self):
        self.commits: Dict[str, Commit] = {}
        self.head: Optional[str] = None
        self.history: List[str] = []
    
    def commit(
        self,
        hrt: SparseHRT3D,
        W: Dict[int, Dict[int, Dict[int, float]]],
        source: str,
        perceptron: str
    ) -> Commit:
        """Create and store a new commit."""
        c = Commit.create(hrt, W, source, perceptron, self.head)
        self.commits[c.id] = c
        self.head = c.id
        self.history.append(c.id)
        return c
    
    def get(self, commit_id: str) -> Optional[Commit]:
        """Get commit by ID."""
        return self.commits.get(commit_id)
    
    def rollback(self, commit_id: str) -> Optional[Commit]:
        """Set HEAD to a previous commit."""
        if commit_id in self.commits:
            self.head = commit_id
            return self.commits[commit_id]
        return None
    
    def latest(self) -> Optional[Commit]:
        """Get the latest commit."""
        return self.commits.get(self.head) if self.head else None
    
    def log(self, limit: int = 10) -> List[Commit]:
        """Get recent commits."""
        return [self.commits[cid] for cid in self.history[-limit:] if cid in self.commits]
    
    def __len__(self) -> int:
        return len(self.commits)


# ═══════════════════════════════════════════════════════════════════════════
# BOUNDED EVOLUTION STORE - Δ-based Growth Control
# ═══════════════════════════════════════════════════════════════════════════
# 
# Implements: T(t+1) = (T(t) ∪ N(t+1)) \ D(t)
# 
# Instead of full snapshots, we store:
# - Active state: current AM, W, HRT (bounded size)
# - Archive: only evicted entries D(t)
# - Deltas: N(t+1) for each step
#
# This breaks the pure snapshot model but bounds growth.
# ═══════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════
# FINGERPRINT INDEX: HLLSet-based commit lookup
# ═══════════════════════════════════════════════════════════════════════════
#
# Each commit/snapshot has an HLLSet "fingerprint" - a compressed representation
# of its indices. The FingerprintIndex provides:
#
# 1. SystemFingerprint: Union of ALL commit fingerprints
#    - "Has this index EVER existed in any commit?"
#    - No false negatives (if not in system fingerprint, never existed)
#    - May have false positives due to hash collisions
#
# 2. CommitFingerprints: Individual HLLSet per commit
#    - "Which commits contain this index?"
#    - O(1) similarity check per commit
#
# Use case: Before expensive history search, check system fingerprint first.
# If index not present → definitely never existed (skip history search)
# If index present → might exist, dive into commit fingerprints
# ═══════════════════════════════════════════════════════════════════════════

@dataclass 
class CommitFingerprint:
    """
    HLLSet fingerprint of a commit's state.
    
    Properties:
    - Fixed size regardless of commit size
    - O(1) similarity comparison via tau
    - No false negatives (if bit not set, index not in commit)
    - Union is idempotent (merge fingerprints safely)
    """
    commit_id: str
    timestamp: float
    hllset: np.ndarray
    cardinality: int
    source: str
    
    def contains_maybe(self, index: int, p_bits: int) -> bool:
        """
        Check if index MIGHT be in this commit.
        
        Returns:
            True: Index might be in commit (check further)
            False: Index definitely NOT in commit (skip)
        """
        n_registers = len(self.hllset)
        reg = index % n_registers
        zeros = 0
        temp = index >> p_bits
        while temp > 0 and (temp & 1) == 0:
            zeros += 1
            temp >>= 1
        # Bitmap check: is bit at position 'zeros' set?
        return bool(self.hllset[reg] & (1 << zeros))
    
    def tau(self, other: 'CommitFingerprint') -> float:
        """O(1) similarity between commits."""
        # Bitmap intersection = bitwise AND
        intersection = self.hllset & other.hllset
        inter_card = estimate_cardinality(intersection)
        min_card = min(self.cardinality, other.cardinality)
        if min_card == 0:
            return 0.0
        return inter_card / min_card


class FingerprintIndex:
    """
    LUT-like index for commit fingerprints.
    
    Provides fast "should we search history?" decisions:
    
    1. system_contains_maybe(index) → O(1)
       - If False: index NEVER existed in any commit
       - If True: index MIGHT exist, need to search
    
    2. find_candidate_commits(index) → O(n) but filtered
       - Returns commits that MIGHT contain the index
       - Much faster than searching all commits
    
    3. find_similar_commits(fingerprint) → O(n)
       - Find commits similar to a query fingerprint
    
    The SystemFingerprint is the union of all commit fingerprints.
    Due to hash collisions (using subset of hash), it may have false positives
    but NEVER false negatives - perfect for filtering.
    """
    
    def __init__(self, p_bits: int = 10):
        self.p_bits = p_bits
        self.n_registers = 2 ** p_bits
        
        # Individual commit fingerprints
        self.fingerprints: Dict[str, CommitFingerprint] = {}
        
        # System fingerprint = union of all commits (bitmap format)
        self.system_fingerprint: np.ndarray = np.zeros(self.n_registers, dtype=np.uint32)
        self.system_cardinality: int = 0
        
        # Metadata
        self.total_commits: int = 0
    
    def add_commit(
        self,
        commit_id: str,
        indices: Set[int],
        timestamp: float,
        source: str = "commit"
    ) -> CommitFingerprint:
        """
        Add a commit's fingerprint to the index.
        
        Args:
            commit_id: Unique identifier for the commit
            indices: Set of indices in this commit
            timestamp: When this commit occurred
            source: Label for the commit source
            
        Returns:
            The created CommitFingerprint
        """
        # Build HLL registers for this commit
        hllset = self._indices_to_registers(indices)
        cardinality = len(indices)
        
        fingerprint = CommitFingerprint(
            commit_id=commit_id,
            timestamp=timestamp,
            hllset=hllset,
            cardinality=cardinality,
            source=source
        )
        
        self.fingerprints[commit_id] = fingerprint
        self.total_commits += 1
        
        # Update system fingerprint (bitmap union = bitwise OR)
        self.system_fingerprint |= hllset
        self.system_cardinality = estimate_cardinality(self.system_fingerprint)
        
        return fingerprint
    
    def _indices_to_registers(self, indices: Set[int]) -> np.ndarray:
        """Convert indices to HLL register array (bitmap format)."""
        hllset = np.zeros(self.n_registers, dtype=np.uint32)
        
        for idx in indices:
            reg = idx % self.n_registers
            zeros = 0
            temp = idx >> self.p_bits
            while temp > 0 and (temp & 1) == 0:
                zeros += 1
                temp >>= 1
            # Bitmap encoding: set bit at position zeros
            hllset[reg] |= (1 << zeros)
        
        return hllset
    
    def system_contains_maybe(self, index: int) -> bool:
        """
        Check if index MIGHT exist in ANY commit.
        
        This is the first-level filter:
        - False → Index NEVER existed (100% certain, skip history search)
        - True → Index MIGHT exist (need to check individual commits)
        
        No false negatives, may have false positives due to hash collisions.
        """
        reg = index % self.n_registers
        zeros = 0
        temp = index >> self.p_bits
        while temp > 0 and (temp & 1) == 0:
            zeros += 1
            temp >>= 1
        # Bitmap check: is bit at position 'zeros' set?
        return bool(self.system_fingerprint[reg] & (1 << zeros))
    
    def find_candidate_commits(self, index: int) -> List[CommitFingerprint]:
        """
        Find commits that MIGHT contain the given index.
        
        First checks system fingerprint, then filters individual commits.
        
        Returns:
            List of CommitFingerprints that might contain the index
        """
        # Fast path: if not in system, definitely not in any commit
        if not self.system_contains_maybe(index):
            return []
        
        # Check individual commits
        candidates = []
        for fp in self.fingerprints.values():
            if fp.contains_maybe(index, self.p_bits):
                candidates.append(fp)
        
        return candidates
    
    def find_similar_commits(
        self,
        query_indices: Set[int],
        top_k: int = 5
    ) -> List[Tuple[float, CommitFingerprint]]:
        """
        Find commits most similar to query indices.
        
        Returns:
            List of (similarity, fingerprint) tuples sorted by similarity
        """
        query_hllset = self._indices_to_registers(query_indices)
        query_card = len(query_indices)
        
        similarities = []
        for fp in self.fingerprints.values():
            # Bitmap intersection = bitwise AND
            intersection = query_hllset & fp.hllset
            inter_card = estimate_cardinality(intersection)
            min_card = min(query_card, fp.cardinality)
            tau = inter_card / min_card if min_card > 0 else 0.0
            similarities.append((tau, fp))
        
        similarities.sort(key=lambda x: x[0], reverse=True)
        return similarities[:top_k]
    
    def commits_containing(self, indices: Set[int]) -> Set[str]:
        """
        Find commits that MIGHT contain ALL given indices.
        
        Intersection of candidate sets for each index.
        """
        if not indices:
            return set(self.fingerprints.keys())
        
        # Start with candidates for first index
        idx_iter = iter(indices)
        first_idx = next(idx_iter)
        candidates = {fp.commit_id for fp in self.find_candidate_commits(first_idx)}
        
        # Intersect with candidates for remaining indices
        for idx in idx_iter:
            if not candidates:
                break
            idx_candidates = {fp.commit_id for fp in self.find_candidate_commits(idx)}
            candidates &= idx_candidates
        
        return candidates
    
    def stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        return {
            'total_commits': self.total_commits,
            'system_cardinality': self.system_cardinality,
            'n_registers': self.n_registers,
            'p_bits': self.p_bits,
            'avg_commit_cardinality': (
                sum(fp.cardinality for fp in self.fingerprints.values()) / self.total_commits
                if self.total_commits > 0 else 0
            ),
        }
    
    def rebuild_system_fingerprint(self):
        """Rebuild system fingerprint from all commits (after deletions)."""
        self.system_fingerprint = np.zeros(self.n_registers, dtype=np.uint32)
        for fp in self.fingerprints.values():
            # Bitmap union = bitwise OR
            self.system_fingerprint |= fp.hllset
        self.system_cardinality = estimate_cardinality(self.system_fingerprint)


@dataclass
class EvictionRecord:
    """
    Record of evicted entries D(t) at time t.
    
    Instead of storing full snapshots, we store only what was removed.
    This enables reconstruction while bounding active state size.
    """
    timestamp: float
    evicted_edges: Tuple[Edge3D, ...]  # Edges removed from AM
    evicted_indices: FrozenSet[int]    # Indices removed from active set
    evicted_ntokens: Dict[int, Set[Tuple[int, Tuple[str, ...]]]]  # LUT entries
    reason: str  # 'age', 'lru', 'capacity', etc.
    
    def __post_init__(self):
        # Ensure immutability
        if not isinstance(self.evicted_edges, tuple):
            object.__setattr__(self, 'evicted_edges', tuple(self.evicted_edges))
        if not isinstance(self.evicted_indices, frozenset):
            object.__setattr__(self, 'evicted_indices', frozenset(self.evicted_indices))


@dataclass
class StateSnapshot:
    """
    Compressed HLLSet representation of system state at time t.
    
    Key insight: HLLSet provides O(1) similarity comparison via tau measure.
    This enables "memory archaeology" - finding historical states most similar
    to current query with constant-time comparison per archived state.
    
    CA Property: Same indices always produce same HLLSet bits, so:
    - Current state HLLSet is consistent with any archived state containing same indices
    - Union of HLLSets is idempotent (no conflicts)
    - tau(A, B) measures semantic similarity between states
    """
    timestamp: float
    hllset: np.ndarray           # Compressed state representation
    cardinality: int             # Estimated |active_indices| at snapshot time
    source: str                  # 'evolution', 'checkpoint', 'manual'
    metadata: Optional[Dict[str, Any]] = None
    
    def tau(self, other: 'StateSnapshot') -> float:
        """
        O(1) similarity measure between states.
        
        Uses HLLSet intersection/min formula:
        tau(A, B) = |A ∩ B| / min(|A|, |B|)
        """
        # Intersection cardinality via HLLSet (bitmap: bitwise AND)
        intersection = self.hllset & other.hllset
        inter_card = estimate_cardinality(intersection)
        
        min_card = min(self.cardinality, other.cardinality)
        if min_card == 0:
            return 0.0
        return inter_card / min_card
    
    def jaccard(self, other: 'StateSnapshot') -> float:
        """
        Jaccard similarity: |A ∩ B| / |A ∪ B|
        """
        # Bitmap operations: AND for intersection, OR for union
        intersection = self.hllset & other.hllset
        union = self.hllset | other.hllset
        
        inter_card = estimate_cardinality(intersection)
        union_card = estimate_cardinality(union)
        
        if union_card == 0:
            return 0.0
        return inter_card / union_card


def estimate_cardinality(hllset: np.ndarray) -> int:
    """
    Estimate cardinality from HLLSet using HyperLogLog formula.
    
    Works with bitmap format: each register is uint32 where bit k is set
    when an element with k trailing zeros was observed.
    
    For HLL formula, we need max_zeros + 1 (the traditional register value).
    If highest set bit is at position k, then max_zeros = k, so value = k + 1.
    """
    m = len(hllset)
    alpha = 0.7213 / (1 + 1.079 / m)  # Bias correction
    
    # Harmonic mean of 2^(-register_value) where register_value = max_zeros + 1
    Z = 0.0
    zero_registers = 0
    for val in hllset:
        if val == 0:
            zero_registers += 1
            Z += 1.0  # 2^(-0) = 1 for empty register
        else:
            # Highest set bit position = max_zeros observed
            # Register value for HLL = max_zeros + 1
            max_zeros = int(val).bit_length() - 1
            register_value = max_zeros + 1
            # Clamp to avoid overflow
            clamped = min(register_value, 63)
            Z += 2.0 ** (-clamped)
    
    E = alpha * m * m / Z
    
    # Small range correction
    if E <= 2.5 * m and zero_registers > 0:
        E = m * np.log(m / zero_registers)
    
    return int(E)


@dataclass
class DeltaRecord:
    """
    Record of new entries N(t+1) at time t+1.
    
    Captures what was added in each evolution step.
    """
    timestamp: float
    new_edges: Tuple[Edge3D, ...]
    new_indices: FrozenSet[int]
    source: str  # Where this data came from
    
    def __post_init__(self):
        if not isinstance(self.new_edges, tuple):
            object.__setattr__(self, 'new_edges', tuple(self.new_edges))
        if not isinstance(self.new_indices, frozenset):
            object.__setattr__(self, 'new_indices', frozenset(self.new_indices))


@dataclass
class EvolutionState:
    """
    Current state T(t) with bounded size.
    
    Tracks:
    - Active indices (bounded)
    - Age/usage for eviction policy
    - Current AM, W
    """
    active_indices: Set[int]
    index_age: Dict[int, float]       # index → last access time
    index_usage: Dict[int, int]       # index → access count
    capacity: int                      # Maximum active indices
    
    @classmethod
    def empty(cls, capacity: int = 100000) -> 'EvolutionState':
        return cls(
            active_indices=set(),
            index_age={},
            index_usage=defaultdict(int),
            capacity=capacity
        )
    
    def touch(self, indices: Set[int], timestamp: float):
        """Update age and usage for accessed indices."""
        for idx in indices:
            self.index_age[idx] = timestamp
            self.index_usage[idx] = self.index_usage.get(idx, 0) + 1
            self.active_indices.add(idx)
    
    def needs_eviction(self) -> bool:
        """Check if we need to evict entries."""
        return len(self.active_indices) > self.capacity
    
    def select_for_eviction(
        self, 
        n: int, 
        policy: str = 'lru'
    ) -> Set[int]:
        """
        Select n indices for eviction based on policy.
        
        Policies:
        - 'lru': Least Recently Used
        - 'lfu': Least Frequently Used  
        - 'age': Oldest entries
        - 'combined': LRU + LFU hybrid
        """
        if policy == 'lru':
            # Sort by last access time, evict oldest
            sorted_indices = sorted(
                self.active_indices,
                key=lambda i: self.index_age.get(i, 0)
            )
        elif policy == 'lfu':
            # Sort by usage count, evict least used
            sorted_indices = sorted(
                self.active_indices,
                key=lambda i: self.index_usage.get(i, 0)
            )
        elif policy == 'age':
            # Same as LRU but explicit naming
            sorted_indices = sorted(
                self.active_indices,
                key=lambda i: self.index_age.get(i, 0)
            )
        elif policy == 'combined':
            # Combine LRU and LFU: score = age_rank + usage_rank
            import time
            now = time.time()
            sorted_indices = sorted(
                self.active_indices,
                key=lambda i: (
                    (now - self.index_age.get(i, 0)) +  # Age penalty
                    (1.0 / (self.index_usage.get(i, 1) + 1))  # Low usage penalty
                ),
                reverse=True  # Highest penalty = evict first
            )
        else:
            raise ValueError(f"Unknown eviction policy: {policy}")
        
        return set(sorted_indices[:n])
    
    def evict(self, indices: Set[int]):
        """Remove indices from active set."""
        self.active_indices -= indices
        for idx in indices:
            self.index_age.pop(idx, None)
            self.index_usage.pop(idx, None)


class BoundedEvolutionStore:
    """
    Store implementing bounded growth via evolution equation:
    
        T(t+1) = (T(t) ∪ N(t+1)) \\ D(t)
    
    Key property: CONFLICT-FREE due to Content Addressable (CA) identification.
    
    CA Guarantee:
    - Same content always produces same index (hash-based or vocabulary-based)
    - Evicted index X can be re-activated without conflict
    - Archive history is complementary, not conflicting
    - Re-encountered content simply "reheats" its index
    
    Implications:
    - Active set + Archive = complete knowledge (no duplicates, no conflicts)
    - query_with_archive() returns consistent results regardless of eviction state
    - Eviction is purely about memory bounds, not data loss
    
    Key differences from CommitStore:
    - No full snapshots (bounded memory)
    - Archives only evicted entries D(t)
    - Can reconstruct historical states from deltas
    - Supports various eviction policies
    
    HLLSet Memory Feature:
    - Each evolution step can snapshot state as compressed HLLSet
    - O(1) similarity comparison between any two states
    - "Memory archaeology": find best-matching historical state instantly
    - tau(current, archived) finds "deepest memory from childhood"
    
    Trade-offs:
    - ✓ Bounded active state size
    - ✓ Efficient for vocabulary-based schemes
    - ✓ Conflict-free due to CA property
    - ✓ O(1) state similarity via HLLSet
    - ✗ Reconstruction requires replay (slower rollback)
    - ✗ Cannot rollback beyond archived deltas
    """
    
    def __init__(
        self,
        config: Sparse3DConfig,
        capacity: int = 100000,
        eviction_policy: str = 'lru',
        eviction_batch: int = 1000,
        snapshot_interval: int = 10  # Take HLLSet snapshot every N evolutions
    ):
        self.config = config
        self.capacity = capacity
        self.eviction_policy = eviction_policy
        self.eviction_batch = eviction_batch
        self.snapshot_interval = snapshot_interval
        
        # Current state
        self.state = EvolutionState.empty(capacity)
        self.am: Optional[SparseAM3D] = None
        self.W: Dict[int, Dict[int, Dict[int, float]]] = {}
        self.lut: Optional[LookupTable] = None
        
        # History (deltas only, not full snapshots)
        self.delta_history: List[DeltaRecord] = []
        self.eviction_history: List[EvictionRecord] = []
        
        # Archive (compressed storage of evicted data)
        self.archive_edges: List[Edge3D] = []
        self.archive_indices: Set[int] = set()  # Fast lookup for is_archived()
        self.archive_ntokens: Dict[int, Set[Tuple[int, Tuple[str, ...]]]] = defaultdict(set)
        
        # HLLSet state snapshots (compressed memory)
        self.state_snapshots: List[StateSnapshot] = []
        self.evolution_count: int = 0
        
        # Metrics
        self.total_evicted = 0
        self.total_added = 0
    
    def initialize(self, lut: LookupTable):
        """Initialize with LUT."""
        self.lut = lut
        self.am = SparseAM3D.from_edges(self.config, [])
    
    def evolve(
        self,
        new_edges: List[Edge3D],
        source: str = "input"
    ) -> Tuple[int, int]:
        """
        Evolve state: T(t+1) = (T(t) ∪ N(t+1)) \\ D(t)
        
        Returns:
            (n_added, n_evicted)
        """
        import time
        timestamp = time.time()
        
        # N(t+1): New indices from new edges
        new_indices = set()
        for edge in new_edges:
            new_indices.add(edge.row)
            new_indices.add(edge.col)
        
        # Record delta
        delta = DeltaRecord(
            timestamp=timestamp,
            new_edges=tuple(new_edges),
            new_indices=frozenset(new_indices),
            source=source
        )
        self.delta_history.append(delta)
        
        # Touch new indices (update age/usage)
        self.state.touch(new_indices, timestamp)
        self.total_added += len(new_indices)
        
        # Check if eviction needed
        n_evicted = 0
        if self.state.needs_eviction():
            n_evicted = self._perform_eviction(timestamp)
        
        # Update AM with new edges (after eviction)
        self._update_am(new_edges)
        
        # Rebuild W from updated AM
        self.W = build_w_from_am(self.am, self.config)
        
        # Increment evolution counter and maybe snapshot
        self.evolution_count += 1
        if self.snapshot_interval > 0 and self.evolution_count % self.snapshot_interval == 0:
            self._take_snapshot(timestamp, source)
        
        return len(new_indices), n_evicted
    
    def _take_snapshot(self, timestamp: float, source: str):
        """
        Take HLLSet snapshot of current state.
        
        Compresses active_indices into fixed-size HLLSet for O(1) similarity.
        """
        if self.am is None:
            return
        
        # Build HLL registers from current active indices
        hllset = self._indices_to_registers(self.state.active_indices)
        cardinality = len(self.state.active_indices)
        
        snapshot = StateSnapshot(
            timestamp=timestamp,
            hllset=hllset,
            cardinality=cardinality,
            source=source,
            metadata={
                'evolution_count': self.evolution_count,
                'total_evicted': self.total_evicted,
                'delta_count': len(self.delta_history),
            }
        )
        self.state_snapshots.append(snapshot)
    
    def _indices_to_registers(self, indices: Set[int]) -> np.ndarray:
        """Convert set of indices to HLL register array (bitmap format)."""
        # Use config to determine HLLSet size
        n_registers = 2 ** self.config.p_bits
        hllset = np.zeros(n_registers, dtype=np.uint32)
        
        for idx in indices:
            # Use idx to determine register and value
            # Register = first p_bits of hash
            reg = idx % n_registers
            # Count trailing zeros in idx
            zeros = 0
            temp = idx >> self.config.p_bits
            while temp > 0 and (temp & 1) == 0:
                zeros += 1
                temp >>= 1
            # Bitmap encoding: set bit at position zeros
            hllset[reg] |= (1 << zeros)
        
        return hllset
    
    def take_manual_snapshot(self, label: str = "manual") -> StateSnapshot:
        """
        Manually take a snapshot (e.g., before major operation).
        
        Returns the snapshot for reference.
        """
        import time
        timestamp = time.time()
        self._take_snapshot(timestamp, label)
        return self.state_snapshots[-1]
    
    def current_state_hllset(self) -> StateSnapshot:
        """Get HLLSet representation of current state (without archiving)."""
        import time
        hllset = self._indices_to_registers(self.state.active_indices)
        return StateSnapshot(
            timestamp=time.time(),
            hllset=hllset,
            cardinality=len(self.state.active_indices),
            source="current",
            metadata=None
        )
    
    def find_similar_memories(
        self, 
        query_indices: Set[int],
        top_k: int = 5
    ) -> List[Tuple[float, StateSnapshot]]:
        """
        Find archived states most similar to query.
        
        "Memory archaeology" - O(1) per comparison, O(n) total for n snapshots.
        
        Returns:
            List of (similarity, snapshot) tuples, sorted by similarity descending.
        """
        if not self.state_snapshots:
            return []
        
        # Convert query to HLL registers
        query_hllset = self._indices_to_registers(query_indices)
        query_snapshot = StateSnapshot(
            timestamp=0,
            hllset=query_hllset,
            cardinality=len(query_indices),
            source="query",
            metadata=None
        )
        
        # Compute similarity with each archived state
        similarities = []
        for snapshot in self.state_snapshots:
            sim = query_snapshot.tau(snapshot)
            similarities.append((sim, snapshot))
        
        # Sort by similarity descending
        similarities.sort(key=lambda x: x[0], reverse=True)
        return similarities[:top_k]
    
    def find_deepest_memory(self, query_indices: Set[int]) -> Optional[StateSnapshot]:
        """
        Find the single best-matching historical state.
        
        "Bringing childhood memories to life" - find the archived state
        that most closely resembles the query.
        """
        matches = self.find_similar_memories(query_indices, top_k=1)
        if matches:
            return matches[0][1]
        return None
    
    def memory_timeline(self) -> List[Dict[str, Any]]:
        """
        Get timeline of all state snapshots with metadata.
        
        Useful for visualizing system evolution.
        """
        timeline = []
        for snapshot in self.state_snapshots:
            timeline.append({
                'timestamp': snapshot.timestamp,
                'cardinality': snapshot.cardinality,
                'source': snapshot.source,
                'metadata': snapshot.metadata,
            })
        return timeline
    
    def _perform_eviction(self, timestamp: float) -> int:
        """
        Perform eviction: compute D(t) and archive it.
        
        Returns number of evicted indices.
        """
        # How many to evict?
        overflow = len(self.state.active_indices) - self.capacity
        n_to_evict = max(overflow, self.eviction_batch)
        
        # Select indices for eviction
        evict_indices = self.state.select_for_eviction(
            n_to_evict, 
            self.eviction_policy
        )
        
        # Collect edges involving evicted indices
        evicted_edges = []
        for n in range(self.config.max_n):
            for row, col, val in self.am.tensor.layer_edges(n):
                if row in evict_indices or col in evict_indices:
                    evicted_edges.append(Edge3D(n=n, row=row, col=col, value=val))
        
        # Collect LUT entries for evicted indices
        evicted_ntokens = {}
        if self.lut:
            for idx in evict_indices:
                ntokens = self.lut.index_to_ntokens.get(idx, set())
                if ntokens:
                    evicted_ntokens[idx] = ntokens.copy()
        
        # Record eviction
        eviction = EvictionRecord(
            timestamp=timestamp,
            evicted_edges=tuple(evicted_edges),
            evicted_indices=frozenset(evict_indices),
            evicted_ntokens=evicted_ntokens,
            reason=self.eviction_policy
        )
        self.eviction_history.append(eviction)
        
        # Archive evicted data
        self.archive_edges.extend(evicted_edges)
        self.archive_indices.update(evict_indices)  # Fast lookup set
        for idx, ntokens in evicted_ntokens.items():
            self.archive_ntokens[idx].update(ntokens)
        
        # Perform eviction on state
        self.state.evict(evict_indices)
        self.total_evicted += len(evict_indices)
        
        return len(evict_indices)
    
    def _update_am(self, new_edges: List[Edge3D]):
        """Update AM with new edges, removing evicted ones."""
        # Get current edges that are still active
        active_edges = []
        for n in range(self.config.max_n):
            for row, col, val in self.am.tensor.layer_edges(n):
                if row in self.state.active_indices and col in self.state.active_indices:
                    active_edges.append(Edge3D(n=n, row=row, col=col, value=val))
        
        # Add new edges (only if both endpoints are active)
        for edge in new_edges:
            if edge.row in self.state.active_indices and edge.col in self.state.active_indices:
                active_edges.append(edge)
        
        # Rebuild AM from active edges
        self.am = SparseAM3D.from_edges(self.config, active_edges)
    
    def query_with_archive(
        self,
        query_indices: Set[int],
        include_archived: bool = True
    ) -> Set[int]:
        """
        Query including archived data.
        
        If an index was evicted, we can still find its relationships
        from the archive.
        """
        results = set()
        
        # Search active W
        for layer in self.W.values():
            for idx in query_indices:
                if idx in layer:
                    results.update(layer[idx].keys())
        
        # Search archive if requested
        if include_archived:
            archived_indices = set()
            for edge in self.archive_edges:
                if edge.row in query_indices:
                    archived_indices.add(edge.col)
                if edge.col in query_indices:
                    archived_indices.add(edge.row)
            results.update(archived_indices)
        
        return results
    
    def restore_from_archive(self, indices: Set[int]) -> List[Edge3D]:
        """
        Restore archived edges for given indices.
        
        Useful for "reheating" cold data back into active state.
        """
        restored = []
        for edge in self.archive_edges:
            if edge.row in indices or edge.col in indices:
                restored.append(edge)
        return restored
    
    def reheat(self, indices: Set[int]) -> Tuple[int, int]:
        """
        Reheat archived indices back into active state.
        
        CA Property: Same content → same index, so reheating is conflict-free.
        The index simply moves from "archived" to "active" status.
        
        Returns:
            (n_reheated, n_evicted) - may need to evict to make room
        """
        import time
        timestamp = time.time()
        
        # Find archived edges for these indices
        archived_edges = self.restore_from_archive(indices)
        
        # Treat as evolution step (N = reheated indices, may trigger eviction)
        if archived_edges:
            return self.evolve(archived_edges, source="reheat")
        return (0, 0)
    
    def is_archived(self, index: int) -> bool:
        """Check if index exists in archive (was evicted at some point)."""
        return index in self.archive_indices
    
    def is_active(self, index: int) -> bool:
        """Check if index is currently active."""
        return index in self.state.active_indices
    
    def index_status(self, index: int) -> str:
        """
        Get status of an index.
        
        CA Property ensures no conflicts:
        - 'active': Currently in working set
        - 'archived': Was evicted, data preserved in archive
        - 'active+archived': Re-activated after eviction (most comprehensive data)
        - 'unknown': Never seen
        """
        active = self.is_active(index)
        archived = self.is_archived(index)
        
        if active and archived:
            return 'active+archived'  # Best case: full history
        elif active:
            return 'active'
        elif archived:
            return 'archived'
        else:
            return 'unknown'
    
    def stats(self) -> Dict[str, Any]:
        """Get store statistics."""
        return {
            'active_indices': len(self.state.active_indices),
            'capacity': self.capacity,
            'utilization': len(self.state.active_indices) / self.capacity,
            'total_added': self.total_added,
            'total_evicted': self.total_evicted,
            'archive_edges': len(self.archive_edges),
            'archive_indices': len(self.archive_indices),
            'archive_ntokens': len(self.archive_ntokens),
            'delta_count': len(self.delta_history),
            'eviction_count': len(self.eviction_history),
            'eviction_policy': self.eviction_policy,
            # HLLSet memory snapshots
            'snapshot_count': len(self.state_snapshots),
            'snapshot_interval': self.snapshot_interval,
            'evolution_count': self.evolution_count,
        }
    
    def conservation_check(self) -> Dict[str, int]:
        """
        Check conservation: |N| ≈ |D| over time.
        
        From Noether-inspired stability condition.
        """
        total_n = sum(len(d.new_indices) for d in self.delta_history)
        total_d = sum(len(e.evicted_indices) for e in self.eviction_history)
        
        return {
            'total_new': total_n,
            'total_deleted': total_d,
            'imbalance': total_n - total_d,
            'active_size': len(self.state.active_indices),
            'stable': abs(total_n - total_d) < self.capacity * 0.1
        }


# ═══════════════════════════════════════════════════════════════════════════
# PERCEPTRON BASE CLASS - Sense Phase
# ═══════════════════════════════════════════════════════════════════════════

from abc import ABC, abstractmethod
from pathlib import Path


class Perceptron(ABC):
    """
    Base class for perceptrons - sense input and convert to HLLSet.
    
    Each perceptron:
    1. Finds/accepts its input type
    2. Extracts text content
    3. Processes via unified pipeline
    4. Commits after processing
    
    Part of sense-process-act loop:
        Perceptron (sense) → Pipeline (process) → Actuator (act)
    """
    
    def __init__(self, name: str, extensions: List[str], config: Sparse3DConfig):
        self.name = name
        self.extensions = extensions
        self.config = config
        self.lut: Optional[LookupTable] = None
        self.files_processed = 0
        self.total_tokens = 0
    
    def initialize(self, lut: LookupTable):
        """Initialize with shared LUT."""
        self.lut = lut
    
    def find_files(self, root: Path, exclude_dirs: Set[str] = None) -> Iterator[Path]:
        """Find all files matching extensions."""
        exclude_dirs = exclude_dirs or {'__pycache__', '.git', 'build', '.ipynb_checkpoints', 'deprecated'}
        
        for path in root.rglob('*'):
            if path.is_file() and path.suffix in self.extensions:
                if not any(ex in path.parts for ex in exclude_dirs):
                    yield path
    
    @abstractmethod
    def extract_text(self, path: Path) -> str:
        """Extract text content from input."""
        pass
    
    def process_file(
        self,
        path: Path,
        current_hrt: SparseHRT3D,
        current_W: Dict[int, Dict[int, Dict[int, float]]],
        store: CommitStore,
        max_n: int = 3
    ) -> Tuple[SparseHRT3D, Dict[int, Dict[int, Dict[int, float]]], Optional[Commit]]:
        """
        Process a single file and commit.
        
        Returns (new_hrt, new_W, commit)
        """
        if not self.lut:
            raise RuntimeError("Perceptron not initialized - call initialize(lut) first")
        
        text = self.extract_text(path)
        if not text.strip():
            return current_hrt, current_W, None
        
        # Unified processing
        result = unified_process(
            text,
            current_hrt,
            current_W,
            self.config,
            self.lut,
            max_n
        )
        
        # Update state
        new_hrt = result.merged_hrt
        new_W = build_w_from_am(new_hrt.am, self.config)
        
        # Commit
        commit = store.commit(new_hrt, new_W, str(path), self.name)
        
        self.files_processed += 1
        self.total_tokens = len(self.lut.ntoken_to_index)
        
        return new_hrt, new_W, commit
    
    def process_all(
        self,
        root: Path,
        current_hrt: SparseHRT3D,
        current_W: Dict[int, Dict[int, Dict[int, float]]],
        store: CommitStore,
        max_files: Optional[int] = None,
        max_n: int = 3,
        verbose: bool = True
    ) -> Tuple[SparseHRT3D, Dict[int, Dict[int, Dict[int, float]]]]:
        """
        Process all files of this type.
        
        Args:
            root: Root directory to search
            max_files: Limit number of files (None = all)
            max_n: Maximum n-gram size
            verbose: Print progress
        """
        files = list(self.find_files(root))
        if max_files:
            files = files[:max_files]
        
        if verbose:
            print(f"[{self.name}] Processing {len(files)} files")
        
        for path in files:
            try:
                current_hrt, current_W, commit = self.process_file(
                    path, current_hrt, current_W, store, max_n
                )
                if verbose and commit:
                    print(f"  ✓ {path.name} [{current_hrt.nnz} edges]")
            except Exception as e:
                if verbose:
                    print(f"  ✗ {path.name}: {e}")
        
        return current_hrt, current_W


class PromptPerceptron(Perceptron):
    """
    Perceptron for user prompts/queries.
    
    Treats user input exactly like file input:
    - Goes through unified pipeline
    - Gets committed
    - Contributes to manifold (learning from queries!)
    """
    
    def __init__(self, config: Sparse3DConfig):
        super().__init__("p_prompt", [], config)
        self.prompt_history: List[str] = []
    
    def extract_text(self, path: Path) -> str:
        """Not used - prompts come directly as text."""
        return ""
    
    def process_prompt(
        self,
        prompt: str,
        current_hrt: SparseHRT3D,
        current_W: Dict[int, Dict[int, Dict[int, float]]],
        store: CommitStore,
        max_n: int = 3
    ) -> Tuple[SparseHRT3D, Dict[int, Dict[int, Dict[int, float]]], Optional[Commit], Optional[ProcessingResult]]:
        """
        Process a user prompt and commit.
        
        Returns (new_hrt, new_W, commit, processing_result)
        """
        if not self.lut:
            raise RuntimeError("Perceptron not initialized - call initialize(lut) first")
        
        if not prompt.strip():
            return current_hrt, current_W, None, None
        
        self.prompt_history.append(prompt)
        
        # Unified processing (same as files!)
        result = unified_process(
            prompt,
            current_hrt,
            current_W,
            self.config,
            self.lut,
            max_n
        )
        
        new_hrt = result.merged_hrt
        new_W = build_w_from_am(new_hrt.am, self.config)
        
        # Commit with prompt ID as source
        prompt_id = f"prompt_{len(self.prompt_history)}"
        commit = store.commit(new_hrt, new_W, prompt_id, self.name)
        
        self.files_processed += 1
        
        return new_hrt, new_W, commit, result


# ═══════════════════════════════════════════════════════════════════════════
# ACTUATOR BASE CLASS - Act Phase
# ═══════════════════════════════════════════════════════════════════════════

class Actuator(ABC):
    """
    Base class for actuators - turn processed data into action.
    
    Completes the sense-process-act loop:
        Perceptron (sense) → Pipeline (process) → Actuator (act)
    
    Key insight: Actuator output can feed back into the manifold!
    """
    
    def __init__(self, name: str):
        self.name = name
        self.actions_taken = 0
    
    @abstractmethod
    def act(self, commit: Commit, result: ProcessingResult, **kwargs) -> str:
        """
        Perform action based on processed result.
        
        Returns action summary string.
        """
        pass


class ResponseActuator(Actuator):
    """
    Actuator for query responses with FEEDBACK LOOP.
    
    The response itself is ingested back into the manifold!
    This creates co-adaptive learning:
        Query → Response → HLLSet → Commit → (shapes future responses)
    """
    
    def __init__(self):
        super().__init__("a_response")
        self.responses: List[Dict[str, Any]] = []
    
    def act(
        self,
        commit: Commit,
        result: ProcessingResult,
        query_results: List[Tuple[Any, float]] = None,
        hrt: SparseHRT3D = None,
        W: Dict[int, Dict[int, Dict[int, float]]] = None,
        store: CommitStore = None,
        lut: LookupTable = None,
        config: Sparse3DConfig = None,
        ingest_response: bool = True,
        max_n: int = 3,
        **kwargs
    ) -> Tuple[str, SparseHRT3D, Dict[int, Dict[int, Dict[int, float]]]]:
        """
        Generate response and optionally ingest it back.
        
        Returns:
            (response_text, updated_hrt, updated_W)
        """
        import time
        from datetime import datetime
        
        # Build response text
        lines = [
            f"Query: {commit.source}",
            f"Commit: {commit.id[:8]}",
            f"Results ({len(query_results or [])} found):",
        ]
        
        for i, (ntoken, score) in enumerate(query_results or [], 1):
            lines.append(f"  {i:2d}. [{score:5.1f}] {ntoken}")
        
        response_text = "\n".join(lines)
        
        # Track response
        response_record = {
            "timestamp": datetime.fromtimestamp(commit.timestamp).isoformat(),
            "prompt": commit.source,
            "commit_id": commit.id[:8],
            "response": response_text,
            "ingested": False,
        }
        
        new_hrt = hrt
        new_W = W
        
        # FEEDBACK LOOP: Ingest response back into manifold
        if ingest_response and hrt and store and lut and config:
            response_result = unified_process(
                response_text,
                hrt,
                W,
                config,
                lut,
                max_n
            )
            
            new_hrt = response_result.merged_hrt
            new_W = build_w_from_am(new_hrt.am, config)
            
            # Commit response as its own entry
            response_id = f"response_{len(self.responses) + 1}"
            store.commit(new_hrt, new_W, response_id, self.name)
            
            response_record["ingested"] = True
            response_record["response_commit"] = response_id
        
        self.responses.append(response_record)
        self.actions_taken += 1
        
        return response_text, new_hrt, new_W
    
    def history(self) -> List[Dict[str, Any]]:
        """Get response history."""
        return self.responses


# ═══════════════════════════════════════════════════════════════════════════
# QUERY INTERFACE - ask() function
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class QueryContext:
    """
    Mutable context for interactive querying.
    
    Holds the current state that gets updated with each query.
    """
    hrt: SparseHRT3D
    W: Dict[int, Dict[int, Dict[int, float]]]
    config: Sparse3DConfig
    lut: LookupTable
    store: CommitStore
    layer_hllsets: LayerHLLSets
    prompt_perceptron: PromptPerceptron
    response_actuator: ResponseActuator
    max_n: int = 3


def ask(
    prompt: str,
    ctx: QueryContext,
    top_k: int = 10,
    learn: bool = True
) -> Tuple[str, List[DisambiguationResult]]:
    """
    Interactive query with feedback loop.
    
    Full sense-process-act-feedback cycle:
    1. Query → HLLSet → HRT → Commit (SENSE)
    2. Find related concepts (PROCESS)
    3. Disambiguate to tokens (PROCESS)
    4. Generate response (ACT)
    5. Response → HLLSet → HRT → Commit (FEEDBACK!)
    
    The manifold learns from BOTH the question AND its own answer.
    
    Args:
        prompt: User query text
        ctx: QueryContext with current state
        top_k: Number of results to return
        learn: If True, ingest both query AND response
    
    Returns:
        (response_text, disambiguation_results)
    """
    # SENSE: Process query through prompt perceptron
    new_hrt, new_W, commit, result = ctx.prompt_perceptron.process_prompt(
        prompt,
        ctx.hrt,
        ctx.W,
        ctx.store,
        ctx.max_n
    )
    
    if not commit:
        return "No results (empty query)", []
    
    if learn:
        ctx.hrt = new_hrt
        ctx.W = new_W
    
    # PROCESS: Get query indices
    query_indices = set()
    if result:
        for basic in result.input_basics:
            query_indices.add(basic.to_index(ctx.config))
    
    # Find reachable concepts
    AM = Sparse3DMatrix.from_am(ctx.hrt.am, ctx.config)
    layer0 = project_layer(AM, 0)
    reachable = reachable_from(layer0, query_indices, hops=1)
    
    # Score by connectivity
    layer0_dict = layer0.to_dict()
    scores = {}
    for idx in reachable:
        if idx in layer0_dict:
            scores[idx] = sum(layer0_dict[idx].values())
    
    # Get top-k results
    top = sorted(scores.items(), key=lambda x: -x[1])[:top_k]
    query_results = []
    for idx, score in top:
        ntokens = ctx.lut.index_to_ntokens.get(idx, set())
        if ntokens:
            # Get first ntoken
            _, ntoken = next(iter(ntokens))
            query_results.append((ntoken, score))
        else:
            query_results.append((f"<idx:{idx}>", score))
    
    # DISAMBIGUATE: Full cascading disambiguation
    disamb_results = cascading_disambiguate(
        query_indices=query_indices,
        am=ctx.hrt.am,
        layer_hllsets=ctx.layer_hllsets,
        W=ctx.W,
        lut=ctx.lut
    )
    
    # ACT + FEEDBACK: Generate response and ingest it back
    response_text, final_hrt, final_W = ctx.response_actuator.act(
        commit,
        result,
        query_results=query_results,
        hrt=ctx.hrt,
        W=ctx.W,
        store=ctx.store,
        lut=ctx.lut,
        config=ctx.config,
        ingest_response=learn,
        max_n=ctx.max_n
    )
    
    if learn:
        ctx.hrt = final_hrt
        ctx.W = final_W
    
    return response_text, disamb_results


def create_query_context(
    config: Sparse3DConfig,
    lut: Optional[LookupTable] = None
) -> QueryContext:
    """
    Create a new QueryContext for interactive querying.
    
    Initializes all components needed for ask().
    """
    if lut is None:
        lut = LookupTable(config=config)
        lut.add_ntoken(START)
        lut.add_ntoken(END)
    
    # Empty initial structures
    empty_am = SparseAM3D.from_edges(config, [])
    empty_lattice = SparseLattice3D.from_sparse_am(empty_am)
    empty_hrt = SparseHRT3D(
        am=empty_am,
        lattice=empty_lattice,
        config=config,
        lut=frozenset(),
        step=0
    )
    
    empty_W: Dict[int, Dict[int, Dict[int, float]]] = {n: {} for n in range(config.max_n)}
    
    # Components
    store = CommitStore()
    layer_hllsets = LayerHLLSets.empty(config.p_bits)
    
    prompt_perceptron = PromptPerceptron(config)
    prompt_perceptron.initialize(lut)
    
    response_actuator = ResponseActuator()
    
    return QueryContext(
        hrt=empty_hrt,
        W=empty_W,
        config=config,
        lut=lut,
        store=store,
        layer_hllsets=layer_hllsets,
        prompt_perceptron=prompt_perceptron,
        response_actuator=response_actuator,
        max_n=config.max_n
    )


# ═══════════════════════════════════════════════════════════════════════════
# MODULE EXPORTS
# ═══════════════════════════════════════════════════════════════════════════

__all__ = [
    # Identifier Schemes
    'IdentifierScheme',
    'HashIdentifierScheme',
    'VocabularyIdentifierScheme',
    'HashVocabularyScheme',
    'DEFAULT_IDENTIFIER_SCHEME',
    
    # Universal ID
    'UniversalID',
    'content_to_index',
    
    # Sparse Matrices
    'SparseMatrix',
    'Sparse3DMatrix',
    
    # Projection
    'project_layer',
    'project_rows',
    'project_cols',
    'project_submatrix',
    
    # Transform
    'transpose',
    'transpose_3d',
    'normalize_rows',
    'normalize_3d',
    'scale',
    
    # Filter
    'filter_threshold',
    'filter_predicate',
    
    # Composition
    'merge_add',
    'merge_max',
    'compose_chain',
    'merge_3d_add',
    
    # Path
    'reachable_from',
    'path_closure',
    
    # Lift/Lower
    'lift_to_layer',
    'lower_aggregate',
    
    # Cross-structure
    'am_to_w',
    'w_to_am',
    
    # LUT
    'START',
    'END',
    'LookupTable',
    
    # Unified Processing
    'ProcessingResult',
    'tokenize',
    'generate_ntokens',
    'input_to_hllset',
    'build_sub_hrt',
    'extend_with_context',
    'extend_with_intersected_context',
    'merge_hrt',
    'unified_process',
    'build_w_from_am',
    
    # Cascading Disambiguation
    'LayerHLLSets',
    'DisambiguationResult',
    'update_layer_hllsets',
    'cascading_disambiguate',
    'resolve_disambiguation',
    
    # Commit Store
    'Commit',
    'CommitStore',
    
    # Bounded Evolution Store
    'EvictionRecord',
    'DeltaRecord',
    'EvolutionState',
    'StateSnapshot',
    'BoundedEvolutionStore',
    'estimate_cardinality',
    
    # Fingerprint Index (Commit LUT)
    'CommitFingerprint',
    'FingerprintIndex',
    
    # Perceptrons
    'Perceptron',
    'PromptPerceptron',
    
    # Actuators
    'Actuator',
    'ResponseActuator',
    
    # Query Interface
    'QueryContext',
    'ask',
    'create_query_context',
]
