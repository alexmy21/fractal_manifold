"""
Identifier Schemes - Pluggable Content → Index Mapping

Different languages/sign systems use different identification schemes:

- HashScheme (default): content → hash → (reg, zeros) → index
  For inflected languages where vocabulary is open-ended.
  
- VocabularyScheme: sign → vocabulary lookup → index
  For uninflected languages with fixed sign systems (Chinese, etc.)
  where each sign is unambiguous and unmodifiable.

The AM and W structures don't care HOW indices are computed,
only that they're consistent within a system.
"""

from __future__ import annotations
from typing import (
    Dict, Set, Optional, Tuple, Iterable, Any, Protocol, runtime_checkable
)
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np

from ..hllset import (
    HLLSet, 
    HashConfig,
    DEFAULT_HASH_CONFIG,
    P_BITS,
)

# Default hash bits (from HLLSet's centralized config)
DEFAULT_H_BITS = DEFAULT_HASH_CONFIG.h_bits


# Helper function for HLLSet cardinality estimation
def estimate_cardinality(registers: np.ndarray) -> float:
    """
    Estimate cardinality from HLL registers using Linear Counting.
    
    This is a simplified estimator for small cardinalities.
    For full HLL accuracy, use HLLSet.cardinality().
    """
    n_registers = len(registers)
    zeros = np.sum(registers == 0)
    
    if zeros == 0:
        # All registers filled - use harmonic mean
        # This is simplified; full HLL uses more sophisticated estimation
        return n_registers * 32  # Rough upper bound
    
    # Linear Counting: n ≈ -m * ln(V/m) where V = zeros
    return -n_registers * np.log(zeros / n_registers)


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
    
    def signs_at_his_index(self, his_index: int) -> list[str]:
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
        am_edges: list[Tuple[int, int, int, float]]
    ) -> list[Tuple[int, int, int, float]]:
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
        index_counts: Dict[int, list[str]] = defaultdict(list)
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
        return int(estimate_cardinality(registers))
    
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


__all__ = [
    'IdentifierScheme',
    'HashIdentifierScheme',
    'VocabularyIdentifierScheme',
    'HashVocabularyScheme',
    'DEFAULT_IDENTIFIER_SCHEME',
    'DEFAULT_H_BITS',
    'estimate_cardinality',
]
