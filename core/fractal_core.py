# core/fractal_core.py
"""
Fractal Core: Primitives for the Self-Similar Lattice Loop

This module implements the fractal loop from the categorical framework:

    T → H → P → R → Lattice → Edges → (tokenize) → T' → H' → P' → ...

The loop creates self-similar structure by treating lattice edges as tokens
for the next iteration, enabling scale-hierarchy analysis.

================================================================================
TWO-PHASE ARCHITECTURE
================================================================================

PHASE 1: Construction (uses BSS with τ, ρ)
- BSS determines which morphisms/edges exist
- Creates the lattice structure
- Implemented in hrt.py, kernel.py

PHASE 2: Analysis (uses overlap measure)
- Measures topology of the built lattice
- Overlap = |A ∩ B| / min(|A|, |B|)
- This module implements Phase 2

================================================================================
FRACTAL LOOP COMPONENTS
================================================================================

1. N-Tokenization: Create sliding window tokens from sequences
   - Input: token sequence, window size n
   - Output: n-grams as new token set

2. Overlap Measure: Topology analysis after lattice construction
   - Input: two HLLSets (or cardinalities)
   - Output: overlap ratio in [0, 1]

3. Edge Tokenization: Convert lattice edges to tokens
   - Input: lattice with edges (chains of projections)
   - Output: edge-tokens for next iteration

4. Entanglement Number: Cross-scale correlation
   - Input: edge projections at scales m and n
   - Output: E(m,n) = overlap of edge projection sets

5. Scale Hierarchy: Multi-scale lattice tower
   - Level 0: Raw tokens T
   - Level 1: Projections P
   - Level 2: Lattice L
   - Level 3: Edge projections P_edge^(n)
   - Level 4+: Recursive application

Design Principles (IICA):
- Immutable: All functions return new objects
- Idempotent: Same input → same output
- Content-addressed: Results named by SHA1
- Backward compatible: Existing code unaffected
"""

from __future__ import annotations
from typing import List, Tuple, Set, Dict, Optional, Iterable, FrozenSet
from dataclasses import dataclass, field
import hashlib

from .hllset import HLLSet, compute_sha1
from .constants import (
    P_BITS, SHARED_SEED,
    DEFAULT_N_TOKEN_SIZES, MIN_OVERLAP, MAX_CHAIN_LENGTH,
    ENTANGLEMENT_THRESHOLD
)


# =============================================================================
# SECTION 1: N-Tokenization (Scale Hierarchy Generator)
# =============================================================================

def n_tokenize(tokens: List[str], n: int) -> List[str]:
    """
    Create n-grams from a token sequence using sliding window.
    
    This is the fundamental scale-generation operation:
    - n=1: Original tokens (identity)
    - n=2: Bigrams (pairs of consecutive tokens)
    - n=3: Trigrams, etc.
    
    Args:
        tokens: Ordered sequence of tokens
        n: Window size (n-gram size)
        
    Returns:
        List of n-gram strings (joined with separator)
        
    Example:
        >>> n_tokenize(['a', 'b', 'c', 'd'], 2)
        ['a|b', 'b|c', 'c|d']
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if n > len(tokens):
        return []
    
    # Use pipe separator (unlikely in typical tokens)
    separator = "|"
    return [separator.join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]


def multi_scale_tokenize(tokens: List[str], 
                         scales: Tuple[int, ...] = DEFAULT_N_TOKEN_SIZES
                         ) -> Dict[int, List[str]]:
    """
    Generate tokens at multiple scales simultaneously.
    
    Creates the token hierarchy for fractal analysis:
    {1: [original tokens], 2: [bigrams], 3: [trigrams], ...}
    
    Args:
        tokens: Ordered sequence of tokens
        scales: Tuple of window sizes to generate
        
    Returns:
        Dict mapping scale n to list of n-grams
    """
    return {n: n_tokenize(tokens, n) for n in scales}


def tokens_to_hllsets(token_dict: Dict[int, List[str]], 
                      p_bits: int = P_BITS,
                      seed: int = SHARED_SEED) -> Dict[int, HLLSet]:
    """
    Convert multi-scale tokens to HLLSets.
    
    Args:
        token_dict: Dict from scale n to token list
        p_bits: HLL precision
        seed: Hash seed
        
    Returns:
        Dict mapping scale n to HLLSet
    """
    return {
        n: HLLSet.from_batch(tokens, p_bits=p_bits, seed=seed)
        for n, tokens in token_dict.items()
        if tokens  # Skip empty token lists
    }


# =============================================================================
# SECTION 2: Overlap Measure (Lattice Topology Analysis)
# =============================================================================

def overlap(a: HLLSet, b: HLLSet) -> float:
    """
    Compute overlap measure for lattice topology analysis.
    
    overlap(A, B) = |A ∩ B| / min(|A|, |B|)
    
    This is used AFTER BSS has constructed the lattice.
    - BSS(τ, ρ) determines edge existence (construction phase)
    - overlap() measures structure similarity (analysis phase)
    
    Properties:
    - overlap ∈ [0, 1]
    - overlap(A, A) = 1
    - overlap(A, B) = overlap(B, A) (symmetric)
    - overlap = 1 iff smaller set is contained in larger
    
    Args:
        a: First HLLSet
        b: Second HLLSet
        
    Returns:
        Overlap ratio in [0, 1]
    """
    card_a = a.cardinality()
    card_b = b.cardinality()
    
    if card_a == 0 or card_b == 0:
        return 0.0
    
    intersection = a.intersect(b)
    card_intersection = intersection.cardinality()
    
    return card_intersection / min(card_a, card_b)


def overlap_from_cardinalities(card_a: float, card_b: float, 
                                card_intersection: float) -> float:
    """
    Compute overlap from pre-computed cardinalities.
    
    Useful when cardinalities are already known (avoids recomputation).
    
    Args:
        card_a: Cardinality of set A
        card_b: Cardinality of set B  
        card_intersection: Cardinality of A ∩ B
        
    Returns:
        Overlap ratio in [0, 1]
    """
    if card_a == 0 or card_b == 0:
        return 0.0
    return card_intersection / min(card_a, card_b)


# =============================================================================
# SECTION 3: Edge Tokenization (Lattice → Tokens)
# =============================================================================

@dataclass(frozen=True)
class LatticeEdge:
    """
    An edge in the lattice (a chain of projections).
    
    Represents a path P_1 → P_2 → ... → P_n in the lattice.
    The chain length determines the "scale" of this edge.
    """
    chain: Tuple[str, ...]  # Tuple of projection hashes
    
    @property
    def length(self) -> int:
        """Chain length (1 = single projection, 2 = edge, 3+ = path)."""
        return len(self.chain)
    
    @property
    def token(self) -> str:
        """
        Convert edge to token for next fractal iteration.
        
        This is the key operation: treating lattice structure as tokens.
        """
        return "|".join(self.chain)
    
    @property
    def hash(self) -> str:
        """Content-addressed hash of the edge."""
        return compute_sha1(self.token)
    
    def __repr__(self) -> str:
        short_chain = [h[:8] for h in self.chain]
        return f"Edge({' → '.join(short_chain)})"


def extract_chains(lattice_edges: List[Tuple[str, str]], 
                   max_length: int = MAX_CHAIN_LENGTH
                   ) -> Dict[int, List[LatticeEdge]]:
    """
    Extract chains of various lengths from lattice edges.
    
    Given edges [(a,b), (b,c), (c,d), ...], builds:
    - Length 1: [a], [b], [c], [d] (vertices as trivial chains)
    - Length 2: [a,b], [b,c], [c,d] (original edges)
    - Length 3: [a,b,c], [b,c,d] (paths of length 2)
    - etc.
    
    This creates the n-edge hierarchy for the fractal loop.
    
    Args:
        lattice_edges: List of (source_hash, target_hash) pairs
        max_length: Maximum chain length to extract
        
    Returns:
        Dict mapping length to list of LatticeEdge objects
    """
    # Build adjacency for path finding
    adjacency: Dict[str, Set[str]] = {}
    vertices: Set[str] = set()
    
    for src, tgt in lattice_edges:
        vertices.add(src)
        vertices.add(tgt)
        if src not in adjacency:
            adjacency[src] = set()
        adjacency[src].add(tgt)
    
    result: Dict[int, List[LatticeEdge]] = {}
    
    # Length 1: vertices as trivial chains
    result[1] = [LatticeEdge((v,)) for v in sorted(vertices)]
    
    # Length 2: original edges
    result[2] = [LatticeEdge((src, tgt)) for src, tgt in lattice_edges]
    
    # Length 3+: extend paths
    for length in range(3, max_length + 1):
        prev_chains = result.get(length - 1, [])
        new_chains = []
        
        for chain in prev_chains:
            last = chain.chain[-1]
            if last in adjacency:
                for next_vertex in adjacency[last]:
                    new_chain = chain.chain + (next_vertex,)
                    new_chains.append(LatticeEdge(new_chain))
        
        if new_chains:
            result[length] = new_chains
        else:
            break  # No more paths to extend
    
    return result


def chains_to_tokens(chains: Dict[int, List[LatticeEdge]]) -> Dict[int, List[str]]:
    """
    Convert lattice chains to tokens for the next fractal iteration.
    
    This is the key "closing the loop" operation:
    Lattice structure → Token set → (hash, project) → New Lattice
    
    Args:
        chains: Dict mapping length to LatticeEdge lists
        
    Returns:
        Dict mapping length to token strings
    """
    return {
        length: [edge.token for edge in edges]
        for length, edges in chains.items()
    }


# =============================================================================
# SECTION 4: Entanglement Number (Cross-Scale Correlation)
# =============================================================================

def entanglement_number(hll_m: HLLSet, hll_n: HLLSet) -> float:
    """
    Compute entanglement number between two scales.
    
    E(m, n) = |P_edge^(m) ∩ P_edge^(n)| / min(|P_edge^(m)|, |P_edge^(n)|)
    
    When E(m,n) > 0, scales m and n are "entangled" - they share
    common edge projections despite being at different scales.
    
    This is the key observable for fractal structure detection.
    
    Args:
        hll_m: HLLSet of edge projections at scale m
        hll_n: HLLSet of edge projections at scale n
        
    Returns:
        Entanglement number in [0, 1]
    """
    return overlap(hll_m, hll_n)


def is_entangled(hll_m: HLLSet, hll_n: HLLSet, 
                 threshold: float = ENTANGLEMENT_THRESHOLD) -> bool:
    """
    Check if two scales are entangled.
    
    Args:
        hll_m: HLLSet of edge projections at scale m
        hll_n: HLLSet of edge projections at scale n
        threshold: Minimum entanglement for True
        
    Returns:
        True if E(m,n) > threshold
    """
    return entanglement_number(hll_m, hll_n) > threshold


def entanglement_matrix(scale_hllsets: Dict[int, HLLSet]) -> Dict[Tuple[int, int], float]:
    """
    Compute entanglement matrix for all scale pairs.
    
    Args:
        scale_hllsets: Dict mapping scale n to HLLSet of n-edge projections
        
    Returns:
        Dict mapping (m, n) pairs to entanglement numbers
    """
    scales = sorted(scale_hllsets.keys())
    result = {}
    
    for i, m in enumerate(scales):
        for n in scales[i:]:  # Upper triangle (symmetric)
            e_mn = entanglement_number(scale_hllsets[m], scale_hllsets[n])
            result[(m, n)] = e_mn
            if m != n:
                result[(n, m)] = e_mn  # Symmetric
    
    return result


# =============================================================================
# SECTION 5: Scale Hierarchy (Fractal Tower)
# =============================================================================

@dataclass
class ScaleLevel:
    """
    One level in the fractal scale hierarchy.
    
    Captures the state at one iteration of the fractal loop:
    - Scale index (0 = raw tokens, 1+ = derived)
    - N-gram size used to generate this level
    - HLLSet of projections at this level
    - Source (what produced this level)
    """
    level: int
    n_gram_size: int
    hllset: HLLSet
    token_count: int
    source: str  # 'tokens' or 'edges'
    
    @property
    def name(self) -> str:
        """Content-addressed name."""
        return self.hllset.name
    
    @property
    def cardinality(self) -> float:
        """Estimated distinct elements."""
        return self.hllset.cardinality()
    
    def __repr__(self) -> str:
        return (f"ScaleLevel(L{self.level}, {self.n_gram_size}-gram, "
                f"card={self.cardinality:.0f}, src={self.source})")


@dataclass
class FractalTower:
    """
    The complete fractal scale hierarchy.
    
    A tower of ScaleLevels built by iterating the fractal loop:
    - Level 0: Raw tokens
    - Level 1: N-grams of raw tokens
    - Level 2: Edges from Level 1 lattice, tokenized
    - Level 3+: Recursive application
    
    The tower captures the self-similar structure across scales.
    """
    levels: List[ScaleLevel] = field(default_factory=list)
    entanglement: Dict[Tuple[int, int], float] = field(default_factory=dict)
    
    def add_level(self, level: ScaleLevel):
        """Add a new level to the tower."""
        self.levels.append(level)
    
    @property
    def depth(self) -> int:
        """Number of levels in the tower."""
        return len(self.levels)
    
    def get_level(self, idx: int) -> Optional[ScaleLevel]:
        """Get level by index."""
        if 0 <= idx < len(self.levels):
            return self.levels[idx]
        return None
    
    def compute_entanglement(self):
        """Compute entanglement matrix for all levels."""
        hllsets = {level.level: level.hllset for level in self.levels}
        self.entanglement = entanglement_matrix(hllsets)
    
    def __repr__(self) -> str:
        return f"FractalTower(depth={self.depth}, levels={self.levels})"


def build_token_tower(tokens: List[str],
                      scales: Tuple[int, ...] = DEFAULT_N_TOKEN_SIZES,
                      p_bits: int = P_BITS,
                      seed: int = SHARED_SEED) -> FractalTower:
    """
    Build the first iteration of the fractal tower from tokens.
    
    This creates Level 0 (raw tokens) and Level 1 (n-grams).
    Further levels require lattice construction and edge extraction.
    
    Args:
        tokens: Raw token sequence
        scales: N-gram sizes to generate
        p_bits: HLL precision
        seed: Hash seed
        
    Returns:
        FractalTower with token-derived levels
    """
    tower = FractalTower()
    
    # Level 0: Raw tokens (1-grams)
    hll_raw = HLLSet.from_batch(tokens, p_bits=p_bits, seed=seed)
    tower.add_level(ScaleLevel(
        level=0, n_gram_size=1, hllset=hll_raw,
        token_count=len(tokens), source='tokens'
    ))
    
    # Higher scales from n-tokenization
    for n in scales:
        if n == 1:
            continue  # Already added as Level 0
        n_tokens = n_tokenize(tokens, n)
        if n_tokens:
            hll_n = HLLSet.from_batch(n_tokens, p_bits=p_bits, seed=seed)
            tower.add_level(ScaleLevel(
                level=len(tower.levels), n_gram_size=n,
                hllset=hll_n, token_count=len(n_tokens), source='tokens'
            ))
    
    # Compute initial entanglement
    tower.compute_entanglement()
    
    return tower
