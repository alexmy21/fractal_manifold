"""
Cascading Disambiguation

Layer-specific HLLSets for cascading disambiguation and token recovery.

Architecture Note:
    Uses Kernel for HLLSet operations to ensure consistent p_bits and SHARED_SEED.
    HLLSet is imported for type annotations only.
"""

from __future__ import annotations
from typing import Dict, List, Set, Optional
from dataclasses import dataclass

# Type annotation (no runtime behavior)
from ..hllset import HLLSet
# Operations go through kernel for consistent configuration
from ..kernel import Kernel
from ..sparse_hrt_3d import SparseAM3D

from .lut import LookupTable

# Module-level kernel instance for operations
_kernel = Kernel()


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
        """Create empty layer HLLSets via kernel."""
        kernel = Kernel(p_bits=p_bits)
        empty_hll = kernel.absorb(set())  # Empty HLLSet
        return cls(
            L0=empty_hll,
            L1=kernel.absorb(set()),
            L2=kernel.absorb(set()),
            START=kernel.absorb(set()),
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
        """Add index to appropriate layer HLLSet via kernel."""
        # Convert idx to string token for HLLSet API
        token = str(idx)
        if layer == 0:
            self.L0 = _kernel.add(self.L0, token)
        elif layer == 1:
            self.L1 = _kernel.add(self.L1, token)
        elif layer == 2:
            self.L2 = _kernel.add(self.L2, token)
    
    def mark_start(self, idx: int):
        """Mark index as START follower via kernel."""
        token = str(idx)
        self.START = _kernel.add(self.START, token)
    
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
    layer_sets: LayerHLLSets,
    edges: List,  # List[Edge3D] but avoiding circular import
    config,  # Sparse3DConfig
) -> LayerHLLSets:
    """
    Update LayerHLLSets from new edges.
    
    Returns new merged LayerHLLSets.
    """
    for edge in edges:
        layer_sets.add_to_layer(edge.n, edge.row)
        layer_sets.add_to_layer(edge.n, edge.col)
    return layer_sets


def cascading_disambiguate(
    query_hll: HLLSet,
    layer_sets: LayerHLLSets,
    W: Dict[int, Dict[int, Dict[int, float]]],
    am: SparseAM3D
) -> List[DisambiguationResult]:
    """
    Cascading disambiguation via START → 2gram → 3gram transitions.
    
    STEP 1: Identify candidate indices in each layer
    STEP 2: Find START tokens
    STEP 3: Follow W transitions to build sequences
    STEP 4: Process remaining higher-order n-grams
    STEP 5: Process standalone 1-grams
    """
    results = []
    processed: Set[int] = set()
    
    # STEP 1: Get indices from each layer that intersect with query
    H_0 = set()
    H_1 = set()
    H_2 = set()
    
    # TODO: This is a placeholder - actual implementation needs index extraction
    # from HLLSet intersection, which requires bitmap operations
    
    remaining = H_0 | H_1 | H_2
    
    # STEP 2: Find START candidates
    start_candidates = set()
    # START candidates are 1-grams that follow START in W
    
    if 0 in W:
        for row in W[0]:
            if row in H_0:
                start_candidates.add(row)
    
    # STEP 3: Follow W transitions from START candidates
    for start_token in start_candidates:
        if start_token in processed:
            continue
        
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


__all__ = [
    'LayerHLLSets',
    'DisambiguationResult',
    'update_layer_hllsets',
    'cascading_disambiguate',
    'resolve_disambiguation',
]
