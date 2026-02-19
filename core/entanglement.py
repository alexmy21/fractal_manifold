# File: core/entanglement.py
"""
Entanglement for Manifold OS - Lattice-Based Implementation

================================================================================
IMPORTANT: HLLSets are NOT sets containing tokens!
================================================================================

HLLSets are probabilistic register structures ("anti-sets") that:
- ABSORB tokens (hash them into registers)
- DO NOT STORE tokens (only register states remain)
- BEHAVE LIKE sets (union, intersection, cardinality estimation)
- ARE NOT sets (no element retrieval, no membership test)

================================================================================
TWO-LAYER ARCHITECTURE: This module operates at the LATTICE LAYER
================================================================================

┌─────────────────────────────────────────────────────────────────────────────┐
│ ENTANGLEMENT IS BETWEEN LATTICES, NOT HLLSets!                              │
│                                                                             │
│ Layer 1 (HLLSet/Register): Individual register arrays, cardinality est.     │
│ Layer 2 (Lattice/Structure): Graph topology, degree distributions ← HERE    │
│                                                                             │
│ This module implements Layer 2 - TRUE ENTANGLEMENT                          │
│ Individual HLLSets are IRRELEVANT - only STRUCTURE matters                  │
│ Two lattices can be entangled even from completely different inputs         │
└─────────────────────────────────────────────────────────────────────────────┘

Based on "The Trinity of Emergence" from dao_manifold.pdf:
- Entanglement as Measurement: M(L1, L2) = structural similarity of LATTICES
- Lattice W is prior, entanglement is map W(1) → W(2)
- Endomorphism W → W with finite lattices
- Nodes connected by similar degrees (cardinality/topology)
- Maximizes structural pairs between lattices

Key Principles:
- Consistency prior to precision
- Lattice W guarantees finiteness
- Entanglement morphism connects similar-DEGREE nodes (not similar registers)
- Maximizes matching pairs between lattice STRUCTURES

Fundamental Consistency Criterion (Categorical):
    For any a ≠ b in L₁, the mapping m(a) ≉ m(b) in L₂
    
    This preserves distinctness (approximate injectivity).
    Due to hash collisions, we use approximate measures (≉) instead of strict equality.
    
    Structural measures used:
    - Degree similarity (estimated cardinality-based)
    - Graph topology correlation
    - Register pattern overlap (secondary, for refinement)
    
    The core requirement: mapping should preserve STRUCTURAL TOPOLOGY
    by maintaining degree patterns between lattices.
"""

from __future__ import annotations
from typing import List, Dict, Set, Tuple, Optional, Callable
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np

from .kernel import HLLSet, Kernel
from .hrt import HLLSetLattice, BasicHLLSet, HRT


# =============================================================================
# SECTION 1: Entanglement as Measurement
# =============================================================================

@dataclass(frozen=True)
class EntanglementMeasurement:
    """
    Measurement of entanglement between two basic HLLSets.
    
    M(w1, w2) = structural similarity based on:
    - Degree similarity (cardinality match)
    - Register overlap (structural similarity)
    
    This is the M: S × S → R function from the paper.
    
    IMPLEMENTATION NOTE:
    This uses degree + register similarity as the measure.
    This is ONE valid implementation of the consistency criterion:
        ∀ a≠b in L₁: m(a) ≉ m(b) in L₂
    
    Alternative measures could be used (custom metrics, domain-specific
    similarity functions, etc.) as long as they preserve distinctness
    and structural topology.
    """
    source_hash: str          # Hash of source basic HLLSet
    target_hash: str          # Hash of target basic HLLSet
    
    # Measurements (implementation-specific)
    degree_similarity: float  # Based on cardinality match
    register_similarity: float  # Based on HLLSet similarity
    
    @property
    def strength(self) -> float:
        """
        Overall entanglement strength.
        Weighted combination of degree and register similarity.
        Consistency (degree) weighted more than precision (registers).
        """
        # Consistency (degree) > Precision (registers)
        return 0.6 * self.degree_similarity + 0.4 * self.register_similarity
    
    def __repr__(self) -> str:
        return f"EM({self.source_hash[:8]}↔{self.target_hash[:8]}, s={self.strength:.3f})"


# =============================================================================
# SECTION 2: Entanglement Morphism (Lattice Map)
# =============================================================================

@dataclass(frozen=True)
class EntanglementMorphism:
    """
    Entanglement morphism: map between two lattices W(1) → W(2).
    
    Unlike traditional morphisms (functions), this is a relational map:
    - Each node in W(1) can map to multiple nodes in W(2)
    - Mapping based on similarity measures that preserve structure
    - Maximizes total matching pairs
    
    FUNDAMENTAL PROPERTY (Categorical Consistency):
        ∀ a≠b in L₁: m(a) ≉ m(b) in L₂
        
    This ensures the mapping preserves distinctness (approximate injectivity).
    The specific similarity measure (BSS, degree, custom) is implementation-dependent,
    but all must satisfy this consistency criterion.
    
    This implements the paper's insight:
    "Lattice would be prior and we will implement entanglement morphism 
    as map from one lattice to another."
    
    Mutual exclusivity of HLLSets is NOT required—entanglement is about
    preserving structural topology, not enforcing set disjointness.
    """
    source_lattice_hash: str
    target_lattice_hash: str
    
    # Mapping: source_index → list of (target_index, measurement)
    mapping: Dict[int, List[Tuple[int, EntanglementMeasurement]]] = field(default_factory=dict)
    
    # Statistics
    total_pairs: int = 0
    max_degree_diff: float = 0.0
    
    @property
    def name(self) -> str:
        """Content-addressed name."""
        from .kernel import compute_sha1
        data = f"{self.source_lattice_hash}:{self.target_lattice_hash}:{self.total_pairs}"
        return compute_sha1(data)
    
    def get_targets(self, source_idx: int) -> List[Tuple[int, float]]:
        """Get target indices and strengths for source index."""
        if source_idx not in self.mapping:
            return []
        return [(t, m.strength) for t, m in self.mapping[source_idx]]
    
    def get_max_target(self, source_idx: int) -> Optional[Tuple[int, float]]:
        """Get strongest entangled target for source index."""
        targets = self.get_targets(source_idx)
        if not targets:
            return None
        return max(targets, key=lambda x: x[1])
    
    def __repr__(self) -> str:
        return f"Morph({self.source_lattice_hash[:8]}→{self.target_lattice_hash[:8]}, {self.total_pairs} pairs)"


# =============================================================================
# SECTION 3: Entanglement Engine
# =============================================================================

class EntanglementEngine:
    """
    Engine for computing entanglement between lattices.
    
    Implements the degree-based matching algorithm:
    1. Extract degree (cardinality) of each node in both lattices
    2. Sort nodes by degree
    3. Match nodes with similar degrees
    4. Maximize total number of valid pairs
    
    This is the W(1) → W(2) map from the paper.
    """
    
    def __init__(self, kernel: Kernel):
        self.kernel = kernel
        self.degree_tolerance: float = 0.2  # 20% tolerance for degree matching
        self.min_similarity: float = 0.1    # Minimum similarity threshold
    
    def extract_degrees(self, lattice: HLLSetLattice) -> Dict[int, float]:
        """
        Extract degree (cardinality) of each basic HLLSet in lattice.
        
        Returns: {index -> cardinality}
        """
        degrees = {}
        
        # Row basic HLLSets
        for basic in lattice.row_basic:
            degrees[basic.index] = basic.hllset.cardinality()
        
        # Column basic HLLSets (offset by dimension to distinguish)
        dim = lattice.config.dimension
        for basic in lattice.col_basic:
            degrees[basic.index + dim] = basic.hllset.cardinality()
        
        return degrees
    
    def compute_degree_similarity(self, deg1: float, deg2: float) -> float:
        """
        Compute similarity between two degrees.
        
        Uses harmonic mean approach:
        - 1.0 if degrees are identical
        - Decreases as difference increases
        - Tolerance controlled by degree_tolerance
        """
        if deg1 == 0 and deg2 == 0:
            return 1.0
        if deg1 == 0 or deg2 == 0:
            return 0.0
        
        # Relative difference
        max_deg = max(deg1, deg2)
        min_deg = min(deg1, deg2)
        
        if max_deg == 0:
            return 1.0
        
        ratio = min_deg / max_deg
        
        # Similarity decreases as ratio deviates from 1
        if ratio >= (1 - self.degree_tolerance):
            return ratio
        return 0.0
    
    def compute_register_similarity(self, basic1: BasicHLLSet, basic2: BasicHLLSet) -> float:
        """Compute HLLSet structural similarity."""
        return basic1.hllset.similarity(basic2.hllset)
    
    def compute_entanglement(self, 
                            source_lattice: HLLSetLattice,
                            target_lattice: HLLSetLattice,
                            source_hash: str,
                            target_hash: str) -> EntanglementMorphism:
        """
        Compute entanglement morphism between two lattices.
        
        Algorithm:
        1. Extract degrees from both lattices
        2. For each source node, find target nodes with similar degree
        3. Compute full measurement (degree + register similarity)
        4. Maximize pairs by selecting strongest matches
        """
        morphism = EntanglementMorphism(
            source_lattice_hash=source_hash,
            target_lattice_hash=target_hash
        )
        
        dim = source_lattice.config.dimension
        
        # Collect all basic HLLSets with their global indices
        source_nodes = []
        target_nodes = []
        
        # Source nodes
        for basic in source_lattice.row_basic:
            source_nodes.append((basic.index, basic, 'row'))
        for basic in source_lattice.col_basic:
            source_nodes.append((basic.index + dim, basic, 'col'))
        
        # Target nodes
        for basic in target_lattice.row_basic:
            target_nodes.append((basic.index, basic, 'row'))
        for basic in target_lattice.col_basic:
            target_nodes.append((basic.index + dim, basic, 'col'))
        
        # Compute pairwise entanglements
        total_pairs = 0
        max_degree_diff = 0.0
        
        for src_idx, src_basic, src_type in source_nodes:
            src_card = src_basic.hllset.cardinality()
            
            matches = []
            
            for tgt_idx, tgt_basic, tgt_type in target_nodes:
                tgt_card = tgt_basic.hllset.cardinality()
                
                # Compute degree similarity
                deg_sim = self.compute_degree_similarity(src_card, tgt_card)
                
                if deg_sim > 0:  # Within tolerance
                    # Compute register similarity
                    reg_sim = self.compute_register_similarity(src_basic, tgt_basic)
                    
                    # Create measurement
                    measurement = EntanglementMeasurement(
                        source_hash=src_basic.hllset.name,
                        target_hash=tgt_basic.hllset.name,
                        degree_similarity=deg_sim,
                        register_similarity=reg_sim
                    )
                    
                    if measurement.strength >= self.min_similarity:
                        matches.append((tgt_idx % dim, measurement))  # Use local index
                        total_pairs += 1
                        
                        # Track max degree difference
                        degree_diff = abs(src_card - tgt_card)
                        max_degree_diff = max(max_degree_diff, degree_diff)
            
            # Sort by strength and keep top matches
            matches.sort(key=lambda x: x[1].strength, reverse=True)
            if matches:
                morphism.mapping[src_idx % dim] = matches[:5]  # Keep top 5
        
        # Update statistics
        object.__setattr__(morphism, 'total_pairs', total_pairs)
        object.__setattr__(morphism, 'max_degree_diff', max_degree_diff)
        
        return morphism
    
    def compute_self_entanglement(self, lattice: HLLSetLattice, lattice_hash: str) -> EntanglementMorphism:
        """
        Compute endomorphism W → W (self-entanglement).
        
        This maps each node to itself and structurally similar nodes
        within the same lattice.
        """
        return self.compute_entanglement(lattice, lattice, lattice_hash, lattice_hash)
    
    def find_maximum_matching(self, 
                             morphism: EntanglementMorphism,
                             one_to_one: bool = True) -> Dict[int, int]:
        """
        Find maximum matching from entanglement mapping.
        
        If one_to_one=True, each target matched to at most one source.
        Returns: {source_idx -> target_idx}
        """
        if not one_to_one:
            # Just take strongest for each source
            return {s: self.get_max_target(s)[0] 
                   for s in morphism.mapping.keys()}
        
        # Greedy matching: sort all pairs by strength, assign greedily
        all_pairs = []
        for src_idx, matches in morphism.mapping.items():
            for tgt_idx, measurement in matches:
                all_pairs.append((measurement.strength, src_idx, tgt_idx))
        
        # Sort by strength (descending)
        all_pairs.sort(reverse=True)
        
        # Greedy assignment
        used_targets = set()
        matching = {}
        
        for strength, src_idx, tgt_idx in all_pairs:
            if tgt_idx not in used_targets and src_idx not in matching:
                matching[src_idx] = tgt_idx
                used_targets.add(tgt_idx)
        
        return matching


# =============================================================================
# SECTION 4: Entanglement-Preserving Operations
# =============================================================================

class EntanglementPreservingOps:
    """
    Operations that preserve entanglement structure.
    
    Based on paper's compatibility condition:
    M(Φ(s), Φ(s')) = M(s, s')
    
    Operations are idempotent and entanglement-preserving.
    """
    
    def __init__(self, engine: EntanglementEngine):
        self.engine = engine
    
    def merge_lattices_with_entanglement(self,
                                        lattice1: HLLSetLattice,
                                        lattice2: HLLSetLattice,
                                        morphism: EntanglementMorphism,
                                        kernel: Kernel) -> HLLSetLattice:
        """
        Merge two lattices preserving entanglement structure.
        
        When merging lattices W(1) and W(2):
        1. Union corresponding basic HLLSets (via entanglement mapping)
        2. Preserve measurement relationships
        3. Result maintains entanglement invariants
        """
        from .hrt import HRTConfig
        
        # Create new lattice
        config = lattice1.config
        merged = HLLSetLattice(config=config)
        
        # Get maximum matching
        matching = self.engine.find_maximum_matching(morphism, one_to_one=True)
        
        # Merge matched pairs
        for src_idx, tgt_idx in matching.items():
            basic1 = lattice1.row_basic[src_idx] if src_idx < len(lattice1.row_basic) else None
            basic2 = lattice2.row_basic[tgt_idx] if tgt_idx < len(lattice2.row_basic) else None
            
            if basic1 and basic2:
                # Union the HLLSets
                merged_hll = kernel.union(basic1.hllset, basic2.hllset)
                # Update in merged lattice
                if src_idx < len(merged.row_basic):
                    merged.row_basic[src_idx] = BasicHLLSet(
                        index=src_idx,
                        is_row=True,
                        hllset=merged_hll,
                        config=config
                    )
        
        return merged


# =============================================================================
# SECTION 5: Common Subgraph Extraction (Entanglement as Structure)
# =============================================================================

@dataclass
class EntanglementFragment:
    """
    A single matching fragment (disconnected component) of entanglement.
    
    Represents a piece of structural correspondence between two lattices.
    Multiple fragments together form the full entanglement.
    """
    # Node mapping within this fragment: L1 index → L2 index
    node_mapping: Dict[int, int] = field(default_factory=dict)
    
    # Matched edges within this fragment
    matched_edges: List[Tuple[Tuple[int, int], Tuple[int, int]]] = field(default_factory=list)
    
    # START/END markers in this fragment (L1 indices)
    start_nodes: Set[int] = field(default_factory=set)
    end_nodes: Set[int] = field(default_factory=set)
    
    @property
    def size(self) -> int:
        return len(self.node_mapping)
    
    @property
    def edge_count(self) -> int:
        return len(self.matched_edges)
    
    def __repr__(self) -> str:
        markers = ""
        if self.start_nodes:
            markers += f", starts={self.start_nodes}"
        if self.end_nodes:
            markers += f", ends={self.end_nodes}"
        return f"Fragment(nodes={self.size}, edges={self.edge_count}{markers})"


@dataclass
class EntanglementSubgraph:
    """
    Collection of matching fragments representing entanglement between two lattices.
    
    Entanglement = Collection of Matching Subgraphs (possibly disconnected)
    
    Key insight: Entanglement doesn't need to be a single connected subgraph.
    Multiple disconnected matching regions are valid "entanglement seeds" that
    can be extended within each lattice.
    
    Contains:
    - fragments: list of disconnected matching regions
    - matched_nodes: aggregate mapping from L1 node → L2 node
    - matched_edges: aggregate edges across all fragments
    - start_nodes: nodes marked as START in any fragment
    - end_nodes: nodes marked as END in any fragment
    
    Extension: Use extend_in_lattice() to find nodes connected to fragments
    within each lattice (closure operation).
    """
    source_lattice_hash: str
    target_lattice_hash: str
    
    # Collection of disconnected matching fragments
    fragments: List[EntanglementFragment] = field(default_factory=list)
    
    # Aggregate node mapping: L1 index → L2 index (union of all fragments)
    matched_nodes: Dict[int, int] = field(default_factory=dict)
    
    # Aggregate matched edges (union of all fragments)
    matched_edges: List[Tuple[Tuple[int, int], Tuple[int, int]]] = field(default_factory=list)
    
    # START/END markers (union across fragments)
    start_nodes: Set[int] = field(default_factory=set)
    end_nodes: Set[int] = field(default_factory=set)
    
    # Statistics
    node_coverage: float = 0.0  # fraction of nodes matched
    edge_coverage: float = 0.0  # fraction of edges matched
    structural_similarity: float = 0.0  # overall similarity score
    
    @property
    def size(self) -> int:
        """Total number of matched nodes across all fragments."""
        return len(self.matched_nodes)
    
    @property
    def edge_count(self) -> int:
        """Total number of matched edges across all fragments."""
        return len(self.matched_edges)
    
    @property
    def fragment_count(self) -> int:
        """Number of disconnected matching fragments."""
        return len(self.fragments)
    
    @property
    def name(self) -> str:
        """Content-addressed name."""
        from .kernel import compute_sha1
        data = f"{self.source_lattice_hash}:{self.target_lattice_hash}:{self.size}:{self.edge_count}:{self.fragment_count}"
        return compute_sha1(data)
    
    def get_l1_nodes(self) -> Set[int]:
        """Get all L1 nodes in the entanglement."""
        return set(self.matched_nodes.keys())
    
    def get_l2_nodes(self) -> Set[int]:
        """Get all L2 nodes in the entanglement."""
        return set(self.matched_nodes.values())
    
    def __repr__(self) -> str:
        return (f"EntanglementSubgraph({self.source_lattice_hash[:8]}↔{self.target_lattice_hash[:8]}, "
                f"fragments={self.fragment_count}, nodes={self.size}, edges={self.edge_count}, "
                f"sim={self.structural_similarity:.2%})")


@dataclass
class ExtendedEntanglement:
    """
    Entanglement extended (closed) to include connected nodes in each lattice.
    
    Given entanglement fragments, this extends to include:
    - L1 nodes connected to any entanglement node in L1
    - L2 nodes connected to any entanglement node in L2
    
    This is the "closure" operation: propagating entanglement to neighbors.
    """
    base_entanglement: EntanglementSubgraph
    
    # Extended L1 nodes (neighbors of entanglement in L1)
    extended_l1_nodes: Set[int] = field(default_factory=set)
    
    # Extended L2 nodes (neighbors of entanglement in L2)
    extended_l2_nodes: Set[int] = field(default_factory=set)
    
    # Edges from entanglement to extended nodes
    l1_extension_edges: Set[Tuple[int, int]] = field(default_factory=set)
    l2_extension_edges: Set[Tuple[int, int]] = field(default_factory=set)
    
    @property
    def total_l1_nodes(self) -> int:
        """Total L1 nodes (entanglement + extended)."""
        return len(self.base_entanglement.get_l1_nodes() | self.extended_l1_nodes)
    
    @property
    def total_l2_nodes(self) -> int:
        """Total L2 nodes (entanglement + extended)."""
        return len(self.base_entanglement.get_l2_nodes() | self.extended_l2_nodes)
    
    def __repr__(self) -> str:
        return (f"ExtendedEntanglement(base={self.base_entanglement.size} nodes, "
                f"L1_ext={len(self.extended_l1_nodes)}, L2_ext={len(self.extended_l2_nodes)})")


class CommonSubgraphExtractor:
    """
    Extracts common subgraph fragments (entanglement) between two lattices.
    
    Algorithm (Approximate):
    1. Compute node signatures (in-degree, out-degree, START/END flags)
    2. Match nodes by signature similarity (greedy, degree-based)
       - Degrees don't need exact match, just "close enough"
    3. Verify edges: keep edges that exist in both lattices under mapping
    4. Identify disconnected fragments (connected components)
    5. Optionally extend fragments to neighbors (closure)
    
    Properties:
    - Respects START/END markers
    - Works with directed graphs (not necessarily acyclic)
    - Returns MULTIPLE disconnected fragments (not forced to be connected)
    - Polynomial time O(n² + e)
    - "Good enough" approximation, not optimal MCS
    """
    
    def __init__(self, 
                 degree_tolerance: float = 0.3,
                 min_edge_ratio: float = 0.0,  # Relaxed: allow isolated node matches
                 prefer_start_end: bool = True):
        """
        Args:
            degree_tolerance: Max relative degree difference for matching (relaxed)
            min_edge_ratio: Min edge preservation ratio (0.0 = allow isolated matches)
            prefer_start_end: Give priority to matching START/END nodes
        """
        self.degree_tolerance = degree_tolerance
        self.min_edge_ratio = min_edge_ratio
        self.prefer_start_end = prefer_start_end
    
    def extract(self, 
                lattice1: HLLSetLattice, 
                lattice2: HLLSetLattice,
                lattice1_hash: str = "",
                lattice2_hash: str = "") -> EntanglementSubgraph:
        """
        Extract common subgraph fragments (entanglement) between two lattices.
        
        Returns:
            EntanglementSubgraph containing potentially multiple disconnected fragments
        """
        # Step 1: Build adjacency lists for both lattices
        adj1, edges1 = self._build_adjacency(lattice1)
        adj2, edges2 = self._build_adjacency(lattice2)
        
        # Step 2: Compute node signatures
        sig1 = self._compute_signatures(lattice1, adj1)
        sig2 = self._compute_signatures(lattice2, adj2)
        
        # Step 3: Match nodes by signature (greedy, "close enough" degrees)
        node_mapping = self._match_nodes(sig1, sig2, lattice1, lattice2)
        
        # Step 4: Verify edges under mapping
        matched_edges = self._verify_edges(edges1, edges2, node_mapping)
        
        # Step 5: Optionally refine mapping (remove poorly-connected nodes if required)
        if self.min_edge_ratio > 0:
            node_mapping, matched_edges = self._refine_mapping(
                node_mapping, matched_edges, adj1, adj2
            )
        
        # Step 6: Identify START/END nodes in subgraph
        start_nodes, end_nodes = self._identify_special_nodes(node_mapping, lattice1)
        
        # Step 7: Identify disconnected fragments (connected components)
        fragments = self._identify_fragments(node_mapping, matched_edges, start_nodes, end_nodes)
        
        # Compute statistics
        total_nodes = min(len(sig1), len(sig2))
        total_edges = min(len(edges1), len(edges2)) if edges1 and edges2 else 1
        
        node_coverage = len(node_mapping) / total_nodes if total_nodes > 0 else 0.0
        edge_coverage = len(matched_edges) / total_edges if total_edges > 0 else 0.0
        structural_similarity = (node_coverage + edge_coverage) / 2.0
        
        return EntanglementSubgraph(
            source_lattice_hash=lattice1_hash or "L1",
            target_lattice_hash=lattice2_hash or "L2",
            fragments=fragments,
            matched_nodes=node_mapping,
            matched_edges=matched_edges,
            start_nodes=start_nodes,
            end_nodes=end_nodes,
            node_coverage=node_coverage,
            edge_coverage=edge_coverage,
            structural_similarity=structural_similarity
        )
    
    def _identify_fragments(self,
                           node_mapping: Dict[int, int],
                           matched_edges: List[Tuple[Tuple[int, int], Tuple[int, int]]],
                           start_nodes: Set[int],
                           end_nodes: Set[int]) -> List[EntanglementFragment]:
        """
        Identify disconnected fragments (connected components) in the matching.
        
        Each fragment is a connected component of matched nodes/edges.
        Isolated matched nodes (no edges) become their own fragments.
        """
        if not node_mapping:
            return []
        
        # Build undirected adjacency from matched edges (for L1 indices)
        adj = defaultdict(set)
        for (src1, dst1), _ in matched_edges:
            adj[src1].add(dst1)
            adj[dst1].add(src1)  # Undirected for component finding
        
        # Find connected components using BFS
        visited = set()
        fragments = []
        
        for node in node_mapping.keys():
            if node in visited:
                continue
            
            # BFS to find component
            component_nodes = set()
            queue = [node]
            
            while queue:
                current = queue.pop(0)
                if current in visited:
                    continue
                if current not in node_mapping:
                    continue
                    
                visited.add(current)
                component_nodes.add(current)
                
                # Add neighbors (from matched edges)
                for neighbor in adj.get(current, []):
                    if neighbor not in visited and neighbor in node_mapping:
                        queue.append(neighbor)
            
            # Build fragment from component
            frag_mapping = {n: node_mapping[n] for n in component_nodes}
            frag_edges = [
                (e1, e2) for (e1, e2) in matched_edges
                if e1[0] in component_nodes and e1[1] in component_nodes
            ]
            frag_starts = start_nodes & component_nodes
            frag_ends = end_nodes & component_nodes
            
            fragments.append(EntanglementFragment(
                node_mapping=frag_mapping,
                matched_edges=frag_edges,
                start_nodes=frag_starts,
                end_nodes=frag_ends
            ))
        
        # Sort fragments by size (largest first)
        fragments.sort(key=lambda f: (f.size, f.edge_count), reverse=True)
        
        return fragments
    
    def extend_in_lattice(self,
                          entanglement: EntanglementSubgraph,
                          lattice1: HLLSetLattice,
                          lattice2: HLLSetLattice,
                          depth: int = 1) -> ExtendedEntanglement:
        """
        Extend entanglement to include neighboring nodes in each lattice.
        
        This is the "closure" operation: propagate entanglement seeds to
        their neighbors within each lattice structure.
        
        Args:
            entanglement: Base entanglement (fragments)
            lattice1: Source lattice (for L1 extension)
            lattice2: Target lattice (for L2 extension)
            depth: How many hops to extend (1 = immediate neighbors)
            
        Returns:
            ExtendedEntanglement with additional nodes/edges
        """
        # Build adjacency for both lattices
        adj1, _ = self._build_adjacency(lattice1)
        adj2, _ = self._build_adjacency(lattice2)
        
        # Get base entanglement nodes
        l1_base = entanglement.get_l1_nodes()
        l2_base = entanglement.get_l2_nodes()
        
        # Extend L1
        extended_l1 = set()
        l1_ext_edges = set()
        current_l1 = l1_base.copy()
        
        for _ in range(depth):
            next_l1 = set()
            for node in current_l1:
                for neighbor in adj1.get(node, []):
                    if neighbor not in l1_base and neighbor not in extended_l1:
                        extended_l1.add(neighbor)
                        next_l1.add(neighbor)
                        l1_ext_edges.add((node, neighbor))
            current_l1 = next_l1
        
        # Extend L2
        extended_l2 = set()
        l2_ext_edges = set()
        current_l2 = l2_base.copy()
        
        for _ in range(depth):
            next_l2 = set()
            for node in current_l2:
                for neighbor in adj2.get(node, []):
                    if neighbor not in l2_base and neighbor not in extended_l2:
                        extended_l2.add(neighbor)
                        next_l2.add(neighbor)
                        l2_ext_edges.add((node, neighbor))
            current_l2 = next_l2
        
        return ExtendedEntanglement(
            base_entanglement=entanglement,
            extended_l1_nodes=extended_l1,
            extended_l2_nodes=extended_l2,
            l1_extension_edges=l1_ext_edges,
            l2_extension_edges=l2_ext_edges
        )
    
    def _build_adjacency(self, lattice: HLLSetLattice) -> Tuple[Dict[int, Set[int]], Set[Tuple[int, int]]]:
        """
        Build adjacency list and edge set from lattice.
        
        Nodes are row indices. Edges are (src, dst) where morphism exists.
        Column basics are treated as intermediate (implicit) connections.
        """
        adj = defaultdict(set)
        edges = set()
        
        dim = lattice.config.dimension
        
        # Build row→col and col→row morphisms, then compose row→row
        row_to_col = defaultdict(set)  # row_idx → {col_idx}
        col_to_row = defaultdict(set)  # col_idx → {row_idx}
        
        for r_basic in lattice.row_basic:
            for c_basic in lattice.col_basic:
                if r_basic.has_morphism_to(c_basic):
                    row_to_col[r_basic.index].add(c_basic.index)
                if c_basic.has_morphism_to(r_basic):
                    col_to_row[c_basic.index].add(r_basic.index)
        
        # Compose: row → col → row (via column as intermediate)
        for r1 in lattice.row_basic:
            for c_idx in row_to_col[r1.index]:
                for r2_idx in col_to_row[c_idx]:
                    if r1.index != r2_idx:  # No self-loops
                        adj[r1.index].add(r2_idx)
                        edges.add((r1.index, r2_idx))
        
        return dict(adj), edges
    
    def _compute_signatures(self, 
                           lattice: HLLSetLattice, 
                           adj: Dict[int, Set[int]]) -> Dict[int, Tuple[int, int, bool, bool, float]]:
        """
        Compute signature for each node.
        
        Signature = (out_degree, in_degree, is_start, is_end, cardinality)
        """
        signatures = {}
        
        # Compute in-degrees
        in_degree = defaultdict(int)
        for src, dsts in adj.items():
            for dst in dsts:
                in_degree[dst] += 1
        
        for basic in lattice.row_basic:
            idx = basic.index
            out_deg = len(adj.get(idx, set()))
            in_deg = in_degree[idx]
            is_start = basic.is_start
            is_end = basic.is_end
            card = basic.hllset.cardinality()
            
            signatures[idx] = (out_deg, in_deg, is_start, is_end, card)
        
        return signatures
    
    def _match_nodes(self,
                     sig1: Dict[int, Tuple],
                     sig2: Dict[int, Tuple],
                     lattice1: HLLSetLattice,
                     lattice2: HLLSetLattice) -> Dict[int, int]:
        """
        Match nodes between lattices by signature similarity.
        
        Greedy matching: prioritize START/END, then by degree similarity.
        """
        # Build candidate pairs with similarity scores
        candidates = []
        
        for idx1, (out1, in1, start1, end1, card1) in sig1.items():
            for idx2, (out2, in2, start2, end2, card2) in sig2.items():
                
                # Compute degree similarity
                max_out = max(out1, out2, 1)
                max_in = max(in1, in2, 1)
                out_sim = 1.0 - abs(out1 - out2) / max_out
                in_sim = 1.0 - abs(in1 - in2) / max_in
                
                # Check tolerance
                if out_sim < (1.0 - self.degree_tolerance):
                    continue
                if in_sim < (1.0 - self.degree_tolerance):
                    continue
                
                # Similarity score
                degree_sim = (out_sim + in_sim) / 2.0
                
                # Bonus for START/END match
                special_bonus = 0.0
                if self.prefer_start_end:
                    if start1 and start2:
                        special_bonus = 0.3
                    elif end1 and end2:
                        special_bonus = 0.3
                    elif (start1 != start2) or (end1 != end2):
                        # Penalty for mismatched special status
                        special_bonus = -0.2
                
                total_sim = degree_sim + special_bonus
                candidates.append((total_sim, idx1, idx2))
        
        # Sort by similarity (descending)
        candidates.sort(reverse=True)
        
        # Greedy matching
        mapping = {}
        used2 = set()
        
        for sim, idx1, idx2 in candidates:
            if idx1 not in mapping and idx2 not in used2:
                mapping[idx1] = idx2
                used2.add(idx2)
        
        return mapping
    
    def _verify_edges(self,
                      edges1: Set[Tuple[int, int]],
                      edges2: Set[Tuple[int, int]],
                      node_mapping: Dict[int, int]) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """
        Verify which edges exist in both lattices under the node mapping.
        """
        matched_edges = []
        
        for src1, dst1 in edges1:
            if src1 in node_mapping and dst1 in node_mapping:
                src2 = node_mapping[src1]
                dst2 = node_mapping[dst1]
                
                if (src2, dst2) in edges2:
                    matched_edges.append(((src1, dst1), (src2, dst2)))
        
        return matched_edges
    
    def _refine_mapping(self,
                        node_mapping: Dict[int, int],
                        matched_edges: List,
                        adj1: Dict[int, Set[int]],
                        adj2: Dict[int, Set[int]]) -> Tuple[Dict[int, int], List]:
        """
        Refine mapping by removing poorly-connected nodes.
        
        A node is poorly-connected if most of its edges are not preserved.
        """
        if not node_mapping:
            return {}, []
        
        # Count preserved edges per node
        edge_count = defaultdict(int)
        for (src1, dst1), _ in matched_edges:
            edge_count[src1] += 1
            edge_count[dst1] += 1
        
        # Remove nodes with poor edge preservation
        refined_mapping = {}
        for idx1, idx2 in node_mapping.items():
            total_edges = len(adj1.get(idx1, set()))
            preserved = edge_count.get(idx1, 0)
            
            if total_edges == 0 or preserved / total_edges >= self.min_edge_ratio:
                refined_mapping[idx1] = idx2
        
        # Recompute matched edges with refined mapping
        if refined_mapping != node_mapping:
            new_matched = []
            for (src1, dst1), (src2, dst2) in matched_edges:
                if src1 in refined_mapping and dst1 in refined_mapping:
                    new_matched.append(((src1, dst1), (src2, dst2)))
            return refined_mapping, new_matched
        
        return node_mapping, matched_edges
    
    def _identify_special_nodes(self,
                                node_mapping: Dict[int, int],
                                lattice1: HLLSetLattice) -> Tuple[Set[int], Set[int]]:
        """
        Identify START and END nodes in the common subgraph.
        """
        start_nodes = set()
        end_nodes = set()
        
        for basic in lattice1.row_basic:
            if basic.index in node_mapping:
                if basic.is_start:
                    start_nodes.add(basic.index)
                if basic.is_end:
                    end_nodes.add(basic.index)
        
        return start_nodes, end_nodes


def extract_entanglement(lattice1: HLLSetLattice, 
                        lattice2: HLLSetLattice,
                        degree_tolerance: float = 0.3) -> EntanglementSubgraph:
    """
    High-level API: Extract entanglement (common subgraph) between two lattices.
    
    Args:
        lattice1: First lattice
        lattice2: Second lattice
        degree_tolerance: Max relative degree difference for node matching
        
    Returns:
        EntanglementSubgraph representing the common structure
    """
    extractor = CommonSubgraphExtractor(degree_tolerance=degree_tolerance)
    return extractor.extract(lattice1, lattice2)


# =============================================================================
# SECTION 6: N-Edge Based Entanglement (Stochastic Approach)
# =============================================================================
"""
Key Insight: Entanglement is about EDGES, not nodes.

Just as we have:
- n-tokens (sequences of tokens) → absorbed into HLLSets
- Token LUT (lookup table preserving n-gram structure)

We can have:
- n-edges (sequences of edges / paths in W lattice)
- Edge LUT (lookup table for edge sequences)
- HLLSet of n-edges for each lattice

Then: Entanglement = Intersection of n-edge HLLSets

This is:
- Pure stochastic (HLLSet-based, probabilistic)
- Simple (just intersection, no complex graph matching)
- Consistent with existing architecture (reuses token LUT pattern)
- Inspired by Karoubi envelope / idempotent completion (preserves splits)

The n-edges implicitly preserve order on HLLSets, just like n-tokens.
We don't need to build extended W - just extract paths and absorb them.
"""


@dataclass
class EdgeSignature:
    """
    Signature of an edge in the W lattice.
    
    An edge is identified by:
    - Source node hash (BasicHLLSet.name)
    - Target node hash (BasicHLLSet.name)
    - BSS weight (optional, for weighted edges)
    """
    source_hash: str
    target_hash: str
    weight: float = 1.0
    
    @property
    def token(self) -> str:
        """Edge as a token for HLLSet absorption."""
        return f"{self.source_hash}→{self.target_hash}"
    
    @property
    def weighted_token(self) -> str:
        """Edge with weight as token."""
        return f"{self.source_hash}→{self.target_hash}@{self.weight:.3f}"
    
    def __hash__(self) -> int:
        return hash(self.token)
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, EdgeSignature):
            return False
        return self.token == other.token
    
    def __repr__(self) -> str:
        return f"Edge({self.source_hash[:8]}→{self.target_hash[:8]})"


@dataclass
class NEdgePath:
    """
    An n-edge path: sequence of n consecutive edges in the W lattice.
    
    Analogous to n-token (n-gram) in text processing.
    Preserves order implicitly through the sequence structure.
    """
    edges: Tuple[EdgeSignature, ...]
    
    @property
    def n(self) -> int:
        """Length of the path (number of edges)."""
        return len(self.edges)
    
    @property
    def token(self) -> str:
        """Path as a single token for HLLSet absorption."""
        return "::".join(e.token for e in self.edges)
    
    @property
    def start_node(self) -> str:
        """Starting node of the path."""
        return self.edges[0].source_hash if self.edges else ""
    
    @property
    def end_node(self) -> str:
        """Ending node of the path."""
        return self.edges[-1].target_hash if self.edges else ""
    
    def __hash__(self) -> int:
        return hash(self.token)
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, NEdgePath):
            return False
        return self.token == other.token
    
    def __repr__(self) -> str:
        if not self.edges:
            return "NEdgePath(empty)"
        return f"NEdgePath({self.n}-edges: {self.start_node[:6]}→...→{self.end_node[:6]})"


@dataclass
class EdgeLUT:
    """
    Edge Look-Up Table - analogous to token LUT.
    
    Stores edge signatures and provides efficient lookup.
    Structure borrowed from token LUT pattern.
    """
    edges: Dict[str, EdgeSignature] = field(default_factory=dict)
    adjacency: Dict[str, Set[str]] = field(default_factory=dict)  # source → {targets}
    
    def add_edge(self, edge: EdgeSignature) -> None:
        """Add an edge to the LUT."""
        self.edges[edge.token] = edge
        if edge.source_hash not in self.adjacency:
            self.adjacency[edge.source_hash] = set()
        self.adjacency[edge.source_hash].add(edge.target_hash)
    
    def get_successors(self, node_hash: str) -> Set[str]:
        """Get all successor nodes."""
        return self.adjacency.get(node_hash, set())
    
    def has_edge(self, source: str, target: str) -> bool:
        """Check if edge exists."""
        return target in self.adjacency.get(source, set())
    
    @property
    def edge_count(self) -> int:
        return len(self.edges)
    
    @property
    def node_count(self) -> int:
        nodes = set(self.adjacency.keys())
        for targets in self.adjacency.values():
            nodes.update(targets)
        return len(nodes)
    
    def __repr__(self) -> str:
        return f"EdgeLUT(nodes={self.node_count}, edges={self.edge_count})"


@dataclass
class NEdgeEntanglement:
    """
    Entanglement computed via n-edge HLLSet intersection.
    
    Stochastic approach inspired by Karoubi idempotent completion:
    - Build HLLSet of n-edges from each lattice
    - Entanglement = intersection of these HLLSets
    - Preserves splits (structural correspondence)
    
    Properties:
    - source_lattice_hash: identifier of first lattice
    - target_lattice_hash: identifier of second lattice
    - source_nedge_hll: HLLSet absorbing all n-edges from L1
    - target_nedge_hll: HLLSet absorbing all n-edges from L2
    - entanglement_hll: intersection HLLSet
    - n: edge path length used
    """
    source_lattice_hash: str
    target_lattice_hash: str
    n: int  # n-edge length
    
    source_nedge_hll: HLLSet
    target_nedge_hll: HLLSet
    entanglement_hll: HLLSet
    
    # Statistics
    source_path_count: int = 0
    target_path_count: int = 0
    
    @property
    def source_cardinality(self) -> float:
        """Estimated distinct n-edges in source lattice."""
        return self.source_nedge_hll.cardinality()
    
    @property
    def target_cardinality(self) -> float:
        """Estimated distinct n-edges in target lattice."""
        return self.target_nedge_hll.cardinality()
    
    @property
    def entanglement_cardinality(self) -> float:
        """Estimated shared n-edges (entanglement strength)."""
        return self.entanglement_hll.cardinality()
    
    @property
    def jaccard_similarity(self) -> float:
        """Jaccard similarity of n-edge sets."""
        intersection = self.entanglement_cardinality
        union = self.source_cardinality + self.target_cardinality - intersection
        return intersection / union if union > 0 else 0.0
    
    @property
    def overlap_coefficient(self) -> float:
        """Overlap coefficient (Szymkiewicz–Simpson)."""
        intersection = self.entanglement_cardinality
        min_card = min(self.source_cardinality, self.target_cardinality)
        return intersection / min_card if min_card > 0 else 0.0
    
    @property
    def name(self) -> str:
        """Content-addressed name."""
        from .kernel import compute_sha1
        data = f"{self.source_lattice_hash}:{self.target_lattice_hash}:{self.n}:{self.entanglement_cardinality:.0f}"
        return compute_sha1(data)
    
    def __repr__(self) -> str:
        return (f"NEdgeEntanglement({self.source_lattice_hash[:8]}↔{self.target_lattice_hash[:8]}, "
                f"n={self.n}, shared≈{self.entanglement_cardinality:.0f}, "
                f"jaccard={self.jaccard_similarity:.2%})")


class NEdgeExtractor:
    """
    Extract n-edge paths from a W lattice and build HLLSet.
    
    Algorithm:
    1. Build EdgeLUT from lattice (row→col→row composition)
    2. Enumerate all n-edge paths using DFS
    3. Absorb each path into HLLSet
    
    The resulting HLLSet represents the "edge signature" of the lattice.
    Two lattices with similar structure will have similar n-edge HLLSets.
    """
    
    def __init__(self, 
                 n: int = 2,
                 p_bits: int = 8,
                 h_bits: int = 16,
                 max_paths: int = 100000):
        """
        Args:
            n: Length of edge paths (n=2 means 2-edge paths)
            p_bits: Precision bits for HLLSet
            h_bits: Hash bits for HLLSet
            max_paths: Maximum paths to enumerate (prevents explosion)
        """
        self.n = n
        self.p_bits = p_bits
        self.h_bits = h_bits
        self.max_paths = max_paths
    
    def build_edge_lut(self, lattice: HLLSetLattice) -> EdgeLUT:
        """
        Build EdgeLUT from W lattice.
        
        Edges are defined by row→col→row morphism composition.
        """
        lut = EdgeLUT()
        
        # Build row→col and col→row morphisms
        row_to_col: Dict[int, Set[int]] = defaultdict(set)
        col_to_row: Dict[int, Set[int]] = defaultdict(set)
        
        for r_basic in lattice.row_basic:
            for c_basic in lattice.col_basic:
                if r_basic.has_morphism_to(c_basic):
                    row_to_col[r_basic.index].add(c_basic.index)
                if c_basic.has_morphism_to(r_basic):
                    col_to_row[c_basic.index].add(r_basic.index)
        
        # Compose: row → col → row (via column as intermediate)
        # This gives us the effective edges in W
        for r1 in lattice.row_basic:
            for c_idx in row_to_col[r1.index]:
                for r2_idx in col_to_row[c_idx]:
                    if r1.index != r2_idx:  # No self-loops
                        r2 = lattice.row_basic[r2_idx]
                        
                        # Compute edge weight (BSS between endpoints)
                        weight = 1.0
                        if hasattr(r1.hllset, 'similarity'):
                            weight = r1.hllset.similarity(r2.hllset)
                        
                        edge = EdgeSignature(
                            source_hash=r1.hllset.name,
                            target_hash=r2.hllset.name,
                            weight=weight
                        )
                        lut.add_edge(edge)
        
        return lut
    
    def enumerate_n_paths(self, 
                          lut: EdgeLUT,
                          node_hashes: List[str]) -> List[NEdgePath]:
        """
        Enumerate all n-edge paths using DFS.
        
        Args:
            lut: Edge lookup table
            node_hashes: List of all node hashes (starting points)
            
        Returns:
            List of n-edge paths
        """
        paths = []
        
        def dfs(current_node: str, current_path: List[EdgeSignature], depth: int):
            if len(paths) >= self.max_paths:
                return
            
            if depth == self.n:
                # Found complete n-edge path
                paths.append(NEdgePath(edges=tuple(current_path)))
                return
            
            # Explore successors
            for next_node in lut.get_successors(current_node):
                edge_token = f"{current_node}→{next_node}"
                edge = lut.edges.get(edge_token)
                if edge:
                    dfs(next_node, current_path + [edge], depth + 1)
        
        # Start DFS from each node
        for start_node in node_hashes:
            if len(paths) >= self.max_paths:
                break
            dfs(start_node, [], 0)
        
        return paths
    
    def extract_hllset(self, lattice: HLLSetLattice) -> Tuple[HLLSet, int]:
        """
        Extract n-edge HLLSet from a lattice.
        
        Returns:
            (HLLSet absorbing all n-edge paths, path count)
        """
        from .kernel import Kernel
        kernel = Kernel()
        
        # Build edge LUT
        lut = self.build_edge_lut(lattice)
        
        # Get all node hashes
        node_hashes = [basic.hllset.name for basic in lattice.row_basic]
        
        # Enumerate n-edge paths
        paths = self.enumerate_n_paths(lut, node_hashes)
        
        # Create HLLSet and absorb all paths
        hll = HLLSet(p_bits=self.p_bits)
        for path in paths:
            hll = kernel.add(hll, path.token)
        return hll, len(paths)
    
    def compute_entanglement(self,
                            lattice1: HLLSetLattice,
                            lattice2: HLLSetLattice,
                            lattice1_hash: str = "",
                            lattice2_hash: str = "") -> NEdgeEntanglement:
        """
        Compute entanglement via n-edge HLLSet intersection.
        
        This is the core stochastic entanglement algorithm:
        1. Extract n-edge HLLSet from L1
        2. Extract n-edge HLLSet from L2
        3. Entanglement = intersection(L1_edges, L2_edges)
        
        Returns:
            NEdgeEntanglement with intersection HLLSet
        """
        from .kernel import Kernel
        kernel = Kernel()
        
        # Extract n-edge HLLSets
        hll1, count1 = self.extract_hllset(lattice1)
        hll2, count2 = self.extract_hllset(lattice2)
        
        # Compute intersection (entanglement)
        entanglement_hll = kernel.intersection(hll1, hll2)
        
        return NEdgeEntanglement(
            source_lattice_hash=lattice1_hash or hll1.name[:16],
            target_lattice_hash=lattice2_hash or hll2.name[:16],
            n=self.n,
            source_nedge_hll=hll1,
            target_nedge_hll=hll2,
            entanglement_hll=entanglement_hll,
            source_path_count=count1,
            target_path_count=count2
        )


def compute_nedge_entanglement(lattice1: HLLSetLattice,
                               lattice2: HLLSetLattice,
                               n: int = 2,
                               p_bits: int = 8) -> NEdgeEntanglement:
    """
    High-level API: Compute n-edge entanglement between two lattices.
    
    Stochastic approach:
    - Extract n-edge paths from each lattice
    - Absorb into HLLSets
    - Entanglement = intersection
    
    Args:
        lattice1: First W lattice
        lattice2: Second W lattice
        n: Edge path length (default 2)
        p_bits: HLLSet precision
        
    Returns:
        NEdgeEntanglement with shared structure estimate
    """
    extractor = NEdgeExtractor(n=n, p_bits=p_bits)
    return extractor.compute_entanglement(lattice1, lattice2)


# =============================================================================
# SECTION 7: Integration with MOS
# =============================================================================

def compute_hrt_entanglement(hrt1: HRT, hrt2: HRT, kernel: Kernel) -> EntanglementMorphism:
    """
    Compute entanglement between two HRTs.
    
    This is the high-level API for MOS integration.
    """
    engine = EntanglementEngine(kernel)
    
    lattice1 = hrt1.get_lattice()
    lattice2 = hrt2.get_lattice()
    
    if not lattice1 or not lattice2:
        raise ValueError("HRTs must have loaded lattices")
    
    return engine.compute_entanglement(
        lattice1, lattice2,
        hrt1.name, hrt2.name
    )


def compute_entanglement_network(hrts: List[HRT], kernel: Kernel) -> Dict[Tuple[str, str], EntanglementMorphism]:
    """
    Compute pairwise entanglements among multiple HRTs.
    
    Returns a network of entanglement morphisms.
    """
    network = {}
    engine = EntanglementEngine(kernel)
    
    for i, hrt1 in enumerate(hrts):
        for j, hrt2 in enumerate(hrts[i+1:], i+1):
            if hrt1.get_lattice() and hrt2.get_lattice():
                morphism = engine.compute_entanglement(
                    hrt1.get_lattice(),
                    hrt2.get_lattice(),
                    hrt1.name,
                    hrt2.name
                )
                network[(hrt1.name, hrt2.name)] = morphism
    
    return network


# =============================================================================
# SECTION 6: Example Usage
# =============================================================================

def main():
    """Example entanglement computation."""
    from .hrt import HRTConfig, HRTFactory
    
    print("="*70)
    print("ENTANGLEMENT - Lattice-Based Implementation")
    print("="*70)
    
    # Create kernel
    kernel = Kernel()
    
    # Create two HRTs
    config = HRTConfig(p_bits=8, h_bits=16)
    factory = HRTFactory(config)
    
    data1 = {"sensor_a": {"x", "y", "z", "w"}, "sensor_b": {"a", "b", "c"}}
    data2 = {"sensor_a": {"x", "y", "z"}, "sensor_c": {"d", "e", "f", "g"}}
    
    hrt1 = factory.create_from_perceptrons(data1, kernel)
    hrt2 = factory.create_from_perceptrons(data2, kernel)
    
    print(f"\nHRT 1: {hrt1.name[:20]}...")
    print(f"HRT 2: {hrt2.name[:20]}...")
    
    # Compute entanglement
    print("\nComputing entanglement morphism...")
    morphism = compute_hrt_entanglement(hrt1, hrt2, kernel)
    
    print(f"\nEntanglement Morphism:")
    print(f"  {morphism}")
    print(f"  Total pairs: {morphism.total_pairs}")
    print(f"  Max degree diff: {morphism.max_degree_diff:.2f}")
    
    # Show some mappings
    print(f"\nSample mappings:")
    for src_idx in list(morphism.mapping.keys())[:3]:
        targets = morphism.get_targets(src_idx)
        if targets:
            print(f"  Source {src_idx} → {len(targets)} targets")
            for tgt, strength in targets[:2]:
                print(f"    → Target {tgt}, strength={strength:.3f}")
    
    # Maximum matching
    engine = EntanglementEngine(kernel)
    matching = engine.find_maximum_matching(morphism, one_to_one=True)
    print(f"\nMaximum matching: {len(matching)} pairs")
    
    print("\n" + "="*70)
    print("Entanglement computation complete")
    print("="*70)
    
    return morphism


if __name__ == "__main__":
    main()
