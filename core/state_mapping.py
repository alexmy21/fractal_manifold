# File: core/state_mapping.py
"""
MOS State-to-State Mapping via Entanglement

Maps MOS(t-1) to MOS(t) through:
1. Decomposition of state HLLSets into optimal covers using W(t-1) and W(t)
2. Swarm representation: collections of HLLSets at each time point
3. Entanglement morphism between W(t-1) and W(t)
4. Trajectory analysis using (D, R, N) triple:
   - D (Drift): What changed from t-1 to t
   - R (Retention): What was preserved
   - N (Novelty): What emerged at t

Design Principle:
- State HLLSet = union of all HLLSets in that state
- Cover = optimal decomposition into basic HLLSets
- Entanglement = morphism between lattices W(t-1) → W(t)
- Trajectory = evolution of the HLLSet swarm

Note: Uses stateless kernel and unified storage patterns.
"""

from __future__ import annotations
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np

from .kernel import HLLSet, Kernel
from .hrt import HRT, HLLSetLattice, Cover, BasicHLLSet
from .entanglement import EntanglementMorphism, EntanglementEngine


# =============================================================================
# SECTION 1: State Decomposition
# =============================================================================

@dataclass
class StateDecomposition:
    """
    Decomposition of a state HLLSet into basic HLLSets.
    
    State HLLSet = union of all HLLSets in that state
    Cover = optimal set of basic HLLSets from lattice W
    """
    state_hash: str           # Hash of the state HLLSet
    hrt_hash: str            # Hash of HRT (lattice W)
    cover: Cover             # Optimal cover using basic HLLSets
    
    # Decomposition stats
    num_basic_used: int      # Number of basic HLLSets in cover
    coverage_ratio: float    # How well cover represents state
    
    def get_basic_indices(self) -> Set[int]:
        """Get all basic HLLSet indices used in cover."""
        indices = set(self.cover.row_indices)
        indices.update(self.cover.col_indices)
        return indices
    
    def __repr__(self) -> str:
        return f"Decomp({self.state_hash[:8]}... using {self.num_basic_used} basic)"


# =============================================================================
# SECTION 2: HLLSet Swarm
# =============================================================================

@dataclass
class HLLSetSwarm:
    """
    A swarm of HLLSets at a specific time point.
    
    Represents the collection of all HLLSets that constitute the state:
    - Root state HLLSet (union of all)
    - Constituent HLLSets from perceptrons
    - Intermediate HLLSets from processing
    """
    timestamp: float
    state_hash: str                    # Root state HLLSet
    hrt_hash: str                      # Associated HRT
    
    # Constituent HLLSets
    perceptron_hllsets: Dict[str, str] = field(default_factory=dict)  # perceptron_id -> hash
    intermediate_hllsets: List[str] = field(default_factory=list)     # Processing intermediates
    
    # Cached HLLSets (loaded on demand)
    _hllset_cache: Dict[str, HLLSet] = field(default_factory=dict, repr=False)
    
    def get_state_hllset(self, store) -> Optional[HLLSet]:
        """Get the root state HLLSet."""
        if self.state_hash not in self._hllset_cache:
            self._hllset_cache[self.state_hash] = store.get_hllset(self.state_hash)
        return self._hllset_cache.get(self.state_hash)
    
    def get_all_hashes(self) -> Set[str]:
        """Get all HLLSet hashes in this swarm."""
        hashes = {self.state_hash}
        hashes.update(self.perceptron_hllsets.values())
        hashes.update(self.intermediate_hllsets)
        return hashes
    
    def cardinality_sum(self, store) -> float:
        """Sum of cardinalities of all HLLSets in swarm."""
        total = 0.0
        for h in self.get_all_hashes():
            hll = store.get_hllset(h)
            if hll:
                total += hll.cardinality()
        return total


# =============================================================================
# SECTION 3: Trajectory Triple (D, R, N)
# =============================================================================

@dataclass(frozen=True)
class TrajectoryTriple:
    """
    Trajectory analysis triple (D, R, N).
    
    - D (Drift): What changed from t-1 to t
    - R (Retention): What was preserved  
    - N (Novelty): What emerged at t
    
    Based on the conservation law: |N| - |D| = 0 (Noether)
    """
    drift: float           # |D|: Information that drifted away
    retention: float       # |R|: Information retained
    novelty: float         # |N|: New information
    
    @property
    def noether_phi(self) -> float:
        """Noether current: Φ = |N| - |D|."""
        return self.novelty - self.drift
    
    @property
    def is_conserved(self) -> bool:
        """Check if |N| - |D| ≈ 0."""
        return abs(self.noether_phi) < 1e-6
    
    @property
    def retention_ratio(self) -> float:
        """Fraction of information retained: |R| / (|R| + |D|)."""
        total = self.retention + self.drift
        return self.retention / total if total > 0 else 0.0
    
    def __repr__(self) -> str:
        return f"DRN(D={self.drift:.2f}, R={self.retention:.2f}, N={self.novelty:.2f}, Φ={self.noether_phi:.4f})"


# =============================================================================
# SECTION 4: State-to-State Mapping
# =============================================================================

@dataclass
class StateMapping:
    """
    Complete mapping from MOS(t-1) to MOS(t).
    
    Contains:
    - Decompositions of both states
    - Entanglement morphism between W(t-1) and W(t)
    - Trajectory triple (D, R, N)
    - Swarm evolution analysis
    """
    source_state: str              # Hash of MOS(t-1) state
    target_state: str              # Hash of MOS(t) state
    
    source_decomp: StateDecomposition
    target_decomp: StateDecomposition
    
    entanglement: EntanglementMorphism  # W(t-1) → W(t)
    
    trajectory: TrajectoryTriple
    
    # Basic HLLSet mapping (derived from entanglement)
    basic_mapping: Dict[int, List[Tuple[int, float]]] = field(default_factory=dict)
    
    @property
    def name(self) -> str:
        """Content-addressed name."""
        from .kernel import compute_sha1
        data = f"{self.source_state}:{self.target_state}:{self.trajectory.noether_phi}"
        return compute_sha1(data)
    
    def get_mapped_basic_sets(self, source_idx: int) -> List[Tuple[int, float]]:
        """Get target basic indices mapped from source index."""
        return self.basic_mapping.get(source_idx, [])
    
    def __repr__(self) -> str:
        return f"Mapping({self.source_state[:8]}...→{self.target_state[:8]}..., {self.trajectory})"


# =============================================================================
# SECTION 5: State Mapper Engine
# =============================================================================

class StateMapper:
    """
    Engine for computing state-to-state mappings.
    
    Workflow:
    1. Decompose state HLLSets using respective lattices
    2. Compute entanglement between W(t-1) and W(t)
    3. Analyze trajectory using (D, R, N)
    4. Map basic HLLSets and understand swarm evolution
    """
    
    def __init__(self, kernel: Kernel):
        self.kernel = kernel
        self.entanglement_engine = EntanglementEngine(kernel)
    
    def decompose_state(self, 
                       state_hllset: HLLSet,
                       hrt: HRT,
                       state_hash: str) -> StateDecomposition:
        """
        Decompose state HLLSet into optimal cover using lattice W.
        
        State HLLSet = union of all HLLSets in state
        Cover = minimal basic HLLSets with minimal overlap
        """
        lattice = hrt.get_lattice()
        if not lattice:
            raise ValueError("HRT must have loaded lattice")
        
        # Compute optimal cover
        cover = lattice.compute_cover(state_hllset, self.kernel)
        
        # Compute coverage ratio (how well cover represents state)
        composed = lattice.compose_from_cover(cover, self.kernel)
        similarity = state_hllset.similarity(composed)
        
        return StateDecomposition(
            state_hash=state_hash,
            hrt_hash=hrt.name,
            cover=cover,
            num_basic_used=cover.size,
            coverage_ratio=similarity
        )
    
    def compute_trajectory(self,
                          source_swarm: HLLSetSwarm,
                          target_swarm: HLLSetSwarm,
                          store) -> TrajectoryTriple:
        """
        Compute trajectory triple (D, R, N) between swarms.
        
        - D = |source_only| (in source but not target)
        - R = |intersection| (in both)
        - N = |target_only| (in target but not source)
        """
        source_hashes = source_swarm.get_all_hashes()
        target_hashes = target_swarm.get_all_hashes()
        
        # Compute cardinalities
        drift = 0.0      # |D|: in source but not target
        retention = 0.0  # |R|: in both
        novelty = 0.0    # |N|: in target but not source
        
        # Get HLLSets and compute
        for h in source_hashes:
            hll = store.get_hllset(h)
            if hll:
                if h in target_hashes:
                    # In both: retention
                    retention += hll.cardinality()
                else:
                    # Only in source: drift
                    drift += hll.cardinality()
        
        for h in target_hashes:
            hll = store.get_hllset(h)
            if hll and h not in source_hashes:
                # Only in target: novelty
                novelty += hll.cardinality()
        
        return TrajectoryTriple(drift=drift, retention=retention, novelty=novelty)
    
    def map_states(self,
                   source_state: str,
                   target_state: str,
                   source_hrt: HRT,
                   target_hrt: HRT,
                   source_swarm: Optional[HLLSetSwarm] = None,
                   target_swarm: Optional[HLLSetSwarm] = None,
                   store = None) -> StateMapping:
        """
        Compute complete state-to-state mapping.
        
        This is the main API for mapping MOS(t-1) to MOS(t).
        """
        # 1. Decompose states - get HLLSets from store
        # Note: source_state and target_state are state hashes, need to get root_hllset_hash
        source_hll = None
        target_hll = None
        
        # Get root HLLSet hashes from the state objects
        source_root_hash = source_state.root_hllset_hash if hasattr(source_state, 'root_hllset_hash') else source_state
        target_root_hash = target_state.root_hllset_hash if hasattr(target_state, 'root_hllset_hash') else target_state
        
        if store:
            source_hll = store.get_hllset(source_root_hash)
            target_hll = store.get_hllset(target_root_hash)
        
        if not source_hll or not target_hll:
            raise ValueError(f"State HLLSets not found in store: source={source_hll is not None}, target={target_hll is not None}")
        
        source_decomp = self.decompose_state(source_hll, source_hrt, source_root_hash)
        target_decomp = self.decompose_state(target_hll, target_hrt, target_root_hash)
        
        # 2. Compute entanglement between W(t-1) and W(t)
        entanglement = self.entanglement_engine.compute_entanglement(
            source_hrt.get_lattice(),
            target_hrt.get_lattice(),
            source_hrt.name,
            target_hrt.name
        )
        
        # 3. Compute trajectory (D, R, N)
        try:
            if source_swarm and target_swarm:
                trajectory = self.compute_trajectory(source_swarm, target_swarm, store)
            else:
                raise ValueError("No swarms")
        except:
            # Fallback: compute from state HLLSets only
            drift = max(0, source_hll.cardinality() - target_hll.cardinality())
            novelty = max(0, target_hll.cardinality() - source_hll.cardinality())
            retention = min(source_hll.cardinality(), target_hll.cardinality())
            trajectory = TrajectoryTriple(drift, retention, novelty)
        
        # 4. Extract basic mapping from entanglement
        basic_mapping = {}
        for src_idx in source_decomp.get_basic_indices():
            targets = entanglement.get_targets(src_idx)
            if targets:
                basic_mapping[src_idx] = targets
        
        return StateMapping(
            source_state=source_root_hash,  # Use hash (string) for StateMapping
            target_state=target_root_hash,  # Use hash (string) for StateMapping
            source_decomp=source_decomp,
            target_decomp=target_decomp,
            entanglement=entanglement,
            trajectory=trajectory,
            basic_mapping=basic_mapping
        )


# =============================================================================
# SECTION 6: Swarm Evolution Analysis
# =============================================================================

@dataclass
class SwarmEvolution:
    """
    Analysis of HLLSet swarm evolution over time.
    
    Tracks how the swarm of HLLSets evolves:
    - Which HLLSets persist
    - Which HLLSets emerge
    - Which HLLSets vanish
    - Trajectory patterns
    """
    mappings: List[StateMapping] = field(default_factory=list)
    
    def add_mapping(self, mapping: StateMapping):
        """Add a state-to-state mapping."""
        self.mappings.append(mapping)
    
    def get_cumulative_trajectory(self) -> TrajectoryTriple:
        """Get cumulative trajectory over all mappings."""
        total_drift = sum(m.trajectory.drift for m in self.mappings)
        total_retention = sum(m.trajectory.retention for m in self.mappings)
        total_novelty = sum(m.trajectory.novelty for m in self.mappings)
        return TrajectoryTriple(total_drift, total_retention, total_novelty)
    
    def find_persistent_hllsets(self, min_retention: int = 2) -> Set[str]:
        """
        Find HLLSets that persist across multiple states.
        
        Args:
            min_retention: Minimum number of mappings HLLSet must persist through
        """
        persistence_count = defaultdict(int)
        
        for mapping in self.mappings:
            # Count HLLSets that appear in both source and target
            # (This is a simplified version - full version would track individual HLLSets)
            if mapping.trajectory.retention > 0:
                persistence_count[mapping.source_state] += 1
        
        return {h for h, count in persistence_count.items() if count >= min_retention}
    
    def analyze_convergence(self) -> Dict[str, float]:
        """
        Analyze if swarm is converging or diverging.
        
        Returns metrics:
        - retention_trend: Increasing = converging
        - novelty_trend: Decreasing = converging
        - noether_violations: How often |N| - |D| ≠ 0
        """
        if len(self.mappings) < 2:
            return {}
        
        retentions = [m.trajectory.retention_ratio for m in self.mappings]
        noether_phis = [m.trajectory.noether_phi for m in self.mappings]
        
        return {
            'retention_trend': np.mean(retentions[-3:]) - np.mean(retentions[:3]),
            'avg_retention': np.mean(retentions),
            'noether_violations': sum(1 for phi in noether_phis if abs(phi) > 1e-6),
            'avg_noether_phi': np.mean(noether_phis)
        }


# =============================================================================
# SECTION 7: Integration with MOS
# =============================================================================

def create_state_swarm(mos, state_hash: str) -> HLLSetSwarm:
    """
    Create HLLSetSwarm from MOS state.
    
    Collects all HLLSets associated with the state.
    
    Note: Simplified version - full implementation requires persistent state history.
    """
    # Get state from current state only (no persistent state history yet)
    if mos.current_state and mos.current_state.state_hash == state_hash:
        state = mos.current_state
    else:
        raise ValueError(f"State {state_hash} not found in current state")
    
    # Collect perceptron HLLSets from state data
    perceptron_hllsets = {}
    for pid, pdata in state.perceptron_states.items():
        # In real implementation, would track perceptron HLLSet hashes
        # For now, use state hash as proxy
        perceptron_hllsets[pid] = state.root_hllset_hash
    
    # Get HRT hash from current HRT
    hrt_hash = mos.current_hrt.name if mos.current_hrt else ""
    
    return HLLSetSwarm(
        timestamp=getattr(state, 'timestamp', 0.0),
        state_hash=state.root_hllset_hash,
        hrt_hash=hrt_hash,
        perceptron_hllsets=perceptron_hllsets
    )


def map_mos_states(mos, 
                   source_state_hash: Optional[str] = None,
                   target_state_hash: Optional[str] = None) -> Optional[StateMapping]:
    """
    High-level API to map two MOS states.
    
    If hashes not provided, uses current and parent state.
    """
    mapper = StateMapper(mos.kernel)
    
    # Determine target state
    if not target_state_hash:
        if not mos.current_state:
            return None
        target_state = mos.current_state
        target_state_hash = target_state.state_hash
    else:
        # TODO: Add state history retrieval when persistent storage supports it
        if mos.current_state and mos.current_state.state_hash == target_state_hash:
            target_state = mos.current_state
        else:
            return None
    
    if not target_state:
        return None
    
    # Determine source state
    if not source_state_hash:
        if not target_state.parent_state:
            return None
        source_state_hash = target_state.parent_state
    
    # Get source state (from current state only - no persistent state history yet)
    if mos.current_state and mos.current_state.state_hash == source_state_hash:
        source_state = mos.current_state
    else:
        # TODO: Add state history retrieval when persistent storage supports it
        return None
    
    if not source_state:
        return None
    
    # Get HRTs - use current HRT only (no HRT history storage yet)
    # TODO: Add HRT retrieval from storage when implemented
    source_hrt = mos.current_hrt
    target_hrt = mos.current_hrt
    
    if not source_hrt or not target_hrt:
        return None
    
    # Create swarms
    source_swarm = create_state_swarm(mos, source_state_hash)
    target_swarm = create_state_swarm(mos, target_state_hash)
    
    # Compute mapping - pass state objects (not hashes) so we can access root_hllset_hash
    return mapper.map_states(
        source_state,  # State object with root_hllset_hash
        target_state,  # State object with root_hllset_hash
        source_hrt,
        target_hrt,
        source_swarm,
        target_swarm,
        mos.store
    )


# =============================================================================
# SECTION 8: Example Usage
# =============================================================================

def main():
    """Example state-to-state mapping.
    
    Note: This is a simplified demo. Full state-to-state mapping requires:
    - Persistent state history (not yet implemented)
    - HRT creation and storage (works with evolution system)
    - State decomposition and cover computation (requires loaded lattices)
    
    For now, this demonstrates the basic workflow.
    """
    from .manifold_os import ManifoldOS
    from .hrt import HRTConfig
    
    print("="*70)
    print("STATE-TO-STATE MAPPING via ENTANGLEMENT (Demo)")
    print("="*70)
    print("\nNote: This is a simplified demo showing the workflow.")
    print("Full implementation requires HRT evolution system.")
    
    # Create MOS
    config = HRTConfig(p_bits=6, h_bits=8)
    mos = ManifoldOS(hrt_config=config)
    
    # Add perceptrons
    print("\n1. Setting up perceptrons...")
    mos.add_perceptron("p1", "camera")
    mos.add_perceptron("p2", "microphone")
    
    # Process t-1
    print("\n2. Processing MOS(t-1)...")
    try:
        state_t1 = mos.process_cycle({"p1": {"red", "green"}, "p2": {"low", "mid"}})
        print(f"   State created: {state_t1.state_hash[:16]}...")
    except Exception as e:
        print(f"   Error: {e}")
        print("   (This is expected without HRT evolution setup)")
    
    print("\n" + "="*70)
    print("Demo complete - see demo_unified_storage.ipynb for working examples")
    print("="*70)
    
    return None


if __name__ == "__main__":
    main()
