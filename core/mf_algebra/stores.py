"""
Stores Module - Commit Store and Bounded Evolution Store.

This module provides:
- Commit: Timestamped snapshot of processing state
- CommitStore: Linear history tracking with rollback
- CommitFingerprint: HLLSet fingerprint for commit
- FingerprintIndex: LUT-like index for commit lookup
- EvictionRecord: Record of evicted entries D(t)
- DeltaRecord: Record of new entries N(t+1)
- StateSnapshot: HLLSet representation for memory archaeology
- EvolutionState: Current bounded state T(t)
- BoundedEvolutionStore: Evolution equation implementation

Key Evolution Equation:
    T(t+1) = (T(t) ∪ N(t+1)) \\ D(t)

CA Property (Conflict-Free):
    Same content always produces same index (hash-based or vocabulary-based).
    Evicted index X can be re-activated without conflict.
    Active set + Archive = complete knowledge (no duplicates, no conflicts).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple
from collections import defaultdict
import numpy as np

from ..sparse_hrt_3d import SparseAM3D, SparseHRT3D, Sparse3DConfig, Edge3D, SparseLattice3D
from ..hllset import compute_sha1
from .lut import LookupTable
from .processing import build_w_from_am


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
# FINGERPRINT INDEX: HLLSet-based commit lookup
# ═══════════════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════════════
# BOUNDED EVOLUTION STORE - Δ-based Growth Control
# ═══════════════════════════════════════════════════════════════════════════

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
