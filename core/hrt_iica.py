"""
IICA-Enhanced HRT: Immutability, Idempotence, Content Addressability
=====================================================================

Key Properties (IICA):
1. Immutability: HRT state never changes after creation
2. Idempotence: merge(A, A) = A, union(X, X) = X
3. Content Addressability: HRT name is SHA1 hash of content

These properties enable:
- Lossless parallel merge operations
- Easy parallel processing (by perceptrons AND by batches)
- Git-like commit/push separation
- Stacked HRT for in-memory recent history

Architecture:
┌─────────────────────────────────────────────────────────────────────────────┐
│                              HRT_IICA                                       │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                     Core State (Content-Addressed)                    │  │
│  │  ┌──────────────┐  ┌───────────────────┐  ┌──────────────────────┐    │  │
│  │  │      AM      │  │   HLLSetLattice   │  │   EmbeddedLUT        │    │  │
│  │  │ (Adjacency)  │  │    (W Lattice)    │  │ (Token → Metadata)   │    │  │
│  │  └──────────────┘  └───────────────────┘  └──────────────────────┘    │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                       Evolution Metadata                              │  │
│  │  parent_hrt | step_number | noether_current | evolution_triple        │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘

LUT Embedding:
- LUT is now part of HRT (not external store)
- Hash collisions are tracked but don't affect IICA (known limitation)
- Content-addressed: LUT hash included in HRT name
- Can be persisted alongside HRT

HRTStack (Git-like):
- Stack of committed HRTs in memory
- commit(): Add to stack (local)  
- push(): Persist to store
- pop(): Revert to previous
- Enables branching and rollback

Parallel Processing:
- merge(HRT_a, HRT_b) is lossless and commutative
- Can split ingestion by perceptrons OR batches
- merge(merge(A, B), merge(C, D)) = merge(A, B, C, D)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import (
    Dict, List, Set, Optional, Tuple, Any, 
    FrozenSet, Iterator, Callable
)
import time
import hashlib
import json
from enum import Enum, auto

from .hllset import HLLSet, compute_sha1
from .immutable_tensor import ImmutableTensor, compute_element_hash, compute_structural_hash
from .kernel import Kernel


# =============================================================================
# SECTION 1: Embedded LUT (Part of HRT)
# =============================================================================

@dataclass(frozen=True)
class LUTEntry:
    """
    Single LUT entry (immutable).
    
    Maps token hash to source tokens (for collision resolution).
    
    Note: Hash collisions are a known IICA limitation - the same hash
    can point to different tokens. This only affects grounding, not
    the core HRT operations.
    """
    token_hash: int                      # Primary hash value
    reg: int                             # HLL register index
    zeros: int                           # Leading zeros count
    tokens: Tuple[Tuple[str, ...], ...]  # Source tokens (frozen for immutability)
    n_gram_size: int = 1                 # N-gram size (1, 2, 3, etc.)
    
    def with_token(self, token_seq: Tuple[str, ...]) -> 'LUTEntry':
        """Return new entry with additional token (collision handling)."""
        if token_seq in self.tokens:
            return self  # Idempotent
        new_tokens = self.tokens + (token_seq,)
        return LUTEntry(
            token_hash=self.token_hash,
            reg=self.reg,
            zeros=self.zeros,
            tokens=new_tokens,
            n_gram_size=self.n_gram_size
        )
    
    @property
    def has_collision(self) -> bool:
        """Check if this entry has hash collisions."""
        return len(self.tokens) > 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict."""
        return {
            'token_hash': self.token_hash,
            'reg': self.reg,
            'zeros': self.zeros,
            'tokens': [list(t) for t in self.tokens],
            'n_gram_size': self.n_gram_size
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'LUTEntry':
        """Deserialize from dict."""
        return cls(
            token_hash=d['token_hash'],
            reg=d['reg'],
            zeros=d['zeros'],
            tokens=tuple(tuple(t) for t in d['tokens']),
            n_gram_size=d.get('n_gram_size', 1)
        )


@dataclass(frozen=True)
class EmbeddedLUT:
    """
    Immutable LUT embedded in HRT.
    
    Content-addressed: LUT hash is part of HRT identity.
    
    Design:
    - entries: FrozenSet of LUTEntry (immutable)
    - Indexed by token_hash for O(1) lookup
    - Supports merge (union of entries)
    """
    entries: FrozenSet[LUTEntry] = field(default_factory=frozenset)
    _index: Dict[int, LUTEntry] = field(default_factory=dict, compare=False, hash=False)
    
    def __post_init__(self):
        """Build index after creation."""
        # Build index (mutable dict, not part of identity)
        idx = {}
        for entry in self.entries:
            idx[entry.token_hash] = entry
        object.__setattr__(self, '_index', idx)
    
    @classmethod
    def empty(cls) -> 'EmbeddedLUT':
        """Create empty LUT."""
        return cls(entries=frozenset())
    
    @classmethod
    def from_entries(cls, entries: List[LUTEntry]) -> 'EmbeddedLUT':
        """Create from list of entries."""
        return cls(entries=frozenset(entries))
    
    def get(self, token_hash: int) -> Optional[LUTEntry]:
        """Look up entry by token hash."""
        return self._index.get(token_hash)
    
    def with_entry(self, entry: LUTEntry) -> 'EmbeddedLUT':
        """Return new LUT with entry added/updated."""
        existing = self._index.get(entry.token_hash)
        
        if existing is not None:
            # Merge tokens (collision handling)
            merged = existing
            for token_seq in entry.tokens:
                merged = merged.with_token(token_seq)
            
            if merged == existing:
                return self  # Idempotent
            
            new_entries = (self.entries - {existing}) | {merged}
        else:
            new_entries = self.entries | {entry}
        
        return EmbeddedLUT(entries=new_entries)
    
    def merge(self, other: 'EmbeddedLUT') -> 'EmbeddedLUT':
        """
        Merge two LUTs.
        
        Idempotent: merge(A, A) = A
        Commutative: merge(A, B) = merge(B, A)
        Associative: merge(merge(A, B), C) = merge(A, merge(B, C))
        """
        if not other.entries:
            return self
        if not self.entries:
            return other
        
        result = self
        for entry in other.entries:
            result = result.with_entry(entry)
        
        return result
    
    @property
    def name(self) -> str:
        """Content-addressed hash of LUT."""
        if not self.entries:
            return "empty_lut"
        
        # Sort entries by hash for deterministic naming
        sorted_hashes = sorted(e.token_hash for e in self.entries)
        content = json.dumps(sorted_hashes)
        return compute_sha1(content.encode())
    
    def __len__(self) -> int:
        return len(self.entries)
    
    def __iter__(self) -> Iterator[LUTEntry]:
        return iter(self.entries)
    
    @property
    def collision_count(self) -> int:
        """Count entries with hash collisions."""
        return sum(1 for e in self.entries if e.has_collision)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict."""
        return {
            'entries': [e.to_dict() for e in self.entries],
            'name': self.name
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'EmbeddedLUT':
        """Deserialize from dict."""
        entries = [LUTEntry.from_dict(e) for e in d.get('entries', [])]
        return cls.from_entries(entries)


# =============================================================================
# SECTION 2: HRT_IICA (Enhanced HRT with embedded LUT)
# =============================================================================

@dataclass(frozen=True)
class HRT_IICA:
    """
    IICA-Enhanced Hash Relational Tensor.
    
    IICA Properties:
    1. Immutable: All fields frozen, modifications create new HRT
    2. Idempotent: merge(self, self) = self
    3. Content Addressable: name = hash(am, lattice, lut)
    
    Embedded LUT:
    - LUT is part of HRT state
    - Content-addressed along with AM and Lattice
    - Hash collisions are tracked but don't break IICA
    
    This design enables:
    - Lossless parallel merge
    - Easy parallel ingestion (perceptrons OR batches)
    - Git-like commit/push separation
    """
    # Core state (all contribute to content hash)
    am: 'AdjacencyMatrix'
    lattice: 'HLLSetLattice'  
    lut: EmbeddedLUT
    config: 'HRTConfig'
    
    # Evolution metadata
    parent_hrt: Optional[str] = None
    step_number: int = 0
    
    # Optional metadata (not part of identity)
    covers: Tuple['Cover', ...] = field(default_factory=tuple, compare=False)
    noether_current: Optional['NoetherCurrent'] = field(default=None, compare=False)
    timestamp: float = field(default_factory=time.time, compare=False)
    source_info: Optional[str] = field(default=None, compare=False)
    
    @classmethod
    def empty(cls, config: 'HRTConfig') -> 'HRT_IICA':
        """Create empty HRT_IICA (genesis state)."""
        from .hrt import AdjacencyMatrix, HLLSetLattice
        
        return cls(
            am=AdjacencyMatrix.empty(config),
            lattice=HLLSetLattice.empty(config),
            lut=EmbeddedLUT.empty(),
            config=config,
            parent_hrt=None,
            step_number=0
        )
    
    @property
    def name(self) -> str:
        """
        Content-addressed name.
        
        SHA1 of (am_hash, lattice_hash, lut_hash, parent, step)
        """
        components = [
            self.am.name,
            self.lattice.name,
            self.lut.name,
            str(self.step_number),
            self.parent_hrt or "genesis"
        ]
        return compute_structural_hash(*components)
    
    def merge(self, other: 'HRT_IICA', kernel: Kernel) -> 'HRT_IICA':
        """
        Merge two HRT_IICAs.
        
        IICA Properties:
        - Idempotent: merge(A, A) = A
        - Commutative: merge(A, B) = merge(B, A) 
        - Associative: merge(merge(A, B), C) = merge(A, merge(B, C))
        
        This enables lossless parallel processing.
        """
        if self.name == other.name:
            return self  # Idempotent
        
        if self.config != other.config:
            raise ValueError("Cannot merge HRTs with different configs")
        
        # Merge AM (element-wise max)
        merged_am = self.am.merge(other.am)
        
        # Merge Lattice (union of HLLSets)
        merged_lattice = self.lattice.merge(other.lattice, kernel)
        
        # Merge LUT (union of entries)
        merged_lut = self.lut.merge(other.lut)
        
        # Combine covers
        merged_covers = tuple(set(self.covers) | set(other.covers))
        
        # New step number
        new_step = max(self.step_number, other.step_number) + 1
        
        return HRT_IICA(
            am=merged_am,
            lattice=merged_lattice,
            lut=merged_lut,
            config=self.config,
            parent_hrt=self.name,  # This HRT becomes parent
            step_number=new_step,
            covers=merged_covers,
            source_info=f"merge({self.name[:8]}, {other.name[:8]})"
        )
    
    @classmethod
    def parallel_merge(cls, hrts: List['HRT_IICA'], kernel: Kernel) -> 'HRT_IICA':
        """
        Merge multiple HRTs in parallel.
        
        Due to associativity, order doesn't matter:
        merge(A, B, C, D) = merge(merge(A, B), merge(C, D))
        
        This enables efficient parallel ingestion.
        """
        if not hrts:
            raise ValueError("Cannot merge empty list")
        
        if len(hrts) == 1:
            return hrts[0]
        
        # Divide and conquer (can be parallelized)
        mid = len(hrts) // 2
        left = cls.parallel_merge(hrts[:mid], kernel)
        right = cls.parallel_merge(hrts[mid:], kernel)
        
        return left.merge(right, kernel)
    
    def with_lut_entry(self, entry: LUTEntry) -> 'HRT_IICA':
        """Return new HRT with LUT entry added."""
        new_lut = self.lut.with_entry(entry)
        if new_lut is self.lut:
            return self  # Idempotent
        
        return HRT_IICA(
            am=self.am,
            lattice=self.lattice,
            lut=new_lut,
            config=self.config,
            parent_hrt=self.name,
            step_number=self.step_number + 1,
            covers=self.covers
        )
    
    def intersect(self, other: 'HRT_IICA', kernel: Kernel) -> 'HRT_IICA':
        """
        Intersect two HRT_IICAs.
        
        Also idempotent: intersect(A, A) = A
        """
        if self.name == other.name:
            return self  # Idempotent
        
        # Intersect lattices (intersection of HLLSets)
        intersected_lattice = self.lattice.intersect(other.lattice, kernel)
        
        # Intersect LUT (common entries only)
        common_hashes = {e.token_hash for e in self.lut.entries} & {e.token_hash for e in other.lut.entries}
        common_entries = [e for e in self.lut.entries if e.token_hash in common_hashes]
        intersected_lut = EmbeddedLUT.from_entries(common_entries)
        
        # Intersect AM (element-wise min where both non-zero)
        intersected_am = self.am.intersect(other.am)
        
        return HRT_IICA(
            am=intersected_am,
            lattice=intersected_lattice,
            lut=intersected_lut,
            config=self.config,
            parent_hrt=self.name,
            step_number=max(self.step_number, other.step_number) + 1,
            source_info=f"intersect({self.name[:8]}, {other.name[:8]})"
        )
    
    def difference(self, other: 'HRT_IICA', kernel: Kernel) -> 'HRT_IICA':
        """
        Compute difference: self - other.
        
        Note: Not commutative (A - B ≠ B - A)
        """
        # Difference lattices
        diff_lattice = self.lattice.difference(other.lattice, kernel)
        
        # LUT: keep entries not in other
        other_hashes = {e.token_hash for e in other.lut.entries}
        diff_entries = [e for e in self.lut.entries if e.token_hash not in other_hashes]
        diff_lut = EmbeddedLUT.from_entries(diff_entries)
        
        return HRT_IICA(
            am=self.am,  # Keep AM (structural)
            lattice=diff_lattice,
            lut=diff_lut,
            config=self.config,
            parent_hrt=self.name,
            step_number=self.step_number + 1,
            source_info=f"diff({self.name[:8]}, {other.name[:8]})"
        )
    
    # -------------------------------------------------------------------------
    # Convenience Methods
    # -------------------------------------------------------------------------
    
    def get_total_cardinality(self) -> float:
        """Get sum of all cardinalities in lattice."""
        total = 0.0
        for basic in self.lattice.row_basic:
            total += basic.hllset.cardinality()
        for basic in self.lattice.col_basic:
            total += basic.hllset.cardinality()
        return total
    
    def get_lut_coverage(self) -> Dict[str, Any]:
        """Get LUT statistics."""
        return {
            'total_entries': len(self.lut),
            'collision_count': self.lut.collision_count,
            'collision_rate': self.lut.collision_count / len(self.lut) if self.lut else 0.0
        }
    
    def to_summary(self) -> Dict[str, Any]:
        """Comprehensive summary."""
        return {
            'name': self.name,
            'step_number': self.step_number,
            'parent': self.parent_hrt,
            'dimension': self.config.dimension,
            'am_entries': len(self.am.nonzero_entries()),
            'total_cardinality': self.get_total_cardinality(),
            'lut_entries': len(self.lut),
            'lut_collisions': self.lut.collision_count,
            'covers': len(self.covers),
            'timestamp': self.timestamp,
            'source_info': self.source_info
        }
    
    def __hash__(self) -> int:
        return hash(self.name)
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, HRT_IICA):
            return False
        return self.name == other.name
    
    def __repr__(self) -> str:
        return f"HRT_IICA({self.name[:16]}..., step={self.step_number}, lut={len(self.lut)})"


# =============================================================================
# SECTION 3: HRT Stack (Git-like In-Memory History)
# =============================================================================

class CommitInfo:
    """Information about a commit."""
    
    def __init__(self, 
                 hrt: HRT_IICA,
                 message: str = "",
                 timestamp: Optional[float] = None):
        self.hrt = hrt
        self.message = message
        self.timestamp = timestamp or time.time()
        self.commit_hash = hrt.name
    
    def __repr__(self) -> str:
        return f"Commit({self.commit_hash[:8]}, '{self.message[:20]}...')"


class HRTStack:
    """
    Git-like stack of HRT commits.
    
    Separation of concerns:
    - commit(): Add HRT to local stack (in-memory)
    - push(): Persist stack to store
    - pop(): Revert to previous state
    - branch(): Create branch from current state
    
    This enables:
    - Fast local operations (commit is cheap)
    - Batched persistence (push when ready)
    - Easy rollback (pop to previous)
    - Stacked recent history in memory
    """
    
    def __init__(self, 
                 genesis: Optional[HRT_IICA] = None,
                 config: Optional['HRTConfig'] = None,
                 max_stack_size: int = 100):
        """
        Initialize HRT stack.
        
        Args:
            genesis: Initial HRT (creates empty if None)
            config: Config for empty genesis
            max_stack_size: Max commits to keep in memory
        """
        from .hrt import HRTConfig
        
        if genesis is not None:
            self._genesis = genesis
        else:
            cfg = config or HRTConfig()
            self._genesis = HRT_IICA.empty(cfg)
        
        self._stack: List[CommitInfo] = [CommitInfo(self._genesis, "genesis")]
        self._max_size = max_stack_size
        self._pushed_hashes: Set[str] = set()  # Track what's been pushed
        
        # Branch support
        self._branch_name = "main"
        self._branches: Dict[str, List[CommitInfo]] = {"main": self._stack}
    
    @property
    def head(self) -> HRT_IICA:
        """Get current HRT (HEAD)."""
        return self._stack[-1].hrt
    
    @property
    def depth(self) -> int:
        """Number of commits in stack."""
        return len(self._stack)
    
    @property
    def unpushed_count(self) -> int:
        """Number of commits not yet pushed."""
        return sum(1 for c in self._stack if c.commit_hash not in self._pushed_hashes)
    
    def commit(self, hrt: HRT_IICA, message: str = "") -> CommitInfo:
        """
        Commit HRT to local stack.
        
        Cheap operation - just adds to in-memory stack.
        Use push() to persist.
        """
        if hrt.name == self.head.name:
            # Idempotent - same HRT, no new commit
            return self._stack[-1]
        
        info = CommitInfo(hrt, message)
        self._stack.append(info)
        
        # Trim stack if too large
        if len(self._stack) > self._max_size:
            # Keep first (genesis) and last max_size-1
            self._stack = [self._stack[0]] + self._stack[-(self._max_size-1):]
        
        return info
    
    def pop(self) -> Optional[HRT_IICA]:
        """
        Revert to previous state.
        
        Returns the popped HRT, or None if at genesis.
        """
        if len(self._stack) <= 1:
            return None  # Can't pop genesis
        
        popped = self._stack.pop()
        return popped.hrt
    
    def get(self, commit_hash: str) -> Optional[HRT_IICA]:
        """Get HRT by commit hash."""
        for info in self._stack:
            if info.commit_hash == commit_hash:
                return info.hrt
        return None
    
    def history(self, limit: int = 10) -> List[CommitInfo]:
        """Get recent commit history."""
        return list(reversed(self._stack[-limit:]))
    
    def diff(self, from_hash: str, to_hash: str) -> Optional[Dict[str, Any]]:
        """Compute diff between two commits."""
        from_hrt = self.get(from_hash)
        to_hrt = self.get(to_hash)
        
        if from_hrt is None or to_hrt is None:
            return None
        
        return {
            'from': from_hash,
            'to': to_hash,
            'lut_added': len(to_hrt.lut) - len(from_hrt.lut),
            'cardinality_change': to_hrt.get_total_cardinality() - from_hrt.get_total_cardinality(),
            'steps': to_hrt.step_number - from_hrt.step_number
        }
    
    # -------------------------------------------------------------------------
    # Branch Support
    # -------------------------------------------------------------------------
    
    def branch(self, name: str) -> 'HRTStack':
        """Create a new branch from current HEAD."""
        if name in self._branches:
            raise ValueError(f"Branch '{name}' already exists")
        
        # Copy current stack
        new_stack = list(self._stack)
        self._branches[name] = new_stack
        
        # Return new stack view
        branch_stack = HRTStack.__new__(HRTStack)
        branch_stack._genesis = self._genesis
        branch_stack._stack = new_stack
        branch_stack._max_size = self._max_size
        branch_stack._pushed_hashes = set(self._pushed_hashes)
        branch_stack._branch_name = name
        branch_stack._branches = self._branches
        
        return branch_stack
    
    def checkout(self, branch_name: str) -> bool:
        """Switch to a different branch."""
        if branch_name not in self._branches:
            return False
        
        self._stack = self._branches[branch_name]
        self._branch_name = branch_name
        return True
    
    @property
    def current_branch(self) -> str:
        """Get current branch name."""
        return self._branch_name
    
    @property
    def branches(self) -> List[str]:
        """List all branch names."""
        return list(self._branches.keys())
    
    # -------------------------------------------------------------------------
    # Push (Persistence)
    # -------------------------------------------------------------------------
    
    def mark_pushed(self, commit_hash: str):
        """Mark a commit as pushed to persistent store."""
        self._pushed_hashes.add(commit_hash)
    
    def get_unpushed(self) -> List[CommitInfo]:
        """Get commits that haven't been pushed yet."""
        return [c for c in self._stack if c.commit_hash not in self._pushed_hashes]
    
    def __repr__(self) -> str:
        return (f"HRTStack(branch={self._branch_name}, depth={self.depth}, "
                f"unpushed={self.unpushed_count}, head={self.head.name[:8]}...)")


# =============================================================================
# SECTION 4: Factory Functions
# =============================================================================

def create_hrt_iica(config: Optional['HRTConfig'] = None) -> HRT_IICA:
    """Create empty HRT_IICA."""
    from .hrt import HRTConfig
    cfg = config or HRTConfig()
    return HRT_IICA.empty(cfg)


def create_stack(config: Optional['HRTConfig'] = None, 
                 max_size: int = 100) -> HRTStack:
    """Create new HRT stack."""
    return HRTStack(config=config, max_stack_size=max_size)
