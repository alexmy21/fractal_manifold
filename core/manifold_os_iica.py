"""
ManifoldOS IICA: Git-like HRT Management with IICA Properties
==============================================================

Design Principles (IICA):
1. Immutability: HRT never changes after creation
2. Idempotence: Duplicate operations have no effect
3. Content Addressability: HRT identity = hash(content)

This enables:
- Lossless parallel merge operations
- Easy parallel ingestion (perceptrons AND batches)
- Git-like commit/push separation
- Stacked HRT for in-memory recent history
- Single persistent store for complete HRT

Architecture:
┌─────────────────────────────────────────────────────────────────────────────┐
│                          ManifoldOS_IICA                                    │
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                          HRTStack                                     │  │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐                      │  │
│  │  │ commit_0│→│ commit_1│→│ commit_2│→│  HEAD   │  (in-memory)         │  │
│  │  │(genesis)│ │         │ │         │ │         │                      │  │
│  │  └─────────┘ └─────────┘ └─────────┘ └─────────┘                      │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                           ↓ push                                            │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                       HRTPersistentStore                              │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                    │  │
│  │  │   Commits   │  │    Blobs    │  │    Refs     │                    │  │
│  │  │  (indexed)  │  │(deduplicated│  │ main→hash   │                    │  │
│  │  └─────────────┘  └─────────────┘  └─────────────┘                    │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                         Processors                                    │  │
│  │  IngestionProcessor | QueryProcessor | EntanglementProcessor          │  │
│  │  (parallel by perceptrons OR batches via IICA merge)                  │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘

Usage:
    # Create ManifoldOS with persistent store
    mos = ManifoldOS_IICA(store_path="/path/to/store")
    
    # Ingest data (creates delta HRT, merges with HEAD, commits locally)
    mos.ingest(["token1", "token2", "token3"])
    
    # Commit is automatic, but explicit commits work too
    mos.commit("Added new tokens")
    
    # Push to persistent store
    mos.push()
    
    # Rollback
    mos.rollback()
    
    # Parallel ingestion
    mos.parallel_ingest([batch1, batch2, batch3])
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any, Set, Union, Tuple, Callable
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import threading
import time

from .hrt import HRTConfig
from .hrt_iica import HRT_IICA, EmbeddedLUT, LUTEntry, HRTStack, CommitInfo
from .hrt_store import (
    HRTPersistentStore, MemoryHRTStore, FileSystemHRTStore,
    HRTSerializer, CommitObject, create_memory_store, create_file_store
)
from .kernel import Kernel
from .hllset import HLLSet


# =============================================================================
# SECTION 1: Configuration
# =============================================================================

@dataclass(frozen=True)
class ManifoldIICAConfig:
    """Configuration for IICA ManifoldOS."""
    hrt_config: HRTConfig = field(default_factory=HRTConfig)
    
    # Stack settings
    max_stack_size: int = 100       # Max commits in memory
    auto_commit: bool = True        # Auto-commit after operations
    
    # Parallel processing
    parallel_workers: int = 8       # Number of parallel workers
    batch_size: int = 1000          # Tokens per batch for parallel
    
    # Persistence
    auto_push: bool = False         # Auto-push after commit
    push_interval: int = 10         # Steps between auto-push (if enabled)


# =============================================================================
# SECTION 2: Parallel Ingestion Worker
# =============================================================================

def _ingest_batch_worker(
    tokens: List[str],
    source_name: str,
    config: HRTConfig,
    p_bits: int = 10
) -> Tuple[bytes, List[LUTEntry], List[Tuple[int, int, float]]]:
    """
    Worker function for parallel ingestion.
    
    Returns:
        (hllset_bytes, lut_entries, am_entries)
    """
    from .hllset import HLLSet
    from .immutable_tensor import compute_element_hash
    
    # Create HLLSet for batch using from_batch
    hllset = HLLSet.from_batch(tokens, p_bits=p_bits)
    
    # Build LUT entries
    lut_entries = []
    for token in set(tokens):  # Deduplicate
        token_hash = compute_element_hash(token, bits=config.h_bits)
        reg = token_hash % (1 << p_bits)
        zeros = (token_hash >> p_bits).bit_length()
        
        entry = LUTEntry(
            token_hash=token_hash,
            reg=reg,
            zeros=zeros,
            tokens=((token,),),
            n_gram_size=1
        )
        lut_entries.append(entry)
    
    # Build AM entries
    source_hash = compute_element_hash(source_name, bits=config.h_bits)
    source_row = source_hash % config.dimension
    
    am_entries = []
    for token in set(tokens):
        token_hash = compute_element_hash(token, bits=config.h_bits)
        col_idx = (token_hash % (config.dimension - 2)) + 2
        am_entries.append((source_row, col_idx, 1.0))
    
    # Use dump_numpy() to get registers as bytes
    return (hllset.dump_numpy().tobytes(), lut_entries, am_entries)


# =============================================================================
# SECTION 3: ManifoldOS_IICA
# =============================================================================

class ManifoldOS_IICA:
    """
    Git-like HRT Management System with IICA Properties.
    
    Key Features:
    1. HRTStack: In-memory stack of committed HRTs
    2. Commit/Push separation: Local commits, explicit push
    3. Parallel merge: Lossless merge of parallel ingestion
    4. Embedded LUT: LUT is part of HRT state
    5. Single store: Complete HRT in one persistent store
    """
    
    def __init__(self,
                 config: Optional[ManifoldIICAConfig] = None,
                 store_path: Optional[str] = None,
                 store: Optional[HRTPersistentStore] = None):
        """
        Initialize ManifoldOS_IICA.
        
        Args:
            config: Configuration (uses defaults if None)
            store_path: Path to persistent store (creates file store)
            store: Custom store (overrides store_path)
        """
        self.config = config or ManifoldIICAConfig()
        
        # Initialize kernel
        self.kernel = Kernel()
        
        # Initialize persistent store
        if store is not None:
            self._store = store
        elif store_path is not None:
            self._store = create_file_store(store_path)
        else:
            self._store = create_memory_store()
        
        self._serializer = HRTSerializer(self._store)
        
        # Initialize HRT stack
        self._stack = HRTStack(
            config=self.config.hrt_config,
            max_stack_size=self.config.max_stack_size
        )
        
        # Try to load HEAD from store
        head = self._serializer.fetch_head()
        if head is not None:
            self._stack = HRTStack(
                genesis=head,
                max_stack_size=self.config.max_stack_size
            )
        
        # Track push state
        self._step_since_push = 0
        self._lock = threading.RLock()
    
    # -------------------------------------------------------------------------
    # State Access
    # -------------------------------------------------------------------------
    
    @property
    def head(self) -> HRT_IICA:
        """Get current HRT (HEAD)."""
        return self._stack.head
    
    @property
    def step_number(self) -> int:
        """Current step number."""
        return self.head.step_number
    
    @property
    def stack_depth(self) -> int:
        """Number of commits in memory."""
        return self._stack.depth
    
    @property
    def unpushed_count(self) -> int:
        """Number of commits not yet pushed."""
        return self._stack.unpushed_count
    
    def summary(self) -> Dict[str, Any]:
        """Get comprehensive summary."""
        return {
            **self.head.to_summary(),
            'stack_depth': self.stack_depth,
            'unpushed': self.unpushed_count,
            'branch': self._stack.current_branch,
            'store_type': type(self._store).__name__
        }
    
    # -------------------------------------------------------------------------
    # Ingestion (Single-threaded)
    # -------------------------------------------------------------------------
    
    def ingest(self,
               data: Union[str, List[str]],
               source_name: str = "ingestion") -> HRT_IICA:
        """
        Ingest data into system.
        
        Creates delta HRT, merges with HEAD, commits.
        
        Args:
            data: Tokens to ingest
            source_name: Source identifier
        
        Returns:
            New HRT after ingestion
        """
        # Normalize input
        if isinstance(data, str):
            tokens = data.split()
        else:
            tokens = data
        
        if not tokens:
            return self.head
        
        with self._lock:
            # Create delta HRT
            delta = self._create_delta_hrt(tokens, source_name)
            
            # Merge with HEAD (IICA: lossless merge)
            merged = self.head.merge(delta, self.kernel)
            
            # Commit
            if self.config.auto_commit:
                self._stack.commit(merged, f"Ingest {len(tokens)} tokens from {source_name}")
            
            # Auto-push check
            self._step_since_push += 1
            if self.config.auto_push and self._step_since_push >= self.config.push_interval:
                self.push()
            
            return merged
    
    def _create_delta_hrt(self, tokens: List[str], source_name: str) -> HRT_IICA:
        """Create delta HRT from tokens."""
        from .hrt import AdjacencyMatrix, HLLSetLattice
        from .immutable_tensor import compute_element_hash
        
        config = self.config.hrt_config
        
        # Create HLLSet
        hllset = self.kernel.absorb(tokens)
        
        # Build LUT entries
        lut = EmbeddedLUT.empty()
        for token in set(tokens):
            token_hash = compute_element_hash(token, bits=config.h_bits)
            reg = token_hash % (1 << config.p_bits)
            zeros = (token_hash >> config.p_bits).bit_length()
            
            entry = LUTEntry(
                token_hash=token_hash,
                reg=reg,
                zeros=zeros,
                tokens=((token,),)
            )
            lut = lut.with_entry(entry)
        
        # Build AM entries
        source_hash = compute_element_hash(source_name, bits=config.h_bits)
        source_row = source_hash % config.dimension
        
        am = AdjacencyMatrix.empty(config)
        am_entries = []
        for token in set(tokens):
            token_hash = compute_element_hash(token, bits=config.h_bits)
            col_idx = (token_hash % (config.dimension - 2)) + 2
            am_entries.append((source_row, col_idx, 1.0))
        am = am.with_entries(am_entries)
        
        # Create delta HRT
        return HRT_IICA(
            am=am,
            lattice=HLLSetLattice.empty(config),  # Will merge with HEAD
            lut=lut,
            config=config,
            parent_hrt=self.head.name,
            step_number=self.head.step_number + 1,
            source_info=f"delta:{source_name}"
        )
    
    # -------------------------------------------------------------------------
    # Parallel Ingestion (Multi-threaded/Multi-process)
    # -------------------------------------------------------------------------
    
    def parallel_ingest(self,
                        batches: List[List[str]],
                        source_prefix: str = "batch",
                        use_processes: bool = False) -> HRT_IICA:
        """
        Ingest multiple batches in parallel.
        
        IICA property: merge(merge(A, B), merge(C, D)) = merge(A, B, C, D)
        This enables lossless parallel ingestion.
        
        Args:
            batches: List of token batches
            source_prefix: Prefix for source names
            use_processes: Use ProcessPoolExecutor (for CPU-bound)
        
        Returns:
            New HRT after all batches ingested
        """
        if not batches:
            return self.head
        
        # Create delta HRTs in parallel
        delta_hrts = self._parallel_create_deltas(batches, source_prefix, use_processes)
        
        with self._lock:
            # Parallel merge (divide and conquer)
            if len(delta_hrts) == 1:
                merged_delta = delta_hrts[0]
            else:
                merged_delta = HRT_IICA.parallel_merge(delta_hrts, self.kernel)
            
            # Merge with HEAD
            result = self.head.merge(merged_delta, self.kernel)
            
            # Commit
            if self.config.auto_commit:
                self._stack.commit(result, f"Parallel ingest {len(batches)} batches")
            
            return result
    
    def _parallel_create_deltas(self,
                                batches: List[List[str]],
                                source_prefix: str,
                                use_processes: bool) -> List[HRT_IICA]:
        """Create delta HRTs in parallel."""
        from .hrt import AdjacencyMatrix, HLLSetLattice
        
        config = self.config.hrt_config
        delta_hrts = []
        
        # Use thread pool for I/O-bound (default) or process pool for CPU-bound
        executor_class = ProcessPoolExecutor if use_processes else ThreadPoolExecutor
        
        with executor_class(max_workers=self.config.parallel_workers) as executor:
            futures = []
            for i, batch in enumerate(batches):
                source_name = f"{source_prefix}_{i}"
                future = executor.submit(
                    _ingest_batch_worker,
                    batch, source_name, config, config.p_bits
                )
                futures.append((future, source_name))
            
            for future, source_name in futures:
                try:
                    hllset_bytes, lut_entries, am_entries = future.result()
                    
                    # Reconstruct delta HRT from worker results
                    lut = EmbeddedLUT.from_entries(lut_entries)
                    am = AdjacencyMatrix.empty(config).with_entries(am_entries)
                    
                    delta = HRT_IICA(
                        am=am,
                        lattice=HLLSetLattice.empty(config),
                        lut=lut,
                        config=config,
                        parent_hrt=self.head.name,
                        step_number=self.head.step_number + 1,
                        source_info=f"parallel:{source_name}"
                    )
                    delta_hrts.append(delta)
                except Exception as e:
                    # Log error but continue
                    print(f"Error in batch {source_name}: {e}")
        
        return delta_hrts
    
    # -------------------------------------------------------------------------
    # Git-like Operations
    # -------------------------------------------------------------------------
    
    def commit(self, message: str = "") -> CommitInfo:
        """
        Explicit commit (if auto_commit is off).
        
        Returns commit info.
        """
        return self._stack.commit(self.head, message)
    
    def push(self, ref_name: str = "main") -> bool:
        """
        Push all unpushed commits to persistent store.
        
        Returns True if successful.
        """
        unpushed = self._stack.get_unpushed()
        
        for commit_info in unpushed:
            try:
                self._serializer.push(
                    commit_info.hrt,
                    message=commit_info.message,
                    ref_name=ref_name
                )
                self._stack.mark_pushed(commit_info.commit_hash)
            except Exception as e:
                print(f"Error pushing {commit_info.commit_hash}: {e}")
                return False
        
        self._step_since_push = 0
        return True
    
    def pull(self, ref_name: str = "main") -> Optional[HRT_IICA]:
        """
        Pull latest from store.
        
        Returns HEAD if successful.
        """
        hrt = self._serializer.fetch_head(ref_name)
        if hrt is not None:
            with self._lock:
                self._stack.commit(hrt, f"Pulled from {ref_name}")
                self._stack.mark_pushed(hrt.name)
        return hrt
    
    def rollback(self, steps: int = 1) -> Optional[HRT_IICA]:
        """
        Rollback to previous state.
        
        Returns the state after rollback, or None if can't rollback.
        """
        with self._lock:
            for _ in range(steps):
                if self._stack.pop() is None:
                    break
            return self.head
    
    def history(self, limit: int = 10) -> List[CommitInfo]:
        """Get recent commit history."""
        return self._stack.history(limit)
    
    def checkout(self, commit_hash: str) -> Optional[HRT_IICA]:
        """
        Checkout specific commit.
        
        First tries local stack, then fetches from store.
        """
        # Try local
        hrt = self._stack.get(commit_hash)
        if hrt is not None:
            return hrt
        
        # Try store
        hrt = self._serializer.fetch(commit_hash)
        if hrt is not None:
            with self._lock:
                self._stack.commit(hrt, f"Checkout {commit_hash[:8]}")
        return hrt
    
    # -------------------------------------------------------------------------
    # Branch Operations
    # -------------------------------------------------------------------------
    
    def branch(self, name: str) -> bool:
        """Create new branch from current HEAD."""
        try:
            self._stack.branch(name)
            return True
        except ValueError:
            return False
    
    def switch_branch(self, name: str) -> bool:
        """Switch to different branch."""
        return self._stack.checkout(name)
    
    @property
    def branches(self) -> List[str]:
        """List all branches."""
        return self._stack.branches
    
    @property
    def current_branch(self) -> str:
        """Current branch name."""
        return self._stack.current_branch
    
    # -------------------------------------------------------------------------
    # Query Operations (Read-only)
    # -------------------------------------------------------------------------
    
    def get_lut_entry(self, token_hash: int) -> Optional[LUTEntry]:
        """Look up LUT entry by token hash."""
        return self.head.lut.get(token_hash)
    
    def get_cardinality(self, basic_name: str) -> float:
        """Get cardinality of basic HLLSet."""
        basic = self.head.lattice.get_basic(basic_name)
        return basic.hllset.cardinality() if basic else 0.0
    
    def get_similarity(self, name1: str, name2: str) -> float:
        """Get similarity between two basic HLLSets."""
        b1 = self.head.lattice.get_basic(name1)
        b2 = self.head.lattice.get_basic(name2)
        if b1 is None or b2 is None:
            return 0.0
        return b1.hllset.similarity(b2.hllset)
    
    # -------------------------------------------------------------------------
    # Cleanup
    # -------------------------------------------------------------------------
    
    def close(self):
        """Close and release resources."""
        # Push any unpushed commits
        if self.unpushed_count > 0:
            self.push()
        self._store.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


# =============================================================================
# SECTION 4: Factory Functions
# =============================================================================

def create_manifold_iica(
    store_path: Optional[str] = None,
    config: Optional[ManifoldIICAConfig] = None
) -> ManifoldOS_IICA:
    """
    Create ManifoldOS_IICA instance.
    
    Args:
        store_path: Path to persistent store (None for in-memory)
        config: Configuration
    
    Returns:
        Configured ManifoldOS_IICA
    """
    return ManifoldOS_IICA(config=config, store_path=store_path)


def create_parallel_manifold(
    store_path: Optional[str] = None,
    workers: int = 8,
    batch_size: int = 1000
) -> ManifoldOS_IICA:
    """
    Create ManifoldOS_IICA optimized for parallel ingestion.
    """
    config = ManifoldIICAConfig(
        parallel_workers=workers,
        batch_size=batch_size,
        auto_commit=True,
        auto_push=False
    )
    return ManifoldOS_IICA(config=config, store_path=store_path)
