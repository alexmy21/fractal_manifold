"""
ManifoldOS (mf_os) - Orchestration Layer for Manifold Operations
=================================================================

ManifoldOS provides:
- Git-like versioning (commit, checkout, rollback)
- DuckDB persistence for HRT, W, and LUT
- Unified processing pipeline from mf_algebra
- Perceptron/Actuator architecture for input processing

Architecture:
┌─────────────────────────────────────────────────────────────────────────┐
│                          ManifoldOS (mf_os)                             │
│   - Git-like operations: commit, push, pull, rollback                  │
│   - Orchestrates processing pipeline                                    │
│   - Manages DuckDB persistence                                          │
├─────────────────────────────────────────────────────────────────────────┤
│                       ManifoldAlgebra (mf_algebra)                      │
│   - SparseHRT3D (AM + Lattice + LUT)                                   │
│   - Unified Processing Pipeline                                        │
│   - Perceptrons & Actuators                                            │
├─────────────────────────────────────────────────────────────────────────┤
│                        DuckDB Store Layer                               │
│   - Tables: commits, blobs, lut_entries, refs                          │
│   - Content-addressed storage                                          │
└─────────────────────────────────────────────────────────────────────────┘

Usage:
    from core.mf_os import ManifoldOS
    
    # Create ManifoldOS with persistent store
    mos = ManifoldOS("data/manifold.duckdb")
    
    # Ingest text
    mos.ingest("The quick brown fox...", source="sample.txt")
    
    # Query
    result = mos.query("fox", top_k=5)
    
    # View history
    mos.log(10)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any, TYPE_CHECKING
from pathlib import Path

from .mf_algebra import (
    # Core HRT Classes
    Sparse3DConfig,
    SparseHRT3D,
    SparseAM3D,
    SparseLattice3D,
    Edge3D,
    # LUT
    LookupTable,
    START, END,
    # Processing
    unified_process,
    build_w_from_am,
    # Perceptrons
    Perceptron,
    PromptPerceptron,
    ResponseActuator,
    # Operations
    reachable_from,
    project_layer,
    # Sparse matrices
    Sparse3DMatrix,
    # Query interface
    QueryContext,
    create_query_context,
    ask,
    # Disambiguation
    LayerHLLSets,
    # Stores
    CommitStore,
)
from .duckdb_store import DuckDBStore
from .constants import P_BITS


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class ManifoldOSConfig:
    """Configuration for ManifoldOS."""
    p_bits: int = P_BITS
    h_bits: int = 32
    max_n: int = 3
    auto_commit: bool = True
    auto_sync_lut: bool = True
    
    @property
    def sparse_config(self) -> Sparse3DConfig:
        return Sparse3DConfig(
            p_bits=self.p_bits,
            h_bits=self.h_bits,
            max_n=self.max_n
        )


# =============================================================================
# Text File Perceptron
# =============================================================================

class TextFilePerceptron(Perceptron):
    """
    Perceptron for plain text files (.txt).
    
    Simple and effective for normal documents.
    """
    
    def __init__(self, config: Sparse3DConfig):
        super().__init__("p_text", [".txt"], config)
    
    def extract_text(self, path: Path) -> str:
        """Read text file content."""
        try:
            return path.read_text(encoding='utf-8')
        except UnicodeDecodeError:
            # Fallback to latin-1 if UTF-8 fails
            return path.read_text(encoding='latin-1')


# =============================================================================
# ManifoldOS
# =============================================================================

class ManifoldOS:
    """
    ManifoldOS - Orchestration layer for Manifold operations.
    
    Features:
    - Git-like versioning (commit, checkout, rollback)
    - DuckDB persistence for HRT, W, and LUT
    - Unified processing pipeline from mf_algebra
    - Perceptron/Actuator architecture
    """
    
    def __init__(
        self,
        db_path: str = ":memory:",
        config: Optional[ManifoldOSConfig] = None
    ):
        """
        Initialize ManifoldOS.
        
        Args:
            db_path: Path to DuckDB database, or ":memory:" for in-memory
            config: Configuration options
        """
        self.config = config or ManifoldOSConfig()
        self.sparse_config = self.config.sparse_config
        
        # Initialize DuckDB store
        self.store = DuckDBStore(db_path)
        
        # Initialize in-memory state
        self._init_state()
        
        # Track current commit
        self._head_commit_id: Optional[str] = None
    
    def _init_state(self):
        """Initialize empty state or load from store."""
        # Try to load HEAD from store (now includes LayerHLLSets)
        hrt, W, layer_hllsets = self.store.get_head(self.sparse_config)
        
        if hrt is not None and W is not None:
            self.hrt = hrt
            self.W = W
            self.lut = self.store.load_lut_to_memory(self.sparse_config)
            self._head_commit_id = self.store.get_ref('HEAD')
            # Restore LayerHLLSets (or rebuild from AM for legacy commits)
            if layer_hllsets is not None:
                self._layer_hllsets = layer_hllsets
            else:
                self._layer_hllsets = LayerHLLSets.from_am(hrt.am, self.sparse_config.p_bits)
        else:
            # Create empty state
            empty_am = SparseAM3D.from_edges(self.sparse_config, [])
            empty_lattice = SparseLattice3D.from_sparse_am(empty_am)
            self.hrt = SparseHRT3D(
                am=empty_am,
                lattice=empty_lattice,
                config=self.sparse_config,
                lut=frozenset(),
                step=0
            )
            self.W: Dict[int, Dict[int, Dict[int, float]]] = {
                n: {} for n in range(self.config.max_n)
            }
            self.lut = LookupTable(config=self.sparse_config)
            self.lut.add_ntoken(START)
            self.lut.add_ntoken(END)
            self._layer_hllsets = LayerHLLSets.empty(self.sparse_config.p_bits)
        
        # Initialize QueryContext for ask()
        self._init_query_context()
    
    def _init_query_context(self):
        """Initialize or update QueryContext with current state."""
        self._commit_store = CommitStore()
        
        # Use existing _layer_hllsets if already set (from _init_state)
        if not hasattr(self, '_layer_hllsets'):
            self._layer_hllsets = LayerHLLSets.empty(self.sparse_config.p_bits)
        
        self._prompt_perceptron = PromptPerceptron(self.sparse_config)
        self._prompt_perceptron.initialize(self.lut)
        
        self._response_actuator = ResponseActuator()
        
        self.query_ctx = QueryContext(
            config=self.sparse_config,
            lut=self.lut,
            hrt=self.hrt,
            W=self.W,
            store=self._commit_store,
            layer_hllsets=self._layer_hllsets,
            prompt_perceptron=self._prompt_perceptron,
            response_actuator=self._response_actuator,
            max_n=self.config.max_n
        )
    
    # -------------------------------------------------------------------------
    # Ingestion
    # -------------------------------------------------------------------------
    
    def ingest(
        self,
        text: str,
        source: str = "input",
        perceptron: str = "manual"
    ) -> str:
        """
        Ingest text into the manifold.
        
        Args:
            text: Text content to ingest
            source: Source identifier (e.g., filename)
            perceptron: Perceptron name that processed this
        
        Returns:
            commit_id if auto_commit is True, empty string otherwise
        """
        if not text.strip():
            return ""
        
        # Process through unified pipeline
        result = unified_process(
            text,
            self.hrt,
            self.W,
            self.sparse_config,
            self.lut,
            self.config.max_n
        )
        
        # Update state
        self.hrt = result.merged_hrt
        self.W = build_w_from_am(self.hrt.am, self.sparse_config)
        
        # Update LayerHLLSets from new edges (both input AND context)
        from .mf_algebra import update_layer_hllsets
        start_indices = {b.to_index(self.sparse_config) for b in result.input_basics[:1]} if result.input_basics else set()
        # Combine input_edges (from input tokens) and context_edges (from W extension)
        all_edges = list(result.input_edges) + list(result.context_edges)
        self._layer_hllsets = update_layer_hllsets(
            all_edges,
            self._layer_hllsets,
            start_indices
        )
        
        # Auto-commit
        commit_id = ""
        if self.config.auto_commit:
            commit_id = self.commit(source, perceptron)
        
        return commit_id
    
    def ingest_file(
        self,
        path: Path,
        perceptron: Optional[Perceptron] = None
    ) -> str:
        """
        Ingest a file using appropriate perceptron.
        
        Args:
            path: Path to file
            perceptron: Perceptron to use (defaults to TextFilePerceptron)
        
        Returns:
            commit_id
        """
        if perceptron is None:
            perceptron = TextFilePerceptron(self.sparse_config)
            perceptron.initialize(self.lut)
        
        text = perceptron.extract_text(path)
        return self.ingest(text, str(path), perceptron.name)
    
    def ingest_directory(
        self,
        directory: Path,
        extensions: List[str] = None,
        max_files: Optional[int] = None,
        verbose: bool = True
    ) -> List[str]:
        """
        Ingest all files in a directory.
        
        Args:
            directory: Directory to scan
            extensions: File extensions to include (default: ['.txt'])
            max_files: Maximum files to process
            verbose: Print progress
        
        Returns:
            List of commit IDs
        """
        extensions = extensions or ['.txt']
        commit_ids = []
        
        files = [
            f for f in directory.rglob('*')
            if f.is_file() and f.suffix in extensions
        ]
        
        if max_files:
            files = files[:max_files]
        
        for i, path in enumerate(files):
            try:
                commit_id = self.ingest_file(path)
                commit_ids.append(commit_id)
                if verbose:
                    print(f"[{i+1}/{len(files)}] ✓ {path.name}")
            except Exception as e:
                if verbose:
                    print(f"[{i+1}/{len(files)}] ✗ {path.name}: {e}")
        
        return commit_ids
    
    # -------------------------------------------------------------------------
    # Git-like Operations
    # -------------------------------------------------------------------------
    
    def commit(
        self,
        source: str,
        perceptron: str,
        message: str = ""
    ) -> str:
        """
        Create a new commit with current state.
        
        Args:
            source: Source identifier
            perceptron: Perceptron that processed this
            message: Optional commit message
        
        Returns:
            commit_id
        """
        commit_id = self.store.commit(
            self.hrt,
            self.W,
            source,
            perceptron,
            parent_id=self._head_commit_id,
            metadata={'message': message} if message else None,
            layer_hllsets=self._layer_hllsets
        )
        
        self._head_commit_id = commit_id
        self.store.set_ref('HEAD', commit_id)
        
        # Sync LUT to store
        if self.config.auto_sync_lut:
            self.store.sync_lut_from_memory(self.lut)
        
        return commit_id
    
    def checkout(self, commit_id: str) -> bool:
        """
        Checkout a specific commit.
        
        Args:
            commit_id: Commit to checkout
        
        Returns:
            True if successful
        """
        hrt, W, layer_hllsets = self.store.checkout(commit_id, self.sparse_config)
        if hrt is None:
            return False
        
        self.hrt = hrt
        self.W = W
        self._head_commit_id = commit_id
        
        # Restore LayerHLLSets (or create empty if not in commit)
        if layer_hllsets is not None:
            self._layer_hllsets = layer_hllsets
        else:
            # Rebuild from AM for legacy commits
            self._layer_hllsets = LayerHLLSets.from_am(hrt.am, self.sparse_config.p_bits)
        
        # Update QueryContext
        self.query_ctx.hrt = self.hrt
        self.query_ctx.W = self.W
        self.query_ctx.layer_hllsets = self._layer_hllsets
        
        return True
    
    def rollback(self, steps: int = 1) -> bool:
        """
        Rollback to a previous commit.
        
        Args:
            steps: Number of commits to go back
        
        Returns:
            True if successful
        """
        if not self._head_commit_id:
            return False
        
        commit = self.store.get_commit(self._head_commit_id)
        
        for _ in range(steps):
            if commit is None or commit['parent_id'] is None:
                return False
            commit = self.store.get_commit(commit['parent_id'])
        
        if commit:
            return self.checkout(commit['commit_id'])
        return False
    
    def log(self, limit: int = 10) -> List[Dict]:
        """
        Get commit history.
        
        Args:
            limit: Maximum number of commits to return
        
        Returns:
            List of commit metadata
        """
        return self.store.log(limit)
    
    # -------------------------------------------------------------------------
    # Query Interface
    # -------------------------------------------------------------------------
    
    def query(
        self,
        prompt: str,
        top_k: int = 10,
        learn: bool = True
    ) -> Dict[str, Any]:
        """
        Query the manifold.
        
        The query is first ingested into the AM (like any input), then we find
        related concepts by following edges from query tokens.
        
        Args:
            prompt: Query text
            top_k: Number of results to return
            learn: If True, the query is also ingested (co-adaptive learning)
        
        Returns:
            Query results with tokens and scores
        """
        if not prompt.strip():
            return {
                'prompt': prompt,
                'query_indices': [],
                'results': [],
                'edges_added': 0
            }
        
        # First, ingest the query to AM (this connects query tokens to existing structure)
        result = unified_process(
            prompt,
            self.hrt,
            self.W,
            self.sparse_config,
            self.lut,
            self.config.max_n
        )
        
        # Get query indices BEFORE merging (so we know what's new)
        query_indices = set()
        for basic in result.input_basics:
            query_indices.add(basic.to_index(self.sparse_config))
        
        # Update state if learning
        if learn:
            self.hrt = result.merged_hrt
            self.W = build_w_from_am(self.hrt.am, self.sparse_config)
        
        # Now search using the MERGED HRT (so query tokens are connected)
        search_hrt = result.merged_hrt
        
        # Find reachable concepts from query indices
        AM = Sparse3DMatrix.from_am(search_hrt.am, self.sparse_config)
        
        # Search in layer 0 (1-grams) for better token resolution
        layer0 = project_layer(AM, 0)
        
        # Find all reachable nodes (1 hop = direct connections)
        reachable = reachable_from(layer0, query_indices, hops=1)
        
        # Also include concepts that POINT TO query tokens (reverse lookup)
        layer0_dict = layer0.to_dict()
        reverse_reachable = set()
        for row_idx, cols in layer0_dict.items():
            for col_idx in cols:
                if col_idx in query_indices:
                    reverse_reachable.add(row_idx)
        
        all_reachable = reachable | reverse_reachable
        
        # Score by connectivity strength
        scores = {}
        for idx in all_reachable:
            if idx in query_indices:
                continue  # Skip the query tokens themselves
            
            score = 0.0
            # Forward score: sum of edge weights FROM this idx
            if idx in layer0_dict:
                score += sum(layer0_dict[idx].values())
            # Reverse score: sum of edge weights TO this idx
            for row_idx, cols in layer0_dict.items():
                if idx in cols:
                    score += cols[idx]
            
            if score > 0:
                scores[idx] = score
        
        # Get top-k results with token resolution
        top = sorted(scores.items(), key=lambda x: -x[1])[:top_k]
        results = []
        for idx, score in top:
            ntokens = self.lut.index_to_ntokens.get(idx, set())
            if ntokens:
                _, ntoken = next(iter(ntokens))
                results.append({
                    'index': idx,
                    'tokens': ntoken,
                    'score': score
                })
            else:
                results.append({
                    'index': idx,
                    'tokens': f'<idx:{idx}>',
                    'score': score
                })
        
        # Commit if learning
        if learn and self.config.auto_commit:
            self.commit(f"query:{prompt[:50]}", "p_query")
        
        return {
            'prompt': prompt,
            'query_indices': list(query_indices),
            'results': results,
            'edges_added': len(result.context_edges)
        }
    
    # -------------------------------------------------------------------------
    # Status & Info
    # -------------------------------------------------------------------------
    
    def status(self) -> Dict[str, Any]:
        """Get current status."""
        return {
            'step': self.hrt.step,
            'edges': self.hrt.nnz,
            'lut_entries': len(self.lut.ntoken_to_index),
            'layer_hllsets': self._layer_hllsets.summary(),
            'head_commit': self._head_commit_id,
            'current_branch': self.store.get_current_branch(),
            'store_stats': self.store.stats()
        }
    
    def summary(self) -> str:
        """Get human-readable summary."""
        s = self.status()
        lhs = s['layer_hllsets']
        lines = [
            f"ManifoldOS Status",
            f"  Step: {s['step']}",
            f"  Edges: {s['edges']}",
            f"  LUT entries: {s['lut_entries']}",
            f"  LayerHLLSets: L0={lhs['L0']}, L1={lhs['L1']}, L2={lhs['L2']}, START={lhs['START']}",
            f"  HEAD: {s['head_commit'][:8] if s['head_commit'] else 'None'}",
            f"  Branch: {s['current_branch'] or 'detached'}",
            f"  Store: {s['store_stats']}"
        ]
        return "\n".join(lines)
    
    # -------------------------------------------------------------------------
    # Branch Operations (Git-like)
    # -------------------------------------------------------------------------
    
    def create_branch(self, branch_name: str) -> bool:
        """
        Create a new branch at current HEAD.
        
        Args:
            branch_name: Name of the new branch
        
        Returns:
            True if successful
        """
        return self.store.create_branch(branch_name, self._head_commit_id)
    
    def delete_branch(self, branch_name: str) -> bool:
        """Delete a branch."""
        return self.store.delete_branch(branch_name)
    
    def list_branches(self) -> Dict[str, str]:
        """List all branches with their commit IDs."""
        return self.store.list_branches()
    
    def switch_branch(self, branch_name: str) -> bool:
        """
        Switch to a different branch.
        
        Args:
            branch_name: Branch to switch to
        
        Returns:
            True if successful
        """
        commit_id = self.store.switch_branch(branch_name)
        if commit_id:
            return self.checkout(commit_id)
        return False
    
    def merge(self, branch_name: str, message: str = "") -> Optional[str]:
        """
        Merge a branch into current HEAD.
        
        Args:
            branch_name: Branch to merge
            message: Optional merge message
        
        Returns:
            New merge commit ID, or None if failed
        """
        commit_id = self.store.merge_branch(branch_name, self.sparse_config, message)
        if commit_id:
            # Reload merged state
            self.checkout(commit_id)
            # Reload LUT (may have new entries from merged branch)
            self.lut = self.store.load_lut_to_memory(self.sparse_config)
        return commit_id
    
    # -------------------------------------------------------------------------
    # Parallel Ingestion
    # -------------------------------------------------------------------------
    
    def create_worker(self) -> 'ManifoldWorker':
        """
        Create an isolated worker for parallel ingestion.
        
        Workers have their own HRT and can be merged back later.
        """
        return ManifoldWorker(
            config=self.config,
            lut=self.lut,  # Share LUT reference
            parent_hrt=self.hrt,
            parent_W=self.W
        )
    
    def merge_worker(self, worker: 'ManifoldWorker', source: str = "worker") -> str:
        """
        Merge a worker's HRT into main.
        
        Args:
            worker: ManifoldWorker to merge
            source: Source identifier for commit
        
        Returns:
            Commit ID of merge
        """
        # Merge HRTs, W, and LayerHLLSets
        merged_hrt = self.store.merge_hrts(self.hrt, worker.hrt, self.sparse_config)
        merged_W = self.store.merge_w_matrices(self.W, worker.W)
        merged_lhs = self.store.merge_layer_hllsets(
            self._layer_hllsets,
            worker._layer_hllsets,
            self.sparse_config
        )
        
        # Update state
        self.hrt = merged_hrt
        self.W = merged_W
        self._layer_hllsets = merged_lhs
        
        # Commit merge
        commit_id = self.commit(f"merge_worker:{source}", "parallel_merge")
        
        return commit_id
    
    def parallel_ingest(
        self,
        texts: List[Tuple[str, str]],
        num_workers: int = 4
    ) -> List[str]:
        """
        Ingest multiple texts in parallel using workers.
        
        Args:
            texts: List of (text, source) tuples
            num_workers: Number of parallel workers
        
        Returns:
            List of commit IDs from merges
        """
        import concurrent.futures
        from itertools import islice
        
        def chunk_list(lst, n):
            """Split list into n roughly equal chunks."""
            k, m = divmod(len(lst), n)
            return [lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]
        
        chunks = chunk_list(texts, num_workers)
        commit_ids = []
        
        # Create workers and process in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            
            for i, chunk in enumerate(chunks):
                if not chunk:
                    continue
                worker = self.create_worker()
                future = executor.submit(
                    self._worker_ingest,
                    worker,
                    chunk,
                    f"worker_{i}"
                )
                futures.append((future, worker))
            
            # Merge workers sequentially (to avoid conflicts)
            for future, worker in futures:
                future.result()  # Wait for completion
                commit_id = self.merge_worker(worker, f"parallel_batch")
                commit_ids.append(commit_id)
        
        return commit_ids
    
    def _worker_ingest(
        self,
        worker: 'ManifoldWorker',
        texts: List[Tuple[str, str]],
        worker_name: str
    ):
        """Worker function for parallel ingestion."""
        for text, source in texts:
            worker.ingest(text, source)
    
    # -------------------------------------------------------------------------
    # Remote Sync
    # -------------------------------------------------------------------------
    
    def push(self, remote: 'ManifoldOS') -> Dict[str, Any]:
        """
        Push changes to a remote ManifoldOS.
        
        Args:
            remote: Remote ManifoldOS instance
        
        Returns:
            Sync summary
        """
        return self.store.sync_to(remote.store)
    
    def pull(self, remote: 'ManifoldOS') -> Dict[str, Any]:
        """
        Pull changes from a remote ManifoldOS.
        
        Args:
            remote: Remote ManifoldOS instance
        
        Returns:
            Sync summary
        """
        result = self.store.sync_from(remote.store, self.sparse_config)
        
        # Reload state after pull
        if result['new_head']:
            self.checkout(result['new_head'])
            self.lut = self.store.load_lut_to_memory(self.sparse_config)
        
        return result
    
    # -------------------------------------------------------------------------
    # Context Manager
    # -------------------------------------------------------------------------
    
    def close(self):
        """Close store connection."""
        self.store.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()


# =============================================================================
# ManifoldWorker - For Parallel Ingestion
# =============================================================================

class ManifoldWorker:
    """
    Isolated worker for parallel ingestion.
    
    Has its own HRT and LayerHLLSets that can be merged back to main ManifoldOS.
    Shares LUT with parent (thread-safe via immutable operations).
    """
    
    def __init__(
        self,
        config: ManifoldOSConfig,
        lut: LookupTable,
        parent_hrt: SparseHRT3D,
        parent_W: Dict[int, Dict[int, Dict[int, float]]]
    ):
        self.config = config
        self.sparse_config = config.sparse_config
        self.lut = lut  # Shared reference
        
        # Start with empty HRT (will merge with parent later)
        empty_am = SparseAM3D.from_edges(self.sparse_config, [])
        empty_lattice = SparseLattice3D.from_sparse_am(empty_am)
        self.hrt = SparseHRT3D(
            am=empty_am,
            lattice=empty_lattice,
            config=self.sparse_config,
            lut=frozenset(),
            step=0
        )
        self.W: Dict[int, Dict[int, Dict[int, float]]] = {
            n: {} for n in range(config.max_n)
        }
        
        # Worker has its own LayerHLLSets for O(1) layer classification
        self._layer_hllsets = LayerHLLSets.empty(self.sparse_config.p_bits)
        
        self.texts_processed = 0
    
    def ingest(self, text: str, source: str = "input") -> bool:
        """
        Ingest text into worker's isolated HRT.
        
        Args:
            text: Text content
            source: Source identifier
        
        Returns:
            True if processed
        """
        if not text.strip():
            return False
        
        # Process through unified pipeline
        result = unified_process(
            text,
            self.hrt,
            self.W,
            self.sparse_config,
            self.lut,
            self.config.max_n
        )
        
        # Update worker state
        self.hrt = result.merged_hrt
        self.W = build_w_from_am(self.hrt.am, self.sparse_config)
        
        # Update LayerHLLSets from new edges (both input AND context)
        from .mf_algebra import update_layer_hllsets
        start_indices = {b.to_index(self.sparse_config) for b in result.input_basics[:1]} if result.input_basics else set()
        # Combine input_edges (from input tokens) and context_edges (from W extension)
        all_edges = list(result.input_edges) + list(result.context_edges)
        self._layer_hllsets = update_layer_hllsets(
            all_edges,
            self._layer_hllsets,
            start_indices
        )
        
        self.texts_processed += 1
        
        return True
    
    def status(self) -> Dict[str, Any]:
        """Get worker status."""
        return {
            'edges': self.hrt.nnz,
            'step': self.hrt.step,
            'texts_processed': self.texts_processed,
            'layer_hllsets': self._layer_hllsets.summary()
        }


# =============================================================================
# Factory Functions
# =============================================================================

def create_manifold_os(
    db_path: str = ":memory:",
    config: Optional[ManifoldOSConfig] = None
) -> ManifoldOS:
    """Create a ManifoldOS instance."""
    return ManifoldOS(db_path, config)


def create_memory_manifold() -> ManifoldOS:
    """Create an in-memory ManifoldOS instance."""
    return ManifoldOS(":memory:")


def create_persistent_manifold(db_path: str) -> ManifoldOS:
    """Create a file-backed ManifoldOS instance."""
    return ManifoldOS(db_path)


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    'ManifoldOS',
    'ManifoldOSConfig',
    'ManifoldWorker',
    'TextFilePerceptron',
    'create_manifold_os',
    'create_memory_manifold',
    'create_persistent_manifold',
]
