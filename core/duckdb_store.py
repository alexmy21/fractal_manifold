"""
DuckDB Store: Persistent Backend for ManifoldOS
================================================

DuckDB-backed storage for:
- HRT state (SparseAM3D, SparseLattice3D)
- W matrix (transition probabilities)
- LUT (Lookup Table)
- Commits (version history)
- Refs (branch management)

Architecture:
┌─────────────────────────────────────────────────────────────────────────────┐
│                          DuckDBStore                                        │
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                         commits table                                 │  │
│  │  commit_id → {parent_id, timestamp, source, perceptron, step, blobs}  │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                         blobs table                                   │  │
│  │  blob_id → {blob_type, data, created_at}                              │  │
│  │  Content-addressed (deduplicated by hash)                             │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                       lut_entries table                               │  │
│  │  (idx, layer, ntoken) - Token recovery mapping                        │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                         refs table                                    │  │
│  │  ref_name → commit_id (HEAD, branches)                                │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘

Usage:
    store = DuckDBStore("manifold.duckdb")
    
    # Store HRT and W
    commit_id = store.commit(hrt, W, "source.txt", "p_text")
    
    # Retrieve
    hrt, W = store.checkout(commit_id, config)
    
    # LUT operations
    store.store_lut_entry(idx, layer, ("hello", "world"))
    ntokens = store.get_ntokens_at_index(idx)
"""

from __future__ import annotations
from typing import Dict, List, Optional, Set, Tuple, Any, TYPE_CHECKING
from dataclasses import dataclass
import json
import pickle
import time

try:
    import duckdb
except ImportError:
    raise ImportError("DuckDB is required. Install with: pip install duckdb")

from .hllset import compute_sha1
from .mf_algebra import (
    Sparse3DConfig,
    SparseHRT3D,
    SparseAM3D,
    SparseLattice3D,
    Edge3D,
    LookupTable,
    LayerHLLSets,
)


# =============================================================================
# DuckDB Store
# =============================================================================

class DuckDBStore:
    """
    DuckDB-backed persistent store for ManifoldOS.
    
    Features:
    - Content-addressed blob storage (deduplicated)
    - Commit history with parent tracking
    - LUT persistence
    - Ref management (HEAD, branches)
    """
    
    def __init__(self, db_path: str = ":memory:"):
        """
        Initialize DuckDB store.
        
        Args:
            db_path: Path to database file, or ":memory:" for in-memory
        """
        self.db_path = db_path
        self.conn = duckdb.connect(db_path)
        self._init_schema()
    
    def _init_schema(self):
        """Create tables if they don't exist."""
        # Commits table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS commits (
                commit_id VARCHAR PRIMARY KEY,
                parent_id VARCHAR,
                timestamp DOUBLE,
                source VARCHAR,
                perceptron VARCHAR,
                step_number INTEGER,
                hrt_blob_id VARCHAR,
                w_blob_id VARCHAR,
                layer_hllsets_blob_id VARCHAR,
                metadata JSON
            )
        """)
        
        # Add column to existing tables (migration)
        try:
            self.conn.execute("ALTER TABLE commits ADD COLUMN layer_hllsets_blob_id VARCHAR")
        except:
            pass  # Column already exists
        
        # LUT entries table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS lut_entries (
                idx INTEGER,
                layer INTEGER,
                ntoken VARCHAR,
                ntoken_hash VARCHAR,
                PRIMARY KEY (idx, layer, ntoken)
            )
        """)
        
        # Blobs table (content-addressed storage)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS blobs (
                blob_id VARCHAR PRIMARY KEY,
                blob_type VARCHAR,
                data BLOB,
                created_at DOUBLE
            )
        """)
        
        # Refs table (branches)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS refs (
                ref_name VARCHAR PRIMARY KEY,
                commit_id VARCHAR,
                updated_at DOUBLE
            )
        """)
        
        # Create indexes for faster lookups
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_commits_parent ON commits(parent_id)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_commits_timestamp ON commits(timestamp)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_lut_ntoken ON lut_entries(ntoken)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_lut_idx ON lut_entries(idx)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_blobs_type ON blobs(blob_type)")
    
    # -------------------------------------------------------------------------
    # Blob Operations (Content-Addressed Storage)
    # -------------------------------------------------------------------------
    
    def store_blob(self, data: bytes, blob_type: str) -> str:
        """
        Store binary data, return content hash (deduplicated).
        
        Args:
            data: Binary data to store
            blob_type: Type identifier ('hrt', 'w_matrix', etc.)
        
        Returns:
            Content hash (blob_id)
        """
        blob_id = compute_sha1(data)
        
        # Check if already exists (deduplication)
        exists = self.conn.execute(
            "SELECT 1 FROM blobs WHERE blob_id = ?", [blob_id]
        ).fetchone()
        
        if not exists:
            self.conn.execute(
                "INSERT INTO blobs (blob_id, blob_type, data, created_at) VALUES (?, ?, ?, ?)",
                [blob_id, blob_type, data, time.time()]
            )
        
        return blob_id
    
    def fetch_blob(self, blob_id: str) -> Optional[bytes]:
        """Fetch blob data by ID."""
        result = self.conn.execute(
            "SELECT data FROM blobs WHERE blob_id = ?", [blob_id]
        ).fetchone()
        return result[0] if result else None
    
    def has_blob(self, blob_id: str) -> bool:
        """Check if blob exists."""
        result = self.conn.execute(
            "SELECT 1 FROM blobs WHERE blob_id = ?", [blob_id]
        ).fetchone()
        return result is not None
    
    # -------------------------------------------------------------------------
    # HRT Serialization
    # -------------------------------------------------------------------------
    
    def store_hrt(self, hrt: SparseHRT3D) -> str:
        """
        Serialize and store HRT, return blob ID.
        
        Stores:
        - AM edges
        - Config parameters
        - Step number
        """
        # Get all edges from the tensor - use edges() method which returns List[Edge3D]
        all_edges = hrt.am.tensor.edges()
        # Convert Edge3D to tuples for serialization
        edge_tuples = [(e.n, e.row, e.col, e.value) for e in all_edges]
        
        data = pickle.dumps({
            'am_edges': edge_tuples,
            'config': {
                'p_bits': hrt.config.p_bits,
                'h_bits': hrt.config.h_bits,
                'max_n': hrt.config.max_n,
                'dimension': hrt.config.dimension,
            },
            'step': hrt.step,
        })
        return self.store_blob(data, 'hrt')
    
    def fetch_hrt(self, blob_id: str, config: Sparse3DConfig) -> Optional[SparseHRT3D]:
        """Fetch and deserialize HRT."""
        data = self.fetch_blob(blob_id)
        if not data:
            return None
        
        obj = pickle.loads(data)
        # Edge tuples are (n, row, col, value)
        edges = [Edge3D(n=e[0], row=e[1], col=e[2], value=e[3]) for e in obj['am_edges']]
        am = SparseAM3D.from_edges(config, edges)
        lattice = SparseLattice3D.from_sparse_am(am)
        
        return SparseHRT3D(
            am=am,
            lattice=lattice,
            config=config,
            lut=frozenset(),
            step=obj['step']
        )
    
    def store_w(self, W: Dict[int, Dict[int, Dict[int, float]]]) -> str:
        """Serialize and store W matrix, return blob ID."""
        data = pickle.dumps(W)
        return self.store_blob(data, 'w_matrix')
    
    def fetch_w(self, blob_id: str) -> Optional[Dict[int, Dict[int, Dict[int, float]]]]:
        """Fetch and deserialize W matrix."""
        data = self.fetch_blob(blob_id)
        return pickle.loads(data) if data else None
    
    # -------------------------------------------------------------------------
    # LayerHLLSets Serialization
    # -------------------------------------------------------------------------
    
    def store_layer_hllsets(self, layer_hllsets: 'LayerHLLSets') -> str:
        """
        Serialize and store LayerHLLSets, return blob ID.
        
        LayerHLLSets contains:
        - L0, L1, L2: HLLSets for each n-gram layer
        - START: HLLSet for START followers
        
        These enable O(1) layer classification and START extraction.
        """
        from .mf_algebra import LayerHLLSets
        
        # Serialize each HLLSet's registers using dump_numpy()
        data = pickle.dumps({
            'L0_registers': layer_hllsets.L0.dump_numpy().tobytes(),
            'L1_registers': layer_hllsets.L1.dump_numpy().tobytes(),
            'L2_registers': layer_hllsets.L2.dump_numpy().tobytes(),
            'START_registers': layer_hllsets.START.dump_numpy().tobytes(),
            'p_bits': layer_hllsets.p_bits,
        })
        return self.store_blob(data, 'layer_hllsets')
    
    def fetch_layer_hllsets(
        self,
        blob_id: str,
        config: Sparse3DConfig
    ) -> Optional['LayerHLLSets']:
        """Fetch and deserialize LayerHLLSets."""
        from .mf_algebra import LayerHLLSets
        from .hllset import HLLSet
        import numpy as np
        
        data = self.fetch_blob(blob_id)
        if not data:
            return None
        
        obj = pickle.loads(data)
        
        # Get p_bits from stored metadata
        stored_p_bits = obj.get('p_bits', config.p_bits)
        
        # Reconstruct HLLSets from registers
        def hllset_from_registers(reg_bytes: bytes, p_bits: int) -> HLLSet:
            # Restore numpy array from bytes - registers are uint32
            reg_array = np.frombuffer(reg_bytes, dtype=np.uint32)
            
            hll = HLLSet(p_bits=p_bits)
            # Set registers via C core
            hll._core.set_registers(reg_array)
            hll._compute_name()
            return hll
        
        L0 = hllset_from_registers(obj['L0_registers'], stored_p_bits)
        L1 = hllset_from_registers(obj['L1_registers'], stored_p_bits)
        L2 = hllset_from_registers(obj['L2_registers'], stored_p_bits)
        START = hllset_from_registers(obj['START_registers'], stored_p_bits)
        
        return LayerHLLSets(
            L0=L0,
            L1=L1,
            L2=L2,
            START=START,
            p_bits=stored_p_bits,
        )
    
    # -------------------------------------------------------------------------
    # Commit Operations
    # -------------------------------------------------------------------------
    
    def commit(
        self,
        hrt: SparseHRT3D,
        W: Dict[int, Dict[int, Dict[int, float]]],
        source: str,
        perceptron: str,
        parent_id: Optional[str] = None,
        metadata: Optional[Dict] = None,
        layer_hllsets: Optional['LayerHLLSets'] = None
    ) -> str:
        """
        Create a new commit with HRT, W, and LayerHLLSets state.
        
        Args:
            hrt: Current HRT state
            W: Current W matrix
            source: Source identifier (e.g., filename)
            perceptron: Perceptron that processed this
            parent_id: Parent commit ID (None for genesis)
            metadata: Additional metadata
            layer_hllsets: Layer HLLSets (L0, L1, L2, START)
        
        Returns:
            commit_id
        """
        hrt_blob_id = self.store_hrt(hrt)
        w_blob_id = self.store_w(W)
        
        # Store LayerHLLSets if provided
        layer_hllsets_blob_id = None
        if layer_hllsets is not None:
            layer_hllsets_blob_id = self.store_layer_hllsets(layer_hllsets)
        
        # Generate commit ID from content
        content = f"{hrt_blob_id}:{w_blob_id}:{layer_hllsets_blob_id}:{time.time()}:{source}"
        commit_id = compute_sha1(content)
        
        self.conn.execute("""
            INSERT INTO commits 
            (commit_id, parent_id, timestamp, source, perceptron, step_number, hrt_blob_id, w_blob_id, layer_hllsets_blob_id, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            commit_id,
            parent_id,
            time.time(),
            source,
            perceptron,
            hrt.step,
            hrt_blob_id,
            w_blob_id,
            layer_hllsets_blob_id,
            json.dumps(metadata or {})
        ])
        
        return commit_id
    
    def get_commit(self, commit_id: str) -> Optional[Dict[str, Any]]:
        """Get commit metadata by ID."""
        result = self.conn.execute(
            "SELECT * FROM commits WHERE commit_id = ?", [commit_id]
        ).fetchone()
        
        if not result:
            return None
        
        return {
            'commit_id': result[0],
            'parent_id': result[1],
            'timestamp': result[2],
            'source': result[3],
            'perceptron': result[4],
            'step_number': result[5],
            'hrt_blob_id': result[6],
            'w_blob_id': result[7],
            'layer_hllsets_blob_id': result[8] if len(result) > 8 else None,
            'metadata': json.loads(result[9]) if len(result) > 9 and result[9] else {}
        }
    
    def has_commit(self, commit_id: str) -> bool:
        """Check if commit exists."""
        result = self.conn.execute(
            "SELECT 1 FROM commits WHERE commit_id = ?", [commit_id]
        ).fetchone()
        return result is not None
    
    def checkout(
        self,
        commit_id: str,
        config: Sparse3DConfig
    ) -> Tuple[Optional[SparseHRT3D], Optional[Dict[int, Dict[int, Dict[int, float]]]], Optional['LayerHLLSets']]:
        """
        Checkout HRT, W, and LayerHLLSets from a commit.
        
        Args:
            commit_id: Commit to checkout
            config: Sparse3DConfig for HRT reconstruction
        
        Returns:
            (hrt, W, layer_hllsets) tuple, or (None, None, None) if commit not found
        """
        commit = self.get_commit(commit_id)
        if not commit:
            return None, None, None
        
        hrt = self.fetch_hrt(commit['hrt_blob_id'], config)
        W = self.fetch_w(commit['w_blob_id'])
        
        # Fetch LayerHLLSets if available
        layer_hllsets = None
        if commit.get('layer_hllsets_blob_id'):
            layer_hllsets = self.fetch_layer_hllsets(commit['layer_hllsets_blob_id'], config)
        
        return hrt, W, layer_hllsets
    
    # -------------------------------------------------------------------------
    # Refs (Branch Management)
    # -------------------------------------------------------------------------
    
    def set_ref(self, ref_name: str, commit_id: str):
        """Set a ref to point to a commit."""
        self.conn.execute("""
            INSERT OR REPLACE INTO refs (ref_name, commit_id, updated_at)
            VALUES (?, ?, ?)
        """, [ref_name, commit_id, time.time()])
    
    def get_ref(self, ref_name: str) -> Optional[str]:
        """Get commit ID for a ref."""
        result = self.conn.execute(
            "SELECT commit_id FROM refs WHERE ref_name = ?", [ref_name]
        ).fetchone()
        return result[0] if result else None
    
    def list_refs(self) -> Dict[str, str]:
        """List all refs and their commit IDs."""
        results = self.conn.execute(
            "SELECT ref_name, commit_id FROM refs"
        ).fetchall()
        return {r[0]: r[1] for r in results}
    
    def delete_ref(self, ref_name: str) -> bool:
        """Delete a ref."""
        self.conn.execute(
            "DELETE FROM refs WHERE ref_name = ?", [ref_name]
        )
        return True
    
    def get_head(
        self,
        config: Sparse3DConfig
    ) -> Tuple[Optional[SparseHRT3D], Optional[Dict[int, Dict[int, Dict[int, float]]]], Optional['LayerHLLSets']]:
        """Get the current HEAD state (HRT, W, LayerHLLSets)."""
        head_commit = self.get_ref('HEAD')
        if not head_commit:
            return None, None, None
        return self.checkout(head_commit, config)
    
    # -------------------------------------------------------------------------
    # LUT Operations
    # -------------------------------------------------------------------------
    
    def store_lut_entry(self, idx: int, layer: int, ntoken: Tuple[str, ...]):
        """Store a LUT entry."""
        ntoken_str = " ".join(ntoken)
        ntoken_hash = compute_sha1(ntoken_str)
        
        self.conn.execute("""
            INSERT OR IGNORE INTO lut_entries (idx, layer, ntoken, ntoken_hash)
            VALUES (?, ?, ?, ?)
        """, [idx, layer, ntoken_str, ntoken_hash])
    
    def get_ntokens_at_index(self, idx: int) -> Set[Tuple[int, Tuple[str, ...]]]:
        """Get all (layer, ntoken) pairs at an index."""
        results = self.conn.execute(
            "SELECT layer, ntoken FROM lut_entries WHERE idx = ?", [idx]
        ).fetchall()
        
        return {(row[0], tuple(row[1].split())) for row in results}
    
    def get_ntoken_index(self, ntoken: Tuple[str, ...]) -> Optional[int]:
        """Get index for an ntoken."""
        ntoken_str = " ".join(ntoken)
        result = self.conn.execute(
            "SELECT idx FROM lut_entries WHERE ntoken = ? LIMIT 1", [ntoken_str]
        ).fetchone()
        return result[0] if result else None
    
    def sync_lut_from_memory(self, lut: 'LookupTable'):
        """
        Sync in-memory LUT to DuckDB.
        
        Efficient batch insert using transactions.
        """
        entries = []
        for idx, layer_ntokens in lut.index_to_ntokens.items():
            for layer, ntoken in layer_ntokens:
                ntoken_str = " ".join(ntoken)
                ntoken_hash = compute_sha1(ntoken_str)
                entries.append((idx, layer, ntoken_str, ntoken_hash))
        
        if entries:
            self.conn.executemany("""
                INSERT OR IGNORE INTO lut_entries (idx, layer, ntoken, ntoken_hash)
                VALUES (?, ?, ?, ?)
            """, entries)
    
    def load_lut_to_memory(self, config: Sparse3DConfig) -> 'LookupTable':
        """Load LUT from DuckDB to memory."""
        from .mf_algebra import LookupTable
        
        lut = LookupTable(config=config)
        
        results = self.conn.execute(
            "SELECT idx, layer, ntoken FROM lut_entries"
        ).fetchall()
        
        for idx, layer, ntoken_str in results:
            ntoken = tuple(ntoken_str.split())
            lut.index_to_ntokens[idx].add((layer, ntoken))
            lut.ntoken_to_index[ntoken] = idx
        
        return lut
    
    def lut_count(self) -> int:
        """Get number of LUT entries."""
        result = self.conn.execute("SELECT COUNT(*) FROM lut_entries").fetchone()
        return result[0] if result else 0
    
    # -------------------------------------------------------------------------
    # History & Querying
    # -------------------------------------------------------------------------
    
    def log(self, limit: int = 10, ref_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get recent commits.
        
        Args:
            limit: Maximum number of commits to return
            ref_name: If provided, start from this ref
        
        Returns:
            List of commit metadata dictionaries
        """
        if ref_name:
            start_commit = self.get_ref(ref_name)
            if not start_commit:
                return []
            
            # Walk parent chain
            commits = []
            current = start_commit
            while current and len(commits) < limit:
                commit = self.get_commit(current)
                if commit:
                    commits.append(commit)
                    current = commit['parent_id']
                else:
                    break
            return commits
        else:
            # All commits by timestamp
            results = self.conn.execute("""
                SELECT commit_id, parent_id, timestamp, source, perceptron, step_number
                FROM commits
                ORDER BY timestamp DESC
                LIMIT ?
            """, [limit]).fetchall()
            
            return [{
                'commit_id': r[0],
                'parent_id': r[1],
                'timestamp': r[2],
                'source': r[3],
                'perceptron': r[4],
                'step_number': r[5]
            } for r in results]
    
    def find_commits_by_source(self, source_pattern: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Find commits matching source pattern."""
        results = self.conn.execute("""
            SELECT commit_id, parent_id, timestamp, source, perceptron, step_number
            FROM commits
            WHERE source LIKE ?
            ORDER BY timestamp DESC
            LIMIT ?
        """, [f"%{source_pattern}%", limit]).fetchall()
        
        return [{
            'commit_id': r[0],
            'parent_id': r[1],
            'timestamp': r[2],
            'source': r[3],
            'perceptron': r[4],
            'step_number': r[5]
        } for r in results]
    
    # -------------------------------------------------------------------------
    # Statistics & Maintenance
    # -------------------------------------------------------------------------
    
    def stats(self) -> Dict[str, Any]:
        """Get store statistics."""
        commits = self.conn.execute("SELECT COUNT(*) FROM commits").fetchone()[0]
        blobs = self.conn.execute("SELECT COUNT(*) FROM blobs").fetchone()[0]
        lut_entries = self.conn.execute("SELECT COUNT(*) FROM lut_entries").fetchone()[0]
        refs = self.conn.execute("SELECT COUNT(*) FROM refs").fetchone()[0]
        
        # Blob storage by type (use OCTET_LENGTH for BLOB data)
        blob_types = self.conn.execute("""
            SELECT blob_type, COUNT(*), SUM(OCTET_LENGTH(data))
            FROM blobs
            GROUP BY blob_type
        """).fetchall()
        
        return {
            'commits': commits,
            'blobs': blobs,
            'lut_entries': lut_entries,
            'refs': refs,
            'blob_types': {r[0]: {'count': r[1], 'bytes': r[2]} for r in blob_types},
            'db_path': self.db_path
        }
    
    def vacuum(self):
        """Optimize database storage."""
        self.conn.execute("VACUUM")
    
    # -------------------------------------------------------------------------
    # Branch Operations (Git-like)
    # -------------------------------------------------------------------------
    
    def create_branch(self, branch_name: str, commit_id: Optional[str] = None) -> bool:
        """
        Create a new branch pointing to a commit.
        
        Args:
            branch_name: Name of the new branch
            commit_id: Commit to point to (defaults to HEAD)
        
        Returns:
            True if successful
        """
        if commit_id is None:
            commit_id = self.get_ref('HEAD')
            if not commit_id:
                return False
        
        # Check if commit exists
        if not self.get_commit(commit_id):
            return False
        
        # Create branch ref
        self.set_ref(f"refs/heads/{branch_name}", commit_id)
        return True
    
    def delete_branch(self, branch_name: str) -> bool:
        """Delete a branch."""
        ref_name = f"refs/heads/{branch_name}"
        if not self.get_ref(ref_name):
            return False
        return self.delete_ref(ref_name)
    
    def list_branches(self) -> Dict[str, str]:
        """List all branches and their commit IDs."""
        results = self.conn.execute("""
            SELECT ref_name, commit_id 
            FROM refs 
            WHERE ref_name LIKE 'refs/heads/%'
        """).fetchall()
        
        return {r[0].replace('refs/heads/', ''): r[1] for r in results}
    
    def get_branch_commit(self, branch_name: str) -> Optional[str]:
        """Get commit ID for a branch."""
        return self.get_ref(f"refs/heads/{branch_name}")
    
    def switch_branch(self, branch_name: str) -> Optional[str]:
        """
        Switch HEAD to a branch.
        
        Args:
            branch_name: Branch to switch to
        
        Returns:
            The commit ID, or None if branch doesn't exist
        """
        commit_id = self.get_branch_commit(branch_name)
        if commit_id:
            self.set_ref('HEAD', commit_id)
            self.set_ref('HEAD_BRANCH', branch_name)  # Track current branch
        return commit_id
    
    def get_current_branch(self) -> Optional[str]:
        """Get the name of the current branch."""
        return self.get_ref('HEAD_BRANCH')
    
    # -------------------------------------------------------------------------
    # Merge Operations
    # -------------------------------------------------------------------------
    
    def find_common_ancestor(
        self,
        commit_a: str,
        commit_b: str
    ) -> Optional[str]:
        """
        Find the common ancestor of two commits.
        
        Uses simple BFS - walks both commit chains and finds intersection.
        """
        ancestors_a = set()
        ancestors_b = set()
        
        # Walk commit A's history
        current = commit_a
        while current:
            ancestors_a.add(current)
            commit = self.get_commit(current)
            current = commit['parent_id'] if commit else None
        
        # Walk commit B's history until we find common ancestor
        current = commit_b
        while current:
            if current in ancestors_a:
                return current
            ancestors_b.add(current)
            commit = self.get_commit(current)
            current = commit['parent_id'] if commit else None
        
        return None
    
    def get_commits_since(
        self,
        start_commit: str,
        end_commit: str
    ) -> List[str]:
        """
        Get list of commits from start to end (exclusive of start).
        
        Returns commits in chronological order (oldest first).
        """
        commits = []
        current = end_commit
        
        while current and current != start_commit:
            commits.append(current)
            commit = self.get_commit(current)
            current = commit['parent_id'] if commit else None
        
        commits.reverse()  # Oldest first
        return commits
    
    def merge_hrts(
        self,
        hrt_a: SparseHRT3D,
        hrt_b: SparseHRT3D,
        config: Sparse3DConfig
    ) -> SparseHRT3D:
        """
        Merge two HRTs by combining their edges.
        
        HRT properties (IICA) make this straightforward:
        - Immutable: We create a new HRT
        - Idempotent: Duplicate edges just add values
        - Content-addressed: Same content = same hash
        """
        # Collect all edges from both HRTs
        edges_a = hrt_a.am.tensor.edges()
        edges_b = hrt_b.am.tensor.edges()
        
        # Combine edges (add values for duplicates)
        edge_dict: Dict[Tuple[int, int, int], float] = {}
        
        for edge in edges_a:
            key = (edge.n, edge.row, edge.col)
            edge_dict[key] = edge_dict.get(key, 0.0) + edge.value
        
        for edge in edges_b:
            key = (edge.n, edge.row, edge.col)
            edge_dict[key] = edge_dict.get(key, 0.0) + edge.value
        
        # Create merged HRT
        merged_edges = [
            Edge3D(n=k[0], row=k[1], col=k[2], value=v)
            for k, v in edge_dict.items()
        ]
        
        merged_am = SparseAM3D.from_edges(config, merged_edges)
        merged_lattice = SparseLattice3D.from_sparse_am(merged_am)
        
        return SparseHRT3D(
            am=merged_am,
            lattice=merged_lattice,
            config=config,
            lut=frozenset(),
            step=max(hrt_a.step, hrt_b.step) + 1
        )
    
    def merge_w_matrices(
        self,
        W_a: Dict[int, Dict[int, Dict[int, float]]],
        W_b: Dict[int, Dict[int, Dict[int, float]]]
    ) -> Dict[int, Dict[int, Dict[int, float]]]:
        """
        Merge two W matrices by averaging overlapping entries.
        """
        W_merged: Dict[int, Dict[int, Dict[int, float]]] = {}
        
        all_layers = set(W_a.keys()) | set(W_b.keys())
        
        for n in all_layers:
            W_merged[n] = {}
            rows_a = W_a.get(n, {})
            rows_b = W_b.get(n, {})
            all_rows = set(rows_a.keys()) | set(rows_b.keys())
            
            for row in all_rows:
                W_merged[n][row] = {}
                cols_a = rows_a.get(row, {})
                cols_b = rows_b.get(row, {})
                all_cols = set(cols_a.keys()) | set(cols_b.keys())
                
                for col in all_cols:
                    val_a = cols_a.get(col, 0.0)
                    val_b = cols_b.get(col, 0.0)
                    
                    if col in cols_a and col in cols_b:
                        # Average for overlapping entries
                        W_merged[n][row][col] = (val_a + val_b) / 2
                    else:
                        # Take the non-zero value
                        W_merged[n][row][col] = val_a if val_a > 0 else val_b
        
        return W_merged
    
    def merge_layer_hllsets(
        self,
        lhs_a: Optional['LayerHLLSets'],
        lhs_b: Optional['LayerHLLSets'],
        config: Sparse3DConfig
    ) -> 'LayerHLLSets':
        """
        Merge two LayerHLLSets by union.
        
        Each layer HLLSet (L0, L1, L2, START) is merged via union.
        This enables O(1) layer classification after merge.
        """
        from .mf_algebra import LayerHLLSets
        
        # Handle None cases
        if lhs_a is None and lhs_b is None:
            return LayerHLLSets.empty(config.p_bits)
        if lhs_a is None:
            return lhs_b
        if lhs_b is None:
            return lhs_a
        
        # Merge via union (uses HLLSet.union internally)
        return lhs_a.merge(lhs_b)
    
    def merge_branch(
        self,
        branch_name: str,
        config: Sparse3DConfig,
        message: str = ""
    ) -> Optional[str]:
        """
        Merge a branch into HEAD.
        
        Args:
            branch_name: Branch to merge
            config: Sparse3DConfig
            message: Optional merge message
        
        Returns:
            New merge commit ID, or None if merge failed
        """
        head_commit = self.get_ref('HEAD')
        branch_commit = self.get_branch_commit(branch_name)
        
        if not head_commit or not branch_commit:
            return None
        
        if head_commit == branch_commit:
            return head_commit  # Already merged
        
        # Get HRT states (now includes LayerHLLSets)
        hrt_head, W_head, lhs_head = self.checkout(head_commit, config)
        hrt_branch, W_branch, lhs_branch = self.checkout(branch_commit, config)
        
        if hrt_head is None or hrt_branch is None:
            return None
        
        # Merge HRTs, W matrices, and LayerHLLSets
        merged_hrt = self.merge_hrts(hrt_head, hrt_branch, config)
        merged_W = self.merge_w_matrices(W_head, W_branch)
        merged_lhs = self.merge_layer_hllsets(lhs_head, lhs_branch, config)
        
        # Create merge commit
        commit_id = self.commit(
            merged_hrt,
            merged_W,
            source=f"merge:{branch_name}",
            perceptron="merge",
            parent_id=head_commit,
            metadata={
                'merge_type': 'branch',
                'merged_branch': branch_name,
                'merged_commit': branch_commit,
                'message': message
            },
            layer_hllsets=merged_lhs
        )
        
        # Update HEAD
        self.set_ref('HEAD', commit_id)
        
        return commit_id
    
    # -------------------------------------------------------------------------
    # Remote Sync (Simulated)
    # -------------------------------------------------------------------------
    
    def export_commits(
        self,
        since_commit: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Export commits and their blobs for remote sync.
        
        Args:
            since_commit: Export commits after this one
            limit: Maximum commits to export
        
        Returns:
            List of commit data with embedded blob data
        """
        if since_commit:
            commits = self.get_commits_since(since_commit, self.get_ref('HEAD') or '')
        else:
            commits = [c['commit_id'] for c in self.log(limit)]
        
        export_data = []
        for commit_id in commits[:limit]:
            commit = self.get_commit(commit_id)
            if not commit:
                continue
            
            # Include blob data
            hrt_data = self.fetch_blob(commit['hrt_blob_id'])
            w_data = self.fetch_blob(commit['w_blob_id'])
            
            export_data.append({
                'commit': commit,
                'hrt_blob': {
                    'id': commit['hrt_blob_id'],
                    'data': hrt_data.hex() if hrt_data else None
                },
                'w_blob': {
                    'id': commit['w_blob_id'],
                    'data': w_data.hex() if w_data else None
                }
            })
        
        return export_data
    
    def import_commits(
        self,
        export_data: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Import commits from export data.
        
        Args:
            export_data: Data from export_commits()
        
        Returns:
            List of imported commit IDs
        """
        imported = []
        
        for item in export_data:
            commit = item['commit']
            
            # Skip if commit already exists
            if self.get_commit(commit['commit_id']):
                continue
            
            # Import blobs
            if item['hrt_blob']['data']:
                hrt_data = bytes.fromhex(item['hrt_blob']['data'])
                self.store_blob(hrt_data, 'hrt')
            
            if item['w_blob']['data']:
                w_data = bytes.fromhex(item['w_blob']['data'])
                self.store_blob(w_data, 'w_matrix')
            
            # Insert commit
            self.conn.execute("""
                INSERT OR IGNORE INTO commits 
                (commit_id, parent_id, timestamp, source, perceptron, step_number, hrt_blob_id, w_blob_id, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                commit['commit_id'],
                commit['parent_id'],
                commit['timestamp'],
                commit['source'],
                commit['perceptron'],
                commit['step_number'],
                commit['hrt_blob_id'],
                commit['w_blob_id'],
                json.dumps(commit.get('metadata', {}))
            ])
            
            imported.append(commit['commit_id'])
        
        return imported
    
    def export_lut(self) -> List[Tuple[int, int, str]]:
        """Export all LUT entries."""
        results = self.conn.execute(
            "SELECT idx, layer, ntoken FROM lut_entries"
        ).fetchall()
        return [(r[0], r[1], r[2]) for r in results]
    
    def import_lut(self, entries: List[Tuple[int, int, str]]):
        """Import LUT entries."""
        if entries:
            self.conn.executemany("""
                INSERT OR IGNORE INTO lut_entries (idx, layer, ntoken, ntoken_hash)
                VALUES (?, ?, ?, ?)
            """, [(idx, layer, ntoken, compute_sha1(ntoken)) for idx, layer, ntoken in entries])
    
    def sync_from(
        self,
        remote_store: 'DuckDBStore',
        config: Sparse3DConfig
    ) -> Dict[str, Any]:
        """
        Pull changes from a remote store.
        
        Args:
            remote_store: Remote DuckDB store
            config: Sparse3DConfig
        
        Returns:
            Sync summary
        """
        local_head = self.get_ref('HEAD')
        
        # Find what we need to import
        if local_head:
            # Export from remote since our HEAD
            remote_data = remote_store.export_commits(since_commit=local_head)
        else:
            # Fresh sync - get everything
            remote_data = remote_store.export_commits()
        
        # Import commits
        imported = self.import_commits(remote_data)
        
        # Import LUT
        remote_lut = remote_store.export_lut()
        self.import_lut(remote_lut)
        
        # Update HEAD to match remote
        remote_head = remote_store.get_ref('HEAD')
        if remote_head:
            self.set_ref('HEAD', remote_head)
        
        return {
            'commits_imported': len(imported),
            'lut_entries_synced': len(remote_lut),
            'new_head': remote_head
        }
    
    def sync_to(
        self,
        remote_store: 'DuckDBStore'
    ) -> Dict[str, Any]:
        """
        Push changes to a remote store.
        
        Args:
            remote_store: Remote DuckDB store
        
        Returns:
            Sync summary
        """
        remote_head = remote_store.get_ref('HEAD')
        
        # Export our commits since remote HEAD
        if remote_head:
            export_data = self.export_commits(since_commit=remote_head)
        else:
            export_data = self.export_commits()
        
        # Import to remote
        imported = remote_store.import_commits(export_data)
        
        # Sync LUT
        local_lut = self.export_lut()
        remote_store.import_lut(local_lut)
        
        # Update remote HEAD
        local_head = self.get_ref('HEAD')
        if local_head:
            remote_store.set_ref('HEAD', local_head)
        
        return {
            'commits_pushed': len(imported),
            'lut_entries_synced': len(local_lut),
            'remote_head': local_head
        }
    
    def close(self):
        """Close the database connection."""
        self.conn.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()


# =============================================================================
# Factory Functions
# =============================================================================

def create_duckdb_store(db_path: str = ":memory:") -> DuckDBStore:
    """Create a DuckDB store instance."""
    return DuckDBStore(db_path)


def create_memory_store() -> DuckDBStore:
    """Create an in-memory DuckDB store."""
    return DuckDBStore(":memory:")


def create_file_store(path: str) -> DuckDBStore:
    """Create a file-backed DuckDB store."""
    return DuckDBStore(path)


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    'DuckDBStore',
    'create_duckdb_store',
    'create_memory_store',
    'create_file_store',
]
