"""
HRT Persistent Store: Single Store for Complete HRT State
==========================================================

Design Principles:
1. Single store for entire HRT (AM + Lattice + LUT)
2. Git-like commit/push separation
3. Content-addressed storage (deduplication)
4. Supports both memory and disk

Architecture:
┌─────────────────────────────────────────────────────────────────────────────┐
│                          HRTPersistentStore                                 │
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                         Commit Index                                  │  │
│  │  commit_hash → {parent, step, timestamp, message}                     │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                         Content Store                                 │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                    │  │
│  │  │  AM Blobs   │  │Lattice Blobs│  │  LUT Blobs  │                    │  │
│  │  │(tensor data)│  │(HLLSet regs)│  │(token maps) │                    │  │
│  │  └─────────────┘  └─────────────┘  └─────────────┘                    │  │
│  │  content_hash → blob_data (deduplicated)                              │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                           Refs                                        │  │
│  │  main → commit_hash                                                   │  │
│  │  branch_name → commit_hash                                            │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘

Storage Format:
- Each HRT is stored as a commit object pointing to content blobs
- Content blobs are deduplicated by hash (IICA property)
- Refs map branch names to commit hashes
- Supports pack files for efficient storage

Backends:
- FileSystemStore: Directory-based (Git-like)
- DuckDBStore: Embedded SQL database
- MemoryStore: In-memory (testing/temporary)
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Tuple, Any, Iterator
from pathlib import Path
import json
import gzip
import time
import hashlib
import pickle

from .hllset import compute_sha1


# =============================================================================
# SECTION 1: Store Data Types
# =============================================================================

@dataclass(frozen=True)
class CommitObject:
    """
    Commit object in persistent store.
    
    Like Git commit - points to content, has parent.
    """
    commit_hash: str              # Content-addressed hash
    parent_hash: Optional[str]    # Parent commit (None for genesis)
    step_number: int
    timestamp: float
    message: str
    
    # Content references (hashes pointing to blobs)
    am_hash: str
    lattice_hash: str
    lut_hash: str
    config_hash: str
    
    # Optional metadata
    source_info: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'commit_hash': self.commit_hash,
            'parent_hash': self.parent_hash,
            'step_number': self.step_number,
            'timestamp': self.timestamp,
            'message': self.message,
            'am_hash': self.am_hash,
            'lattice_hash': self.lattice_hash,
            'lut_hash': self.lut_hash,
            'config_hash': self.config_hash,
            'source_info': self.source_info
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'CommitObject':
        return cls(
            commit_hash=d['commit_hash'],
            parent_hash=d.get('parent_hash'),
            step_number=d['step_number'],
            timestamp=d['timestamp'],
            message=d['message'],
            am_hash=d['am_hash'],
            lattice_hash=d['lattice_hash'],
            lut_hash=d['lut_hash'],
            config_hash=d['config_hash'],
            source_info=d.get('source_info')
        )


@dataclass 
class Blob:
    """Content-addressed blob."""
    hash: str
    data: bytes
    blob_type: str  # 'am', 'lattice', 'lut', 'config'
    
    @classmethod
    def from_data(cls, data: bytes, blob_type: str) -> 'Blob':
        hash_val = compute_sha1(data)
        return cls(hash=hash_val, data=data, blob_type=blob_type)


# =============================================================================
# SECTION 2: Abstract Store Interface
# =============================================================================

class HRTPersistentStore(ABC):
    """
    Abstract interface for HRT persistent storage.
    
    Single store for complete HRT state.
    Supports Git-like operations: push, fetch, log.
    """
    
    @abstractmethod
    def push_commit(self, commit: CommitObject) -> bool:
        """Push commit object to store."""
        pass
    
    @abstractmethod
    def push_blob(self, blob: Blob) -> bool:
        """Push content blob to store (deduplicated)."""
        pass
    
    @abstractmethod
    def fetch_commit(self, commit_hash: str) -> Optional[CommitObject]:
        """Fetch commit by hash."""
        pass
    
    @abstractmethod
    def fetch_blob(self, blob_hash: str) -> Optional[Blob]:
        """Fetch blob by hash."""
        pass
    
    @abstractmethod
    def get_ref(self, ref_name: str) -> Optional[str]:
        """Get commit hash for ref (e.g., 'main')."""
        pass
    
    @abstractmethod
    def set_ref(self, ref_name: str, commit_hash: str) -> bool:
        """Set ref to point to commit."""
        pass
    
    @abstractmethod
    def list_refs(self) -> Dict[str, str]:
        """List all refs and their commit hashes."""
        pass
    
    @abstractmethod
    def log(self, ref_name: str = "main", limit: int = 10) -> List[CommitObject]:
        """Get commit history from ref."""
        pass
    
    @abstractmethod
    def has_commit(self, commit_hash: str) -> bool:
        """Check if commit exists."""
        pass
    
    @abstractmethod
    def has_blob(self, blob_hash: str) -> bool:
        """Check if blob exists."""
        pass
    
    @abstractmethod
    def close(self):
        """Close store and release resources."""
        pass


# =============================================================================
# SECTION 3: Memory Store (Testing/Temporary)
# =============================================================================

class MemoryHRTStore(HRTPersistentStore):
    """
    In-memory HRT store.
    
    Useful for testing or temporary operations.
    Data is lost when store is closed.
    """
    
    def __init__(self):
        self._commits: Dict[str, CommitObject] = {}
        self._blobs: Dict[str, Blob] = {}
        self._refs: Dict[str, str] = {}
    
    def push_commit(self, commit: CommitObject) -> bool:
        self._commits[commit.commit_hash] = commit
        return True
    
    def push_blob(self, blob: Blob) -> bool:
        if blob.hash not in self._blobs:
            self._blobs[blob.hash] = blob
        return True  # Deduplicated
    
    def fetch_commit(self, commit_hash: str) -> Optional[CommitObject]:
        return self._commits.get(commit_hash)
    
    def fetch_blob(self, blob_hash: str) -> Optional[Blob]:
        return self._blobs.get(blob_hash)
    
    def get_ref(self, ref_name: str) -> Optional[str]:
        return self._refs.get(ref_name)
    
    def set_ref(self, ref_name: str, commit_hash: str) -> bool:
        if commit_hash in self._commits:
            self._refs[ref_name] = commit_hash
            return True
        return False
    
    def list_refs(self) -> Dict[str, str]:
        return dict(self._refs)
    
    def log(self, ref_name: str = "main", limit: int = 10) -> List[CommitObject]:
        commit_hash = self._refs.get(ref_name)
        if not commit_hash:
            return []
        
        history = []
        current = self._commits.get(commit_hash)
        
        while current and len(history) < limit:
            history.append(current)
            if current.parent_hash:
                current = self._commits.get(current.parent_hash)
            else:
                break
        
        return history
    
    def has_commit(self, commit_hash: str) -> bool:
        return commit_hash in self._commits
    
    def has_blob(self, blob_hash: str) -> bool:
        return blob_hash in self._blobs
    
    def close(self):
        self._commits.clear()
        self._blobs.clear()
        self._refs.clear()
    
    @property
    def stats(self) -> Dict[str, int]:
        return {
            'commits': len(self._commits),
            'blobs': len(self._blobs),
            'refs': len(self._refs),
            'blob_bytes': sum(len(b.data) for b in self._blobs.values())
        }


# =============================================================================
# SECTION 4: File System Store (Git-like)
# =============================================================================

class FileSystemHRTStore(HRTPersistentStore):
    """
    File system-based HRT store.
    
    Directory structure:
    store_path/
    ├── commits/
    │   └── ab/
    │       └── cdef123...  # commit objects
    ├── blobs/
    │   └── 01/
    │       └── 234abc...   # content blobs (gzipped)
    ├── refs/
    │   └── heads/
    │       └── main        # ref files
    └── config.json
    """
    
    def __init__(self, store_path: str):
        self.path = Path(store_path)
        self._init_structure()
    
    def _init_structure(self):
        """Initialize directory structure."""
        (self.path / "commits").mkdir(parents=True, exist_ok=True)
        (self.path / "blobs").mkdir(parents=True, exist_ok=True)
        (self.path / "refs" / "heads").mkdir(parents=True, exist_ok=True)
        
        config_file = self.path / "config.json"
        if not config_file.exists():
            config_file.write_text(json.dumps({
                'version': 1,
                'created': time.time()
            }))
    
    def _hash_path(self, hash_val: str, category: str) -> Path:
        """Get path for hash (sharded by first 2 chars)."""
        return self.path / category / hash_val[:2] / hash_val
    
    def push_commit(self, commit: CommitObject) -> bool:
        path = self._hash_path(commit.commit_hash, "commits")
        path.parent.mkdir(parents=True, exist_ok=True)
        
        data = json.dumps(commit.to_dict())
        path.write_text(data)
        return True
    
    def push_blob(self, blob: Blob) -> bool:
        path = self._hash_path(blob.hash, "blobs")
        if path.exists():
            return True  # Deduplicated
        
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Store gzipped with type prefix
        header = f"{blob.blob_type}\n".encode()
        compressed = gzip.compress(header + blob.data)
        path.write_bytes(compressed)
        return True
    
    def fetch_commit(self, commit_hash: str) -> Optional[CommitObject]:
        path = self._hash_path(commit_hash, "commits")
        if not path.exists():
            return None
        
        data = json.loads(path.read_text())
        return CommitObject.from_dict(data)
    
    def fetch_blob(self, blob_hash: str) -> Optional[Blob]:
        path = self._hash_path(blob_hash, "blobs")
        if not path.exists():
            return None
        
        decompressed = gzip.decompress(path.read_bytes())
        newline_pos = decompressed.index(b'\n')
        blob_type = decompressed[:newline_pos].decode()
        data = decompressed[newline_pos + 1:]
        
        return Blob(hash=blob_hash, data=data, blob_type=blob_type)
    
    def get_ref(self, ref_name: str) -> Optional[str]:
        ref_path = self.path / "refs" / "heads" / ref_name
        if ref_path.exists():
            return ref_path.read_text().strip()
        return None
    
    def set_ref(self, ref_name: str, commit_hash: str) -> bool:
        ref_path = self.path / "refs" / "heads" / ref_name
        ref_path.parent.mkdir(parents=True, exist_ok=True)
        ref_path.write_text(commit_hash)
        return True
    
    def list_refs(self) -> Dict[str, str]:
        refs = {}
        heads_dir = self.path / "refs" / "heads"
        if heads_dir.exists():
            for ref_file in heads_dir.iterdir():
                if ref_file.is_file():
                    refs[ref_file.name] = ref_file.read_text().strip()
        return refs
    
    def log(self, ref_name: str = "main", limit: int = 10) -> List[CommitObject]:
        commit_hash = self.get_ref(ref_name)
        if not commit_hash:
            return []
        
        history = []
        while commit_hash and len(history) < limit:
            commit = self.fetch_commit(commit_hash)
            if not commit:
                break
            history.append(commit)
            commit_hash = commit.parent_hash
        
        return history
    
    def has_commit(self, commit_hash: str) -> bool:
        return self._hash_path(commit_hash, "commits").exists()
    
    def has_blob(self, blob_hash: str) -> bool:
        return self._hash_path(blob_hash, "blobs").exists()
    
    def close(self):
        pass  # File system doesn't need cleanup
    
    def gc(self) -> Dict[str, int]:
        """
        Garbage collect unreferenced blobs.
        
        Returns stats about collected objects.
        """
        # Get all referenced blob hashes
        referenced: Set[str] = set()
        
        for ref_name in self.list_refs():
            for commit in self.log(ref_name, limit=1000):
                referenced.add(commit.am_hash)
                referenced.add(commit.lattice_hash)
                referenced.add(commit.lut_hash)
                referenced.add(commit.config_hash)
        
        # Find unreferenced blobs
        collected = 0
        bytes_freed = 0
        
        blobs_dir = self.path / "blobs"
        for shard_dir in blobs_dir.iterdir():
            if shard_dir.is_dir():
                for blob_file in shard_dir.iterdir():
                    if blob_file.name not in referenced:
                        bytes_freed += blob_file.stat().st_size
                        blob_file.unlink()
                        collected += 1
        
        return {'collected': collected, 'bytes_freed': bytes_freed}


# =============================================================================
# SECTION 5: HRT Serializer (HRT_IICA ↔ Store)
# =============================================================================

class HRTSerializer:
    """
    Serializes HRT_IICA to/from persistent store.
    
    Handles:
    - Converting HRT components to blobs
    - Content-addressed deduplication
    - Commit creation
    """
    
    def __init__(self, store: HRTPersistentStore):
        self.store = store
    
    def push(self, hrt: 'HRT_IICA', message: str = "", ref_name: str = "main") -> CommitObject:
        """
        Push HRT to store.
        
        Creates blobs for AM, Lattice, LUT, Config.
        Creates commit object pointing to blobs.
        Updates ref.
        """
        from .deprecated.hrt_iica import HRT_IICA
        
        # Serialize components to blobs
        am_blob = self._serialize_am(hrt.am)
        lattice_blob = self._serialize_lattice(hrt.lattice)
        lut_blob = self._serialize_lut(hrt.lut)
        config_blob = self._serialize_config(hrt.config)
        
        # Push blobs (deduplicated)
        for blob in [am_blob, lattice_blob, lut_blob, config_blob]:
            self.store.push_blob(blob)
        
        # Create commit
        commit = CommitObject(
            commit_hash=hrt.name,
            parent_hash=hrt.parent_hrt,
            step_number=hrt.step_number,
            timestamp=hrt.timestamp,
            message=message,
            am_hash=am_blob.hash,
            lattice_hash=lattice_blob.hash,
            lut_hash=lut_blob.hash,
            config_hash=config_blob.hash,
            source_info=hrt.source_info
        )
        
        self.store.push_commit(commit)
        self.store.set_ref(ref_name, commit.commit_hash)
        
        return commit
    
    def fetch(self, commit_hash: str) -> Optional['HRT_IICA']:
        """
        Fetch HRT from store by commit hash.
        """
        from .deprecated.hrt_iica import HRT_IICA, EmbeddedLUT
        from .deprecated.hrt import AdjacencyMatrix, HLLSetLattice, HRTConfig
        
        commit = self.store.fetch_commit(commit_hash)
        if not commit:
            return None
        
        # Fetch blobs
        am_blob = self.store.fetch_blob(commit.am_hash)
        lattice_blob = self.store.fetch_blob(commit.lattice_hash)
        lut_blob = self.store.fetch_blob(commit.lut_hash)
        config_blob = self.store.fetch_blob(commit.config_hash)
        
        if not all([am_blob, lattice_blob, lut_blob, config_blob]):
            return None  # Missing blobs
        
        # Deserialize components
        config = self._deserialize_config(config_blob)
        am = self._deserialize_am(am_blob, config)
        lattice = self._deserialize_lattice(lattice_blob, config)
        lut = self._deserialize_lut(lut_blob)
        
        return HRT_IICA(
            am=am,
            lattice=lattice,
            lut=lut,
            config=config,
            parent_hrt=commit.parent_hash,
            step_number=commit.step_number,
            timestamp=commit.timestamp,
            source_info=commit.source_info
        )
    
    def fetch_head(self, ref_name: str = "main") -> Optional['HRT_IICA']:
        """Fetch HRT at HEAD of ref."""
        commit_hash = self.store.get_ref(ref_name)
        if not commit_hash:
            return None
        return self.fetch(commit_hash)
    
    # -------------------------------------------------------------------------
    # Serialization helpers
    # -------------------------------------------------------------------------
    
    def _serialize_am(self, am: 'AdjacencyMatrix') -> Blob:
        """Serialize adjacency matrix."""
        data = pickle.dumps({
            'tensor_data': am.tensor.data.numpy().tobytes(),
            'shape': list(am.tensor.data.shape),
            'dtype': str(am.tensor.data.dtype)
        })
        return Blob.from_data(data, 'am')
    
    def _serialize_lattice(self, lattice: 'HLLSetLattice') -> Blob:
        """Serialize HLLSet lattice."""
        data = pickle.dumps({
            'row_basic': [(b.index, b.hllset.p_bits, b.hllset.dump_numpy().tobytes()) 
                          for b in lattice.row_basic],
            'col_basic': [(b.index, b.hllset.p_bits, b.hllset.dump_numpy().tobytes())
                          for b in lattice.col_basic]
        })
        return Blob.from_data(data, 'lattice')
    
    def _serialize_lut(self, lut: 'EmbeddedLUT') -> Blob:
        """Serialize embedded LUT."""
        data = json.dumps(lut.to_dict()).encode()
        return Blob.from_data(data, 'lut')
    
    def _serialize_config(self, config: 'HRTConfig') -> Blob:
        """Serialize HRT config."""
        data = json.dumps({
            'p_bits': config.p_bits,
            'h_bits': config.h_bits,
            'tau': config.tau,
            'rho': config.rho,
            'epsilon': config.epsilon
        }).encode()
        return Blob.from_data(data, 'config')
    
    def _deserialize_config(self, blob: Blob) -> 'HRTConfig':
        """Deserialize HRT config."""
        from .deprecated.hrt import HRTConfig
        d = json.loads(blob.data.decode())
        return HRTConfig(**d)
    
    def _deserialize_am(self, blob: Blob, config: 'HRTConfig') -> 'AdjacencyMatrix':
        """Deserialize adjacency matrix."""
        from .deprecated.hrt import AdjacencyMatrix
        from .deprecated.immutable_tensor import ImmutableTensor
        import torch
        import numpy as np
        
        d = pickle.loads(blob.data)
        arr = np.frombuffer(d['tensor_data'], dtype=np.float32).reshape(d['shape'])
        tensor = torch.from_numpy(arr.copy())
        
        return AdjacencyMatrix(
            tensor=ImmutableTensor.from_tensor(tensor),
            config=config
        )
    
    def _deserialize_lattice(self, blob: Blob, config: 'HRTConfig') -> 'HLLSetLattice':
        """Deserialize HLLSet lattice."""
        from .deprecated.hrt import HLLSetLattice, BasicHLLSet
        from .hllset import HLLSet
        import numpy as np
        
        d = pickle.loads(blob.data)
        
        row_basic = []
        for idx, p_bits, reg_bytes in d['row_basic']:
            # dump_numpy() returns uint32, so deserialize as uint32
            registers = np.frombuffer(reg_bytes, dtype=np.uint32).copy()
            hllset = HLLSet(p_bits=p_bits)
            hllset._core.set_registers(registers)
            row_basic.append(BasicHLLSet(
                index=idx, is_row=True, hllset=hllset, config=config
            ))
        
        col_basic = []
        for idx, p_bits, reg_bytes in d['col_basic']:
            # dump_numpy() returns uint32, so deserialize as uint32
            registers = np.frombuffer(reg_bytes, dtype=np.uint32).copy()
            hllset = HLLSet(p_bits=p_bits)
            hllset._core.set_registers(registers)
            col_basic.append(BasicHLLSet(
                index=idx, is_row=False, hllset=hllset, config=config
            ))
        
        return HLLSetLattice(
            config=config,
            row_basic=tuple(row_basic),
            col_basic=tuple(col_basic)
        )
    
    def _deserialize_lut(self, blob: Blob) -> 'EmbeddedLUT':
        """Deserialize embedded LUT."""
        from .deprecated.hrt_iica import EmbeddedLUT
        d = json.loads(blob.data.decode())
        return EmbeddedLUT.from_dict(d)


# =============================================================================
# SECTION 6: Factory Functions
# =============================================================================

def create_memory_store() -> MemoryHRTStore:
    """Create in-memory HRT store."""
    return MemoryHRTStore()


def create_file_store(path: str) -> FileSystemHRTStore:
    """Create file system HRT store."""
    return FileSystemHRTStore(path)


def create_serializer(store: HRTPersistentStore) -> HRTSerializer:
    """Create HRT serializer for store."""
    return HRTSerializer(store)
