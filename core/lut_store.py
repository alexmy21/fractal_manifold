#!/usr/bin/env python3
"""
LUT Persistent Storage - Enterprise to AI Metadata Bridge

Provides persistent storage for LUT (Lookup Table) mappings:
  (reg, zeros) → [token tuples, hashes, metadata]

This is the critical link for grounding AI operations back to source data.
The LUT acts as metadata that bridges:
  - ED (Enterprise Data): Original tokens/records
  - AI: HLLSet fingerprints (content-addressed)

Architecture:
  - Abstract base: LUTPersistentStore
  - DuckDB implementation: Fast, embedded, ACID
  - Extensible: Redis, PostgreSQL, etc.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set
from pathlib import Path
import json

try:
    import duckdb
    DUCKDB_AVAILABLE = True
except ImportError:
    DUCKDB_AVAILABLE = False


@dataclass
class LUTRecord:
    """
    Single LUT record linking (reg, zeros) to tokens.
    
    Handles hash collisions: Multiple tokens can hash to the same (reg, zeros).
    Both hashes and tokens act as sets - no duplicates.
    
    Fields:
        reg: HLL register index
        zeros: Leading zeros count
        hashes: Set of token hashes at this (reg, zeros)
        tokens: List of token tuples that hashed here (for disambiguation)
    """
    reg: int
    zeros: int
    hashes: Set[int] = field(default_factory=set)
    tokens: List[Tuple[str, ...]] = field(default_factory=list)
    
    def add_entry(self, hash_val: int, token_seq: Tuple[str, ...]):
        """
        Add hash and token sequence to this LUT record.
        
        Handles collisions: Multiple tokens can hash to same (reg, zeros).
        - Hashes stored as set (automatic deduplication)
        - Tokens deduplicated manually (list preserves order)
        
        Args:
            hash_val: Token hash value
            token_seq: Token tuple (e.g., ('customer',) or ('premium', 'customer'))
        """
        # Add hash (set handles deduplication automatically)
        self.hashes.add(hash_val)
        
        # Add token if not already present (manual deduplication)
        if token_seq not in self.tokens:
            self.tokens.append(token_seq)
    
    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        return {
            'reg': self.reg,
            'zeros': self.zeros,
            'hashes': list(self.hashes),
            'tokens': [list(t) for t in self.tokens]
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> 'LUTRecord':
        """Create from dict."""
        return cls(
            reg=d['reg'],
            zeros=d['zeros'],
            hashes=set(d['hashes']),
            tokens=[tuple(t) for t in d['tokens']]
        )


class LUTPersistentStore(ABC):
    """
    Abstract base for LUT persistent storage.
    
    The LUT is the metadata layer that grounds HLLSet fingerprints
    back to source enterprise data. Critical for explainability.
    """
    
    @abstractmethod
    def commit_lut(self, n: int, lut: Dict[Tuple[int, int], LUTRecord], 
                   hllset_hash: str, metadata: Optional[dict] = None) -> str:
        """
        Commit LUT to persistent storage.
        
        Args:
            n: N-token group size (1, 2, 3, etc.)
            lut: In-memory LUT from ingestion
            hllset_hash: Content-addressed hash of HLLSet
            metadata: Optional metadata (source, timestamp, etc.)
        
        Returns:
            Commit ID/hash
        """
        pass
    
    @abstractmethod
    def get_tokens(self, n: int, reg: int, zeros: int, 
                   hllset_hash: Optional[str] = None) -> List[Tuple[str, ...]]:
        """
        Retrieve tokens for (reg, zeros) key.
        
        Args:
            n: N-token group size
            reg: Register index
            zeros: Leading zeros
            hllset_hash: Optional - filter by specific HLLSet
        
        Returns:
            List of token tuples that map to this (reg, zeros)
        """
        pass
    
    @abstractmethod
    def query_by_token(self, n: int, token_tuple: Tuple[str, ...]) -> List[Tuple[int, int]]:
        """
        Reverse lookup: Find (reg, zeros) keys containing token.
        
        Args:
            n: N-token group size
            token_tuple: Token tuple to search for
        
        Returns:
            List of (reg, zeros) keys
        """
        pass
    
    @abstractmethod
    def get_metadata(self, hllset_hash: str) -> Optional[dict]:
        """Get metadata for an HLLSet."""
        pass
    
    @abstractmethod
    def close(self):
        """Close store and release resources."""
        pass


class DuckDBLUTStore(LUTPersistentStore):
    """
    DuckDB-based LUT persistent storage.
    
    DuckDB advantages:
    - Embedded (no server needed)
    - ACID transactions
    - Fast analytical queries
    - SQL interface
    - Small footprint
    
    Revised Schema (v2):
        lut_records:
            - token_hash (BIGINT): PRIMARY KEY - the actual hash value
            - n (INTEGER): N-token size (1, 2, 3) - for fast filtering
            - reg (INTEGER): HLL register (derived from hash)
            - zeros (INTEGER): Leading zeros (derived from hash)
            - tokens (TEXT): JSON array of token tuples (collision resolution)
            - created_at (TIMESTAMP): First seen
            - updated_at (TIMESTAMP): Last modified
        
        Key design decisions:
            1. token_hash is PK (not integer id) - hash is the identity
            2. n kept for fast n-gram filtering
            3. No hllset_hash - tokens are global, not per-HLLSet
            4. Single hash (not array) - hash IS the record
            5. No commit_id - LUT is mutable (update tokens on collision)
            6. Both created_at and updated_at for audit trail
        
        Indexes:
            - PRIMARY KEY on token_hash
            - INDEX on (n, reg, zeros) for reverse lookup from HLLSet
            - INDEX on n for filtering by n-gram size
    """
    
    def __init__(self, db_path: str = ":memory:"):
        """
        Initialize DuckDB LUT store.
        
        Args:
            db_path: Path to DuckDB file, or ":memory:" for in-memory
        """
        if not DUCKDB_AVAILABLE:
            raise ImportError("DuckDB not available. Install: pip install duckdb")
        
        self.db_path = db_path
        self.conn = duckdb.connect(db_path)
        self._create_schema()
    
    def _create_schema(self):
        """Create LUT tables with revised schema."""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS lut_records (
                token_hash UBIGINT PRIMARY KEY,
                n INTEGER NOT NULL,
                reg INTEGER NOT NULL,
                zeros INTEGER NOT NULL,
                tokens TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Index for reverse lookup: (reg, zeros) -> tokens
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_reg_zeros 
            ON lut_records(n, reg, zeros)
        """)
        
        # Index for filtering by n-gram size
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_n 
            ON lut_records(n)
        """)
    
    def commit_lut(self, n: int, lut: Dict[Tuple[int, int], LUTRecord], 
                   hllset_hash: str, metadata: Optional[dict] = None) -> int:
        """
        Commit LUT to DuckDB with UPSERT semantics.
        
        Since LUT is mutable (collision resolution), we use INSERT OR REPLACE
        to update existing records when hash collisions are detected.
        
        Args:
            n: N-gram size
            lut: In-memory LUT (may have multiple hashes per (reg,zeros))
            hllset_hash: Ignored (tokens are global)
            metadata: Ignored (no longer per-HLLSet)
        
        Returns:
            Number of records inserted/updated
        """
        # Begin transaction
        self.conn.begin()
        
        try:
            records_affected = 0
            
            # Process each LUT record
            for (reg, zeros), record in lut.items():
                # Each hash in the record gets its own row
                for token_hash in record.hashes:
                    # Find which tokens have this hash
                    tokens_for_hash = [
                        t for t in record.tokens 
                        if hash(f"__n{n}__" + "__".join(t)) & 0xFFFFFFFFFFFFFFFF == token_hash
                    ]
                    
                    if tokens_for_hash:
                        # UPSERT: Insert or update if hash exists
                        self.conn.execute("""
                            INSERT INTO lut_records 
                            (token_hash, n, reg, zeros, tokens)
                            VALUES (?, ?, ?, ?, ?)
                            ON CONFLICT(token_hash) DO UPDATE SET
                                tokens = excluded.tokens,
                                updated_at = now()
                        """, [
                            token_hash,
                            n,
                            reg,
                            zeros,
                            json.dumps([list(t) for t in tokens_for_hash])
                        ])
                        records_affected += 1
            
            self.conn.commit()
            return records_affected
            
        except Exception as e:
            self.conn.rollback()
            raise RuntimeError(f"LUT commit failed: {e}")
    
    def get_tokens(self, n: int, reg: int, zeros: int, 
                   hllset_hash: Optional[str] = None) -> List[Tuple[str, ...]]:
        """
        Retrieve tokens from DuckDB by (n, reg, zeros).
        
        Note: hllset_hash parameter kept for API compatibility but ignored
        since tokens are now global.
        """
        query = """
            SELECT tokens FROM lut_records
            WHERE n = ? AND reg = ? AND zeros = ?
        """
        params = [n, reg, zeros]
        
        results = self.conn.execute(query, params).fetchall()
        
        all_tokens = []
        for (tokens_json,) in results:
            token_lists = json.loads(tokens_json)
            all_tokens.extend([tuple(t) for t in token_lists])
        
        return all_tokens
    
    def query_by_token(self, n: int, token_tuple: Tuple[str, ...]) -> List[Tuple[int, int]]:
        """Reverse lookup by token."""
        # This requires scanning - consider FTS index for production
        query = """
            SELECT DISTINCT reg, zeros FROM lut_records
            WHERE n = ? AND tokens LIKE ?
        """
        token_str = json.dumps(list(token_tuple))
        results = self.conn.execute(query, [n, f"%{token_str}%"]).fetchall()
        return results
    
    def get_metadata(self, hllset_hash: str) -> Optional[dict]:
        """
        Get metadata for HLLSet.
        
        DEPRECATED: Metadata is no longer stored in LUT.
        Kept for API compatibility - always returns None.
        """
        return None
    
    def get_stats(self) -> dict:
        """Get storage statistics."""
        total_hashes = self.conn.execute(
            "SELECT COUNT(*) FROM lut_records"
        ).fetchone()[0]
        
        by_n = self.conn.execute("""
            SELECT n, COUNT(*) as count
            FROM lut_records
            GROUP BY n
            ORDER BY n
        """).fetchall()
        
        oldest = self.conn.execute(
            "SELECT MIN(created_at) FROM lut_records"
        ).fetchone()[0]
        
        newest = self.conn.execute(
            "SELECT MAX(updated_at) FROM lut_records"
        ).fetchone()[0]
        
        return {
            'total_token_hashes': total_hashes,
            'n_groups': {n: count for n, count in by_n},
            'oldest_record': str(oldest) if oldest else None,
            'newest_record': str(newest) if newest else None,
            'db_path': self.db_path
        }
    
    def close(self):
        """Close DuckDB connection."""
        if self.conn:
            self.conn.close()
            self.conn = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# Export for convenience
__all__ = [
    'LUTRecord',
    'LUTPersistentStore',
    'DuckDBLUTStore',
    'DUCKDB_AVAILABLE'
]
