"""
Storage extension interface and reference implementation.

=== REFERENCE IMPLEMENTATION PATTERN ===

DuckDB storage demonstrates proper SEPARATION OF CONCERNS:

  APPLICATION RESPONSIBILITY (ManifoldOS):
    ✓ State management - when to store, what to store
    ✓ Business logic - token ingestion, querying
    ✓ Lifecycle - when to commit LUTs
    ✓ Policy decisions - storage triggers, retention

  EXTENSION RESPONSIBILITY (DuckDBStorage):
    ✓ Processing - execute storage operations
    ✓ Persistence - write to database
    ✓ Resources - manage connections
    ✗ NOT state management - no application state
    ✗ NOT business logic - no policy decisions

Key Principle:
  "Extensions provide capabilities, Applications manage state"

Core Principles:
  1. Immutability: Configurations frozen, data append-only
  2. Idempotence: Same operation → same result
  3. Content-addressability: Everything identified by hash
  4. Statelessness: No application state in extension

Knowledge Base Integration:
  - Extension configs stored immutably
  - Operation logs content-addressed
  - Complete audit trail
"""

from abc import abstractmethod
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

from core.extensions.base import ManifoldExtension, ExtensionInfo


class StorageExtension(ManifoldExtension):
    """
    Abstract interface for storage extensions.
    
    === REFERENCE PATTERN: Separation of Concerns ===
    
    Extension Role: PROCESSING (Pure Service Provider)
      ✓ Provides storage capability
      ✓ Executes storage operations
      ✓ Manages database resources
      ✗ Does NOT decide when to store
      ✗ Does NOT accumulate application state
      ✗ Does NOT implement business logic
    
    Application Role: STATE MANAGEMENT
      ✓ Decides when to persist data
      ✓ Manages LUT state
      ✓ Implements business rules
      ✓ Triggers storage operations
    
    Key Methods (All IDEMPOTENT):
      - store_lut(): Persist LUT (app calls when ready)
      - retrieve_lut(): Query LUT (app requests)
      - get_stats(): Statistics (app requests)
    
    Design Principles:
      - IMMUTABLE: Configuration frozen after init
      - IDEMPOTENT: Same input → same result
      - STATELESS: No application state accumulated
      - CONTENT-ADDRESSED: Records identified by hash
    """
    
    @abstractmethod
    def store_lut(self, n: int, lut: Dict[Tuple[int, int], Any], 
                  hllset_hash: str, metadata: Optional[dict] = None) -> int:
        """
        Store LUT data for an HLLSet.
        
        === SEPARATION OF CONCERNS ===
        
        Application Responsibility (BEFORE calling):
          ✓ Decide WHEN to persist (on commit, threshold, etc.)
          ✓ Prepare WHAT to persist (LUT data structure)
          ✓ Handle business logic (validation, transformation)
        
        Extension Responsibility (THIS method):
          ✓ Execute storage operation (write to database)
          ✓ Ensure idempotence (UPSERT semantics)
          ✓ Manage resources (connections, transactions)
        
        GUARANTEES:
          - IDEMPOTENT: Same data → same stored state
          - APPEND-ONLY: New records added, existing updated
          - CONTENT-ADDRESSED: Records identified by hash
          - NO STATE: Extension doesn't accumulate app data
        
        Example (CORRECT pattern):
          ```python
          # App decides when to store
          if app.should_persist():  # APP LOGIC
              # App prepares data
              lut_data = app.get_lut()  # APP STATE
              # Extension processes
              storage.store_lut(n, lut_data, hash)  # EXTENSION
          ```
        
        Args:
            n: N-gram size (1, 2, 3, etc.)
            lut: Lookup table mapping (reg, zeros) → LUTRecord
            hllset_hash: Content-addressed hash of HLLSet
            metadata: Optional ingestion metadata
            
        Returns:
            Number of records stored/updated
        """
        pass
    
    @abstractmethod
    def query_tokens(self, n: int, reg: int, zeros: int,
                    hllset_hash: Optional[str] = None) -> List[Tuple[str, ...]]:
        """
        Query tokens at specific coordinates.
        
        Args:
            n: N-gram size
            reg: HLL register index
            zeros: Leading zeros count
            hllset_hash: Optional filter by specific HLLSet
            
        Returns:
            List of token tuples at this location
        """
        pass
    
    @abstractmethod
    def query_by_token(self, n: int, token_tuple: Tuple[str, ...]) -> List[Tuple[int, int]]:
        """
        Reverse lookup: Find coordinates for a token.
        
        Args:
            n: N-gram size
            token_tuple: Token sequence to search for
            
        Returns:
            List of (reg, zeros) coordinates where token appears
        """
        pass
    
    @abstractmethod
    def get_metadata(self, hllset_hash: str) -> Optional[dict]:
        """
        Get ingestion metadata for an HLLSet.
        
        Args:
            hllset_hash: HLLSet identifier
            
        Returns:
            Metadata dictionary or None if not found
        """
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """
        Get storage statistics.
        
        Returns:
            Dictionary with storage metrics (size, counts, etc.)
        """
        pass


class DuckDBStorageExtension(StorageExtension):
    """
    DuckDB reference implementation of storage extension.
    
    ✓ CERTIFIED STATELESS - Validated 2026-02-07
    ✓ REFERENCE IMPLEMENTATION - Demonstrates all patterns
    
    === SEPARATION OF CONCERNS DEMONSTRATED ===
    
    What This Extension DOES (Processing):
      ✓ Execute SQL operations (INSERT, SELECT, UPDATE)
      ✓ Manage DuckDB connection lifecycle
      ✓ Ensure ACID guarantees
      ✓ Handle errors gracefully
      ✓ Provide query interface
    
    What This Extension DOES NOT DO (Application's job):
      ✗ Decide when to store (app calls us when ready)
      ✗ Cache LUT data in memory (state in app or DB)
      ✗ Implement business rules (app logic)
      ✗ Batch operations (app batches, we execute)
      ✗ Lifecycle management (app decides commit timing)
    
    Pattern Example:
      ```python
      # APPLICATION manages state & logic:
      class ManifoldOS:
          def ingest(self, tokens):
              # App logic: update internal state
              self.lut[key] = value  # APP STATE
              
              # App decision: when to persist
              if self.should_commit():  # APP LOGIC
                  # Extension processes: pure execution
                  self.storage.store_lut(self.lut)  # EXTENSION
      
      # EXTENSION provides processing capability:
      class DuckDBStorageExtension:
          def store_lut(self, lut):
              # Pure processing: input → external storage
              self.lut_store.commit(lut)  # No state accumulation
              return True
      ```
    
    Implements ManifoldOS core principles:
      1. Immutability: Configuration frozen after init
      2. Idempotence: Same data → same storage (content-addressed)
      3. Content-addressability: Records identified by hash
      4. Statelessness: No accumulated state, all operations pure
    
    Stateless Design:
      - Configuration frozen (immutable)
      - No mutable instance state accumulated
      - All operations idempotent (UPSERT semantics)
      - State stored externally (DuckDB database)
      - Multiple instances independent
      - Same config → same behavior (deterministic)
    
    DuckDB advantages:
      - Embedded: No separate server process
      - Fast: Columnar storage, vectorized execution
      - ACID: Full transaction support
      - SQL: Rich query capabilities
      - Small: Minimal dependencies
      - Append-only support: Perfect for immutable storage
    
    Configuration (immutable after init):
      {
          'db_path': ':memory:' | '/path/to/file.duckdb',
          'read_only': False,
          'threads': None  # Auto-detect
      }
    """
    
    def __init__(self):
        super().__init__()
        self.lut_store = None
        self._available = False
    
    def initialize(self, config: Dict[str, Any]) -> bool:
        """
        Initialize DuckDB storage with IMMUTABLE configuration.
        
        Configuration is frozen to ensure:
          - Content-addressability (stable config hash)
          - Idempotence (same config = same behavior)
          - Knowledge base integration
        """
        try:
            # Check if DuckDB is available
            try:
                import duckdb
            except ImportError:
                print("⚠ DuckDB not installed (pip install duckdb)")
                return False
            
            # Import our LUT store implementation
            from core.lut_store import DuckDBLUTStore
            
            # Freeze configuration (makes it immutable)
            self._freeze_config(config, extension_type='duckdb_storage')
            
            # Get configuration from frozen config
            db_path = self.config.get('db_path', ':memory:')
            
            # Initialize store (connects and creates schema automatically)
            # DuckDBLUTStore is idempotent - safe to create with same path
            self.lut_store = DuckDBLUTStore(db_path)
            self._available = True
            
            return True
            
        except Exception as e:
            print(f"⚠ DuckDB initialization failed: {e}")
            self._available = False
            return False
    
    def is_available(self) -> bool:
        """Check if DuckDB storage is operational."""
        return self._available and self.lut_store is not None
    
    def cleanup(self):
        """Close DuckDB connection."""
        if self.lut_store:
            try:
                self.lut_store.close()
            except Exception as e:
                print(f"⚠ DuckDB cleanup error: {e}")
            finally:
                self.lut_store = None
                self._available = False
    
    def get_capabilities(self) -> Dict[str, bool]:
        """
        DuckDB capabilities (immutable).
        
        Returns fixed set of capabilities - never changes for a given version.
        """
        return {
            'persistent_storage': True,
            'lut_queries': True,
            'metadata_tracking': True,
            'sql_queries': True,
            'transactions': True,
            'analytics': True,
            'content_addressable': True,  # Records identified by hash
            'append_only': True,          # Immutable storage pattern
            'idempotent': True,           # Same data → same result
        }
    
    def get_info(self) -> ExtensionInfo:
        """Get extension metadata with content-addressed hash."""
        try:
            import duckdb
            version = duckdb.__version__
        except:
            version = "unknown"
        
        return ExtensionInfo(
            name="DuckDB Storage",
            version=version,
            description="Embedded SQL database for LUT persistence",
            author="HLLSet Manifold Team",
            config_hash=self.get_config_hash(),  # Content-addressed ID
            capabilities=self.get_capabilities()
        )
    
    def validate_config(self, config: Dict[str, Any]) -> List[str]:
        """Validate DuckDB configuration."""
        errors = []
        
        # Check db_path type
        if 'db_path' in config and not isinstance(config['db_path'], str):
            errors.append("db_path must be a string")
        
        # Check read_only type
        if 'read_only' in config and not isinstance(config['read_only'], bool):
            errors.append("read_only must be boolean")
        
        return errors
    
    # Storage interface implementation
    
    def store_lut(self, n: int, lut: Dict[Tuple[int, int], Any],
                  hllset_hash: str, metadata: Optional[dict] = None) -> int:
        """Store LUT using DuckDB."""
        if not self.is_available():
            raise RuntimeError("DuckDB storage not available")
        
        return self.lut_store.commit_lut(n, lut, hllset_hash, metadata)
    
    def query_tokens(self, n: int, reg: int, zeros: int,
                    hllset_hash: Optional[str] = None) -> List[Tuple[str, ...]]:
        """Query tokens from DuckDB."""
        if not self.is_available():
            return []
        
        return self.lut_store.get_tokens(n, reg, zeros, hllset_hash)
    
    def query_by_token(self, n: int, token_tuple: Tuple[str, ...]) -> List[Tuple[int, int]]:
        """Reverse lookup in DuckDB."""
        if not self.is_available():
            return []
        
        return self.lut_store.query_by_token(n, token_tuple)
    
    def get_metadata(self, hllset_hash: str) -> Optional[dict]:
        """Get metadata from DuckDB."""
        if not self.is_available():
            return None
        
        return self.lut_store.get_metadata(hllset_hash)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics from DuckDB."""
        if not self.is_available():
            return {}
        
        return self.lut_store.get_stats()
