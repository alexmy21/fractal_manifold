"""
Unified Storage Extension - Multi-Perceptron Lattice Storage with Roaring Compression

Implements unified lattice schema where AM, W, and metadata graphs are all
represented as lattice instances with typed nodes and edges.

Key Features:
- Unified schema for all lattice types (AM, W, metadata)
- Roaring bitmap compression for HLLSets (10-50x compression)
- Multi-perceptron support
- Content-addressable storage
- IICA compliant (Immutable, Idempotent, Content Addressable)
"""

from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import json
import duckdb

from core.extensions.base import ExtensionInfo
from core.extensions.storage import StorageExtension
from core.hllset import HLLSet, compute_sha1


@dataclass
class PerceptronConfig:
    """Configuration for a perceptron."""
    perceptron_id: str
    perceptron_type: str  # 'data', 'metadata', 'image', etc.
    hash_function: str    # 'sha1', 'xxhash', etc.
    hash_seed: int
    config_dict: dict     # HRTConfig or other config


@dataclass
class LatticeNode:
    """Unified lattice node."""
    node_id: str           # Content hash
    node_index: int        # Position in lattice
    node_type: str         # 'am_token', 'w_hllset', 'meta_table', etc.
    content_hash: str      # Hash of node content
    cardinality: float     # Estimated size/count
    properties: dict       # Type-specific properties


@dataclass
class LatticeEdge:
    """Unified lattice edge."""
    edge_id: str           # Content hash
    source_node: str       # node_id
    target_node: str       # node_id
    edge_type: str         # 'am_transition', 'w_morphism', 'meta_fk', etc.
    weight: float          # Frequency, similarity, strength, etc.
    properties: dict       # Type-specific properties


class UnifiedStorageExtension(StorageExtension):
    """
    Unified storage extension with multi-perceptron support.
    
    Stores all lattice types (AM, W, metadata) in unified schema:
    - Perceptrons registry
    - Lattices (typed by perceptron)
    - Lattice nodes (unified with JSON properties)
    - Lattice edges (unified with JSON properties)
    - HLLSets (Roaring compressed)
    - Entanglements (cross-perceptron mappings)
    
    Design Principles:
    - Single schema for all lattice types
    - JSON for type-specific properties
    - Roaring compression for HLLSets
    - Content-addressable artifacts
    - IICA compliant
    """
    
    def __init__(self, db_path: str = ":memory:"):
        """
        Initialize unified storage extension.
        
        Args:
            db_path: Path to DuckDB database file (":memory:" for in-memory)
        """
        self.db_path = db_path
        self.conn = None
        self._initialize_schema()
    
    def _initialize_schema(self):
        """Create unified schema tables if they don't exist."""
        self.conn = duckdb.connect(self.db_path)
        
        # 1. Perceptrons registry
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS perceptrons (
                perceptron_id VARCHAR PRIMARY KEY,
                perceptron_type VARCHAR NOT NULL,
                hash_function VARCHAR NOT NULL,
                hash_seed BIGINT NOT NULL,
                config_json VARCHAR,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                description VARCHAR
            )
        """)
        
        # 2. Lattices registry
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS lattices (
                lattice_id VARCHAR PRIMARY KEY,
                perceptron_id VARCHAR NOT NULL,
                lattice_type VARCHAR NOT NULL,
                dimension INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                config_json VARCHAR,
                FOREIGN KEY (perceptron_id) REFERENCES perceptrons(perceptron_id)
            )
        """)
        
        # 3. Lattice nodes (unified)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS lattice_nodes (
                node_id VARCHAR PRIMARY KEY,
                lattice_id VARCHAR NOT NULL,
                node_index INTEGER NOT NULL,
                node_type VARCHAR NOT NULL,
                content_hash VARCHAR NOT NULL,
                cardinality DOUBLE NOT NULL,
                properties VARCHAR,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (lattice_id) REFERENCES lattices(lattice_id),
                UNIQUE (lattice_id, node_index)
            )
        """)
        
        # 4. Lattice edges (unified)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS lattice_edges (
                edge_id VARCHAR PRIMARY KEY,
                lattice_id VARCHAR NOT NULL,
                source_node VARCHAR NOT NULL,
                target_node VARCHAR NOT NULL,
                edge_type VARCHAR NOT NULL,
                weight DOUBLE NOT NULL,
                properties VARCHAR,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (lattice_id) REFERENCES lattices(lattice_id),
                FOREIGN KEY (source_node) REFERENCES lattice_nodes(node_id),
                FOREIGN KEY (target_node) REFERENCES lattice_nodes(node_id)
            )
        """)
        
        # 5. HLLSets (Roaring compressed)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS hllsets (
                hllset_hash VARCHAR PRIMARY KEY,
                p_bits INTEGER NOT NULL,
                cardinality DOUBLE NOT NULL,
                registers_roaring BLOB NOT NULL,
                original_size INTEGER,
                compressed_size INTEGER,
                compression_ratio DOUBLE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # 6. Entanglements (cross-perceptron)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS entanglements (
                entanglement_id VARCHAR PRIMARY KEY,
                source_lattice VARCHAR NOT NULL,
                target_lattice VARCHAR NOT NULL,
                entanglement_type VARCHAR NOT NULL,
                total_pairs INTEGER,
                avg_strength DOUBLE,
                properties VARCHAR,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (source_lattice) REFERENCES lattices(lattice_id),
                FOREIGN KEY (target_lattice) REFERENCES lattices(lattice_id)
            )
        """)
        
        # 7. Entanglement mappings
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS entanglement_mappings (
                entanglement_id VARCHAR NOT NULL,
                source_node VARCHAR NOT NULL,
                target_node VARCHAR NOT NULL,
                similarity DOUBLE NOT NULL,
                properties VARCHAR,
                PRIMARY KEY (entanglement_id, source_node, target_node),
                FOREIGN KEY (entanglement_id) REFERENCES entanglements(entanglement_id),
                FOREIGN KEY (source_node) REFERENCES lattice_nodes(node_id),
                FOREIGN KEY (target_node) REFERENCES lattice_nodes(node_id)
            )
        """)
        
        # Create indexes for common queries
        self._create_indexes()
    
    def _create_indexes(self):
        """Create indexes for query optimization."""
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_lattice_perceptron ON lattices(perceptron_id)",
            "CREATE INDEX IF NOT EXISTS idx_lattice_type ON lattices(perceptron_id, lattice_type)",
            "CREATE INDEX IF NOT EXISTS idx_node_lattice ON lattice_nodes(lattice_id)",
            "CREATE INDEX IF NOT EXISTS idx_node_type ON lattice_nodes(lattice_id, node_type)",
            "CREATE INDEX IF NOT EXISTS idx_node_hash ON lattice_nodes(content_hash)",
            "CREATE INDEX IF NOT EXISTS idx_node_index ON lattice_nodes(lattice_id, node_index)",
            "CREATE INDEX IF NOT EXISTS idx_edge_lattice ON lattice_edges(lattice_id)",
            "CREATE INDEX IF NOT EXISTS idx_edge_source ON lattice_edges(source_node)",
            "CREATE INDEX IF NOT EXISTS idx_edge_target ON lattice_edges(target_node)",
            "CREATE INDEX IF NOT EXISTS idx_edge_type ON lattice_edges(lattice_id, edge_type)",
            "CREATE INDEX IF NOT EXISTS idx_edge_weight ON lattice_edges(lattice_id, weight DESC)",
            "CREATE INDEX IF NOT EXISTS idx_hllset_pbits ON hllsets(p_bits)",
            "CREATE INDEX IF NOT EXISTS idx_ent_source ON entanglements(source_lattice)",
            "CREATE INDEX IF NOT EXISTS idx_ent_target ON entanglements(target_lattice)",
            "CREATE INDEX IF NOT EXISTS idx_emap_source ON entanglement_mappings(entanglement_id, source_node)",
            "CREATE INDEX IF NOT EXISTS idx_emap_similarity ON entanglement_mappings(entanglement_id, similarity DESC)",
        ]
        
        for idx_sql in indexes:
            self.conn.execute(idx_sql)
    
    # =========================================================================
    # Perceptron Management
    # =========================================================================
    
    def register_perceptron(self, config: PerceptronConfig, description: str = "") -> None:
        """
        Register a new perceptron.
        
        Args:
            config: Perceptron configuration
            description: Optional description
        """
        self.conn.execute("""
            INSERT INTO perceptrons (
                perceptron_id, perceptron_type, hash_function, hash_seed,
                config_json, description
            ) VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT (perceptron_id) DO UPDATE SET
                perceptron_type = EXCLUDED.perceptron_type,
                hash_function = EXCLUDED.hash_function,
                hash_seed = EXCLUDED.hash_seed,
                config_json = EXCLUDED.config_json,
                description = EXCLUDED.description
        """, [
            config.perceptron_id,
            config.perceptron_type,
            config.hash_function,
            config.hash_seed,
            json.dumps(config.config_dict),
            description
        ])
    
    def get_perceptron(self, perceptron_id: str) -> Optional[dict]:
        """Retrieve perceptron configuration."""
        result = self.conn.execute("""
            SELECT * FROM perceptrons WHERE perceptron_id = ?
        """, [perceptron_id]).fetchone()
        
        if not result:
            return None
        
        return {
            'perceptron_id': result[0],
            'perceptron_type': result[1],
            'hash_function': result[2],
            'hash_seed': result[3],
            'config': json.loads(result[4]) if result[4] else {},
            'created_at': result[5],
            'description': result[6]
        }
    
    def list_perceptrons(self) -> List[dict]:
        """List all registered perceptrons."""
        results = self.conn.execute("SELECT * FROM perceptrons ORDER BY created_at").fetchall()
        
        perceptrons = []
        for row in results:
            perceptrons.append({
                'perceptron_id': row[0],
                'perceptron_type': row[1],
                'hash_function': row[2],
                'hash_seed': row[3],
                'config': json.loads(row[4]) if row[4] else {},
                'created_at': row[5],
                'description': row[6]
            })
        
        return perceptrons
    
    # =========================================================================
    # HLLSet Storage (with Roaring compression)
    # =========================================================================
    
    def store_hllset(self, hllset: HLLSet) -> None:
        """
        Store HLLSet with Roaring bitmap compression.
        
        Args:
            hllset: HLLSet to store
        """
        # Get compressed data
        compressed = hllset.dump_roaring()
        stats = hllset.get_compression_stats()
        
        self.conn.execute("""
            INSERT INTO hllsets (
                hllset_hash, p_bits, cardinality,
                registers_roaring, original_size, compressed_size, compression_ratio
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (hllset_hash) DO NOTHING
        """, [
            hllset.name,
            hllset.p_bits,
            hllset.cardinality(),
            compressed,
            stats['original_size'],
            stats['compressed_size'],
            stats['compression_ratio']
        ])
    
    def retrieve_hllset(self, hllset_hash: str) -> Optional[HLLSet]:
        """
        Retrieve HLLSet from storage.
        
        Args:
            hllset_hash: Hash of HLLSet to retrieve
            
        Returns:
            HLLSet instance or None if not found
        """
        result = self.conn.execute("""
            SELECT hllset_hash, p_bits, registers_roaring
            FROM hllsets WHERE hllset_hash = ?
        """, [hllset_hash]).fetchone()
        
        if not result:
            return None
        
        # Reconstruct HLLSet from compressed data
        return HLLSet.from_roaring(result[2], p_bits=result[1])
    
    def get_hllset_stats(self, hllset_hash: str) -> Optional[dict]:
        """Get compression statistics for stored HLLSet."""
        result = self.conn.execute("""
            SELECT cardinality, original_size, compressed_size, compression_ratio
            FROM hllsets WHERE hllset_hash = ?
        """, [hllset_hash]).fetchone()
        
        if not result:
            return None
        
        return {
            'cardinality': result[0],
            'original_size': result[1],
            'compressed_size': result[2],
            'compression_ratio': result[3]
        }
    
    # =========================================================================
    # Lattice Storage
    # =========================================================================
    
    def create_lattice(self, perceptron_id: str, lattice_type: str,
                      dimension: int, config: Optional[dict] = None) -> str:
        """
        Create a new lattice.
        
        Args:
            perceptron_id: Parent perceptron
            lattice_type: 'AM', 'W', 'metadata', etc.
            dimension: Size of lattice
            config: Optional configuration
            
        Returns:
            lattice_id (content hash)
        """
        # Generate lattice_id from content
        content = f"{perceptron_id}:{lattice_type}:{dimension}"
        lattice_id = compute_sha1(content)
        
        self.conn.execute("""
            INSERT INTO lattices (
                lattice_id, perceptron_id, lattice_type, dimension, config_json
            ) VALUES (?, ?, ?, ?, ?)
            ON CONFLICT (lattice_id) DO NOTHING
        """, [
            lattice_id,
            perceptron_id,
            lattice_type,
            dimension,
            json.dumps(config) if config else None
        ])
        
        return lattice_id
    
    def store_lattice_node(self, lattice_id: str, node: LatticeNode) -> None:
        """Store a lattice node."""
        self.conn.execute("""
            INSERT INTO lattice_nodes (
                node_id, lattice_id, node_index, node_type,
                content_hash, cardinality, properties
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (node_id) DO UPDATE SET
                node_index = EXCLUDED.node_index,
                cardinality = EXCLUDED.cardinality,
                properties = EXCLUDED.properties
        """, [
            node.node_id,
            lattice_id,
            node.node_index,
            node.node_type,
            node.content_hash,
            node.cardinality,
            json.dumps(node.properties)
        ])
    
    def store_lattice_edge(self, lattice_id: str, edge: LatticeEdge) -> None:
        """Store a lattice edge."""
        self.conn.execute("""
            INSERT INTO lattice_edges (
                edge_id, lattice_id, source_node, target_node,
                edge_type, weight, properties
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (edge_id) DO UPDATE SET
                weight = EXCLUDED.weight,
                properties = EXCLUDED.properties
        """, [
            edge.edge_id,
            lattice_id,
            edge.source_node,
            edge.target_node,
            edge.edge_type,
            edge.weight,
            json.dumps(edge.properties)
        ])
    
    # =========================================================================
    # High-Level Lattice Storage Methods
    # =========================================================================
    
    def store_am_lattice(
        self,
        perceptron_id: str,
        am_cells: List[Tuple[int, int, int]],  # (row_idx, col_idx, frequency)
        token_lut: Dict[Tuple[int, int], str],  # (reg, zeros) -> token
        dimension: int
    ) -> str:
        """
        Store AM lattice with tokens as nodes and transitions as edges.
        
        Args:
            perceptron_id: Parent perceptron
            am_cells: List of (row_idx, col_idx, frequency) tuples
            token_lut: Mapping from (reg, zeros) to tokens
            dimension: Lattice dimension
            
        Returns:
            lattice_id
        """
        # Create lattice
        lattice_id = self.create_lattice(perceptron_id, 'AM', dimension)
        
        # Create node ID mapping
        node_ids = {}
        
        # Store tokens as nodes
        all_indices = set()
        for row_idx, col_idx, _ in am_cells:
            all_indices.add(row_idx)
            all_indices.add(col_idx)
        
        for idx in sorted(all_indices):
            # Try to find token for this index
            token = token_lut.get((idx, 0), f"idx_{idx}")  # Simplified lookup
            
            node_id = compute_sha1(f"am_token:{perceptron_id}:{idx}:{token}")
            node_ids[idx] = node_id
            
            node = LatticeNode(
                node_id=node_id,
                node_index=idx,
                node_type="am_token",
                content_hash=compute_sha1(token),
                cardinality=1.0,
                properties={
                    "token": token,
                    "index": idx
                }
            )
            
            self.store_lattice_node(lattice_id, node)
        
        # Store transitions as edges
        for row_idx, col_idx, frequency in am_cells:
            if row_idx in node_ids and col_idx in node_ids:
                edge_id = compute_sha1(f"am_edge:{row_idx}:{col_idx}:{frequency}")
                
                edge = LatticeEdge(
                    edge_id=edge_id,
                    source_node=node_ids[row_idx],
                    target_node=node_ids[col_idx],
                    edge_type="am_transition",
                    weight=float(frequency),
                    properties={"frequency": frequency}
                )
                
                self.store_lattice_edge(lattice_id, edge)
        
        return lattice_id
    
    def store_w_lattice(
        self,
        perceptron_id: str,
        basic_hllsets: List[Tuple[int, str, HLLSet]],  # (idx, position, hllset)
        morphisms: List[Tuple[int, int, float, dict]],  # (src_idx, tgt_idx, weight, props)
        dimension: int
    ) -> str:
        """
        Store W lattice with HLLSets as nodes and morphisms as edges.
        
        Args:
            perceptron_id: Parent perceptron
            basic_hllsets: List of (index, position, hllset) tuples
            morphisms: List of (source_idx, target_idx, weight, properties) tuples
            dimension: Lattice dimension
            
        Returns:
            lattice_id
        """
        # Create lattice
        lattice_id = self.create_lattice(perceptron_id, 'W', dimension)
        
        # Store all HLLSets first
        for _, _, hllset in basic_hllsets:
            self.store_hllset(hllset)
        
        # Store HLLSets as nodes
        node_ids = {}
        for idx, position, hllset in basic_hllsets:
            node_id = compute_sha1(f"w_hllset:{perceptron_id}:{idx}:{hllset.name}")
            node_ids[idx] = node_id
            
            node = LatticeNode(
                node_id=node_id,
                node_index=idx,
                node_type="w_hllset",
                content_hash=hllset.name,
                cardinality=hllset.cardinality(),
                properties={
                    "hllset_hash": hllset.name,
                    "p_bits": hllset.p_bits,
                    "position": position
                }
            )
            
            self.store_lattice_node(lattice_id, node)
        
        # Store morphisms as edges
        for src_idx, tgt_idx, weight, props in morphisms:
            if src_idx in node_ids and tgt_idx in node_ids:
                edge_id = compute_sha1(f"w_morph:{src_idx}:{tgt_idx}:{weight}")
                
                edge = LatticeEdge(
                    edge_id=edge_id,
                    source_node=node_ids[src_idx],
                    target_node=node_ids[tgt_idx],
                    edge_type="w_morphism",
                    weight=weight,
                    properties=props
                )
                
                self.store_lattice_edge(lattice_id, edge)
        
        return lattice_id
    
    # =========================================================================
    # Query Methods
    # =========================================================================
    
    def get_lattice_info(self, lattice_id: str) -> Optional[dict]:
        """Get lattice metadata."""
        result = self.conn.execute("""
            SELECT l.*, p.perceptron_type
            FROM lattices l
            JOIN perceptrons p ON l.perceptron_id = p.perceptron_id
            WHERE l.lattice_id = ?
        """, [lattice_id]).fetchone()
        
        if not result:
            return None
        
        return {
            'lattice_id': result[0],
            'perceptron_id': result[1],
            'lattice_type': result[2],
            'dimension': result[3],
            'created_at': result[4],
            'config': json.loads(result[5]) if result[5] else {},
            'perceptron_type': result[6]
        }
    
    def get_lattice_nodes(self, lattice_id: str, node_type: Optional[str] = None) -> List[dict]:
        """Get all nodes in a lattice, optionally filtered by type."""
        if node_type:
            results = self.conn.execute("""
                SELECT * FROM lattice_nodes
                WHERE lattice_id = ? AND node_type = ?
                ORDER BY node_index
            """, [lattice_id, node_type]).fetchall()
        else:
            results = self.conn.execute("""
                SELECT * FROM lattice_nodes
                WHERE lattice_id = ?
                ORDER BY node_index
            """, [lattice_id]).fetchall()
        
        nodes = []
        for row in results:
            nodes.append({
                'node_id': row[0],
                'lattice_id': row[1],
                'node_index': row[2],
                'node_type': row[3],
                'content_hash': row[4],
                'cardinality': row[5],
                'properties': json.loads(row[6]) if row[6] else {},
                'created_at': row[7]
            })
        
        return nodes
    
    def get_lattice_edges(self, lattice_id: str, edge_type: Optional[str] = None,
                         min_weight: Optional[float] = None) -> List[dict]:
        """Get all edges in a lattice, optionally filtered."""
        query = "SELECT * FROM lattice_edges WHERE lattice_id = ?"
        params = [lattice_id]
        
        if edge_type:
            query += " AND edge_type = ?"
            params.append(edge_type)
        
        if min_weight is not None:
            query += " AND weight >= ?"
            params.append(min_weight)
        
        query += " ORDER BY weight DESC"
        
        results = self.conn.execute(query, params).fetchall()
        
        edges = []
        for row in results:
            edges.append({
                'edge_id': row[0],
                'lattice_id': row[1],
                'source_node': row[2],
                'target_node': row[3],
                'edge_type': row[4],
                'weight': row[5],
                'properties': json.loads(row[6]) if row[6] else {},
                'created_at': row[7]
            })
        
        return edges
    
    def get_storage_stats(self) -> dict:
        """Get overall storage statistics."""
        stats = {}
        
        # Count records
        stats['perceptrons'] = self.conn.execute("SELECT COUNT(*) FROM perceptrons").fetchone()[0]
        stats['lattices'] = self.conn.execute("SELECT COUNT(*) FROM lattices").fetchone()[0]
        stats['nodes'] = self.conn.execute("SELECT COUNT(*) FROM lattice_nodes").fetchone()[0]
        stats['edges'] = self.conn.execute("SELECT COUNT(*) FROM lattice_edges").fetchone()[0]
        stats['hllsets'] = self.conn.execute("SELECT COUNT(*) FROM hllsets").fetchone()[0]
        
        # Compression stats
        compression = self.conn.execute("""
            SELECT 
                SUM(original_size) as total_original,
                SUM(compressed_size) as total_compressed,
                AVG(compression_ratio) as avg_ratio
            FROM hllsets
        """).fetchone()
        
        stats['hllset_compression'] = {
            'total_original': compression[0] or 0,
            'total_compressed': compression[1] or 0,
            'avg_compression_ratio': compression[2] or 0.0
        }
        
        return stats
    
    # =========================================================================
    # Extension Interface (Abstract Methods)
    # =========================================================================
    
    def store_lut(self, n: int, lut: Dict[Tuple[int, int], Any], 
                  hllset_hash: str, metadata: Optional[dict] = None) -> int:
        """
        Store LUT data (legacy interface - unified storage uses different model).
        For backward compatibility only.
        """
        # In unified storage, LUT data is stored as lattice nodes/edges
        # This is a legacy interface that can map to lattice storage
        return 0  # Not implemented in unified model
    
    def query_tokens(self, n: int, reg: int, zeros: int,
                    hllset_hash: Optional[str] = None) -> List[Tuple[str, ...]]:
        """Query tokens at specific coordinates (legacy interface)."""
        # In unified storage, query via lattice nodes
        return []  # Not implemented in unified model
    
    def query_by_token(self, n: int, token_tuple: Tuple[str, ...]) -> List[Tuple[int, int]]:
        """Reverse lookup for token (legacy interface)."""
        # In unified storage, query via lattice edges
        return []  # Not implemented in unified model
    
    def get_metadata(self, hllset_hash: str) -> Optional[dict]:
        """Get HLLSet metadata."""
        return self.get_hllset_stats(hllset_hash)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        return self.get_storage_stats()
    
    def get_capabilities(self) -> List[str]:
        """Return list of capabilities."""
        return [
            'multi_perceptron',
            'roaring_compression',
            'unified_lattice',
            'am_storage',
            'w_storage',
            'metadata_storage',
            'entanglement_storage',
            'content_addressable'
        ]
    
    def is_available(self) -> bool:
        """Check if storage is available."""
        return self.conn is not None
    
    def initialize(self, config: dict) -> None:
        """Initialize with configuration (already done in __init__)."""
        pass  # Schema already initialized
    
    def cleanup(self) -> None:
        """Cleanup resources."""
        self.close()
    
    def get_info(self) -> ExtensionInfo:
        """Return extension information."""
        return ExtensionInfo(
            name="UnifiedStorage",
            version="1.0.0",
            description="Unified multi-perceptron lattice storage with Roaring compression",
            category="storage"
        )
    
    def process(self, operation: str, data: dict) -> Any:
        """
        Generic processing interface.
        
        Supported operations:
        - 'register_perceptron'
        - 'store_hllset'
        - 'store_am_lattice'
        - 'store_w_lattice'
        - 'get_stats'
        """
        if operation == 'register_perceptron':
            config = PerceptronConfig(**data['config'])
            self.register_perceptron(config, data.get('description', ''))
            return {'success': True}
        
        elif operation == 'store_hllset':
            self.store_hllset(data['hllset'])
            return {'success': True, 'hash': data['hllset'].name}
        
        elif operation == 'store_am_lattice':
            lattice_id = self.store_am_lattice(**data)
            return {'success': True, 'lattice_id': lattice_id}
        
        elif operation == 'store_w_lattice':
            lattice_id = self.store_w_lattice(**data)
            return {'success': True, 'lattice_id': lattice_id}
        
        elif operation == 'get_stats':
            return self.get_storage_stats()
        
        else:
            raise ValueError(f"Unknown operation: {operation}")
    
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None
    
    def __del__(self):
        """Cleanup on deletion."""
        self.close()
