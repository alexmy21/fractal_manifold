"""
Database Columnar Ingestion System

Ingests tabular data from DuckDB column by column, creating:
1. Data hierarchy: DB HLLSet → Table HLLSets → Column HLLSets
2. Metadata hierarchy: Raw metadata + HLLSet fingerprints + SHA1 IDs
3. Explicit entanglement: Data ↔ Metadata mapping

Design rationale:
- Columnar representation: thousands of columns vs billions of rows = massive compression
- Row reconstruction: Row HLLSet ∩ Column HLLSet = Cell HLLSet (can be disambiguated)
- SQL queries return limited rows, stored as collection of row HLLSets
- Metadata uses hybrid approach: raw data (precision) + HLLSet fingerprints (semantic search)
"""

import duckdb
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from core.hllset import HLLSet
from core.manifold_os import ManifoldOS
import json


@dataclass
class ColumnMetadata:
    """Metadata for a database column."""
    database_name: str
    table_name: str
    column_name: str
    data_type: str
    nullable: bool
    
    # Statistics
    row_count: int
    distinct_count: Optional[int] = None
    null_count: Optional[int] = None
    min_value: Optional[str] = None
    max_value: Optional[str] = None
    avg_value: Optional[float] = None
    
    # Identifiers
    sha1_id: str = ""
    
    def __post_init__(self):
        """Generate SHA1 ID from qualified name."""
        if not self.sha1_id:
            qualified_name = f"{self.database_name}.{self.table_name}.{self.column_name}"
            self.sha1_id = hashlib.sha1(qualified_name.encode()).hexdigest()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    def to_tokens(self) -> List[str]:
        """Convert to tokens for HLLSet fingerprint."""
        tokens = [
            self.column_name,
            self.data_type.lower(),
            self.table_name,
        ]
        
        # Add semantic tokens based on data type
        dtype = self.data_type.lower()
        if 'int' in dtype or 'decimal' in dtype or 'numeric' in dtype:
            tokens.append('numeric')
        if 'varchar' in dtype or 'text' in dtype or 'char' in dtype:
            tokens.append('text')
        if 'date' in dtype or 'time' in dtype:
            tokens.append('temporal')
        if 'bool' in dtype:
            tokens.append('boolean')
        
        if self.nullable:
            tokens.append('nullable')
        
        return tokens


@dataclass
class TableMetadata:
    """Metadata for a database table."""
    database_name: str
    table_name: str
    row_count: int
    column_count: int
    columns: List[str]
    
    sha1_id: str = ""
    
    def __post_init__(self):
        """Generate SHA1 ID from qualified name."""
        if not self.sha1_id:
            qualified_name = f"{self.database_name}.{self.table_name}"
            self.sha1_id = hashlib.sha1(qualified_name.encode()).hexdigest()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    def to_tokens(self) -> List[str]:
        """Convert to tokens for HLLSet fingerprint."""
        return [self.table_name] + self.columns


@dataclass
class DatabaseMetadata:
    """Metadata for a database."""
    database_name: str
    table_count: int
    tables: List[str]
    
    sha1_id: str = ""
    
    def __post_init__(self):
        """Generate SHA1 ID."""
        if not self.sha1_id:
            self.sha1_id = hashlib.sha1(self.database_name.encode()).hexdigest()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    def to_tokens(self) -> List[str]:
        """Convert to tokens for HLLSet fingerprint."""
        return [self.database_name] + self.tables


class DatabaseIngestionSystem:
    """
    Columnar ingestion system for database tables.
    
    Creates three hierarchies:
    1. Data: DB HLLSet → Table HLLSets → Column HLLSets
    2. Metadata: DB metadata → Table metadata → Column metadata (hybrid storage)
    3. Entanglement: Explicit data ↔ metadata mapping
    """
    
    def __init__(self, manifold: ManifoldOS):
        """Initialize with ManifoldOS instance."""
        self.manifold = manifold
        self.db_conn: Optional[duckdb.DuckDBPyConnection] = None
        
    def connect(self, db_path: Path):
        """Connect to DuckDB database."""
        self.db_conn = duckdb.connect(str(db_path), read_only=True)
        print(f"Connected to database: {db_path}")
    
    def disconnect(self):
        """Disconnect from database."""
        if self.db_conn:
            self.db_conn.close()
            self.db_conn = None
    
    def get_database_info(self) -> DatabaseMetadata:
        """Extract database metadata."""
        result = self.db_conn.execute("""
            SELECT table_name 
            FROM duckdb_tables() 
            WHERE schema_name = 'main'
            ORDER BY table_name
        """).fetchall()
        
        tables = [row[0] for row in result]
        
        return DatabaseMetadata(
            database_name="main",
            table_count=len(tables),
            tables=tables
        )
    
    def get_table_info(self, table_name: str) -> TableMetadata:
        """Extract table metadata."""
        # Get row count
        row_count = self.db_conn.execute(
            f"SELECT COUNT(*) FROM {table_name}"
        ).fetchone()[0]
        
        # Get columns
        result = self.db_conn.execute(f"""
            SELECT column_name
            FROM duckdb_columns()
            WHERE table_name = '{table_name}' AND schema_name = 'main'
            ORDER BY column_index
        """).fetchall()
        
        columns = [row[0] for row in result]
        
        return TableMetadata(
            database_name="main",
            table_name=table_name,
            row_count=row_count,
            column_count=len(columns),
            columns=columns
        )
    
    def get_column_info(self, table_name: str, column_name: str) -> ColumnMetadata:
        """Extract column metadata with statistics."""
        # Get column metadata
        result = self.db_conn.execute(f"""
            SELECT data_type, is_nullable
            FROM duckdb_columns()
            WHERE table_name = '{table_name}' 
                AND column_name = '{column_name}'
                AND schema_name = 'main'
        """).fetchone()
        
        data_type, is_nullable = result
        
        # Get statistics
        stats = self.db_conn.execute(f"""
            SELECT 
                COUNT(*) as row_count,
                COUNT(DISTINCT "{column_name}") as distinct_count,
                COUNT(*) - COUNT("{column_name}") as null_count
            FROM {table_name}
        """).fetchone()
        
        row_count, distinct_count, null_count = stats
        
        # Try to get min/max/avg (for numeric/temporal columns)
        min_val, max_val, avg_val = None, None, None
        try:
            result = self.db_conn.execute(f"""
                SELECT 
                    MIN("{column_name}")::VARCHAR as min_val,
                    MAX("{column_name}")::VARCHAR as max_val,
                    AVG("{column_name}"::DOUBLE) as avg_val
                FROM {table_name}
            """).fetchone()
            min_val, max_val, avg_val = result
        except:
            pass  # Non-numeric column
        
        return ColumnMetadata(
            database_name="main",
            table_name=table_name,
            column_name=column_name,
            data_type=data_type,
            nullable=is_nullable == "YES",
            row_count=row_count,
            distinct_count=distinct_count,
            null_count=null_count,
            min_value=min_val,
            max_value=max_val,
            avg_value=avg_val
        )
    
    def ingest_column_data(self, table_name: str, column_name: str) -> HLLSet:
        """
        Ingest column data as HLLSet.
        
        Converts all values to strings and tokenizes.
        Special handling for NULLs, numerics, dates.
        """
        # Fetch all non-NULL values
        result = self.db_conn.execute(f"""
            SELECT DISTINCT "{column_name}"::VARCHAR as val
            FROM {table_name}
            WHERE "{column_name}" IS NOT NULL
        """).fetchall()
        
        # Convert to tokens
        tokens = []
        for (val,) in result:
            if val is not None:
                tokens.append(str(val))
        
        # Add NULL token if present
        null_count = self.db_conn.execute(f"""
            SELECT COUNT(*) FROM {table_name} WHERE "{column_name}" IS NULL
        """).fetchone()[0]
        
        if null_count > 0:
            tokens.append("__NULL__")
        
        # Create HLLSet
        hllset = HLLSet.from_batch(tokens)
        return hllset
    
    def ingest_table(self, table_name: str) -> Tuple[HLLSet, TableMetadata, Dict[str, Tuple[HLLSet, ColumnMetadata]]]:
        """
        Ingest entire table column by column.
        
        Returns:
            - Table HLLSet (union of all columns)
            - Table metadata
            - Dictionary mapping column names to (HLLSet, metadata) tuples
        """
        print(f"\nIngesting table: {table_name}")
        
        # Get table metadata
        table_meta = self.get_table_info(table_name)
        print(f"  Columns: {table_meta.column_count}, Rows: {table_meta.row_count:,}")
        
        # Ingest each column
        column_data = {}
        table_hllset = HLLSet()  # Start with empty set
        
        for col_name in table_meta.columns:
            print(f"    Processing column: {col_name}...", end=' ')
            
            # Get column data and metadata
            col_hllset = self.ingest_column_data(table_name, col_name)
            col_meta = self.get_column_info(table_name, col_name)
            
            # Store
            column_data[col_name] = (col_hllset, col_meta)
            
            # Add to table HLLSet
            table_hllset = table_hllset.union(col_hllset)
            
            print(f"✓ ({col_hllset.cardinality()} distinct values)")
        
        return table_hllset, table_meta, column_data
    
    def ingest_database(self, db_path: Path) -> Dict[str, Any]:
        """
        Ingest entire database.
        
        Creates complete hierarchy and stores in ManifoldOS.
        
        Returns:
            Dictionary with ingestion summary and artifact IDs
        """
        print("="*70)
        print(f"COLUMNAR DATABASE INGESTION")
        print("="*70)
        
        # Connect to database
        self.connect(db_path)
        
        # Get database info
        db_meta = self.get_database_info()
        print(f"\nDatabase: {db_meta.database_name}")
        print(f"Tables: {db_meta.table_count}")
        
        # Store database metadata
        db_meta_fingerprint = HLLSet.from_batch(db_meta.to_tokens())
        db_meta_id = self.manifold.store_artifact(
            json.dumps(db_meta.to_dict(), indent=2).encode(),
            metadata={'type': 'database_metadata', 'name': db_meta.database_name}
        )
        
        # Process each table
        db_hllset = HLLSet()  # Union of all tables
        table_artifacts = {}
        
        for table_name in db_meta.tables:
            # Ingest table
            table_hllset, table_meta, column_data = self.ingest_table(table_name)
            
            # Store table data HLLSet
            table_data_id = self.manifold.store_artifact(
                table_hllset.dump_roaring(),
                metadata={'type': 'table_data', 'table': table_name}
            )
            
            # Store table metadata
            table_meta_fingerprint = HLLSet.from_batch(table_meta.to_tokens())
            table_meta_id = self.manifold.store_artifact(
                json.dumps(table_meta.to_dict(), indent=2).encode(),
                metadata={'type': 'table_metadata', 'table': table_name}
            )
            
            # Store column data and metadata
            column_artifacts = {}
            for col_name, (col_hllset, col_meta) in column_data.items():
                # Store column data HLLSet
                col_data_id = self.manifold.store_artifact(
                    col_hllset.dump_roaring(),
                    metadata={'type': 'column_data', 'table': table_name, 'column': col_name}
                )
                
                # Store column metadata (raw)
                col_meta_id = self.manifold.store_artifact(
                    json.dumps(col_meta.to_dict(), indent=2).encode(),
                    metadata={'type': 'column_metadata', 'table': table_name, 'column': col_name}
                )
                
                # Store column metadata fingerprint (HLLSet)
                col_meta_fingerprint = HLLSet.from_batch(col_meta.to_tokens())
                col_meta_fp_id = self.manifold.store_artifact(
                    col_meta_fingerprint.dump_roaring(),
                    metadata={'type': 'column_metadata_fingerprint', 'table': table_name, 'column': col_name}
                )
                
                # Create explicit entanglement: data ↔ metadata
                self.manifold.create_entanglement(col_data_id, col_meta_id, strength=1.0)
                self.manifold.create_entanglement(col_data_id, col_meta_fp_id, strength=0.8)
                
                column_artifacts[col_name] = {
                    'data_id': col_data_id,
                    'metadata_id': col_meta_id,
                    'fingerprint_id': col_meta_fp_id,
                    'sha1': col_meta.sha1_id
                }
            
            # Create table-level entanglements
            self.manifold.create_entanglement(table_data_id, table_meta_id, strength=1.0)
            
            # Add to database HLLSet
            db_hllset = db_hllset.union(table_hllset)
            
            table_artifacts[table_name] = {
                'data_id': table_data_id,
                'metadata_id': table_meta_id,
                'sha1': table_meta.sha1_id,
                'columns': column_artifacts
            }
        
        # Store database data HLLSet
        db_data_id = self.manifold.store_artifact(
            db_hllset.dump_roaring(),
            metadata={'type': 'database_data', 'name': db_meta.database_name}
        )
        
        # Create database-level entanglement
        self.manifold.create_entanglement(db_data_id, db_meta_id, strength=1.0)
        
        self.disconnect()
        
        # Summary
        print("\n" + "="*70)
        print("INGESTION COMPLETE")
        print("="*70)
        print(f"Database HLLSet cardinality: {db_hllset.cardinality():,}")
        print(f"Tables ingested: {len(table_artifacts)}")
        
        total_columns = sum(len(t['columns']) for t in table_artifacts.values())
        print(f"Columns ingested: {total_columns}")
        
        return {
            'database': {
                'data_id': db_data_id,
                'metadata_id': db_meta_id,
                'sha1': db_meta.sha1_id,
                'cardinality': db_hllset.cardinality()
            },
            'tables': table_artifacts
        }
