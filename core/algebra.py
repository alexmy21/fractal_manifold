"""
HLLSet Relational Algebra Engine

Implements the relational algebra homomorphism described in
DOCS/HLLSET_RELATIONAL_ALGEBRA.md:

    SQL/Relational Algebra  ──homomorphism──▶  HLLSet Operations

Layers:
    HLLCatalog       – Ingest DuckDB → per-column HLLSets + metadata
    RelAlgebra       – Composable relational operators on HLLSets
    QueryResult      – Structured output with estimates + confidence

Design Principles:
    - Zero raw-data access after ingestion (privacy-preserving)
    - O(m) per operation (m = register count, constant space)
    - All HLLSet operations return composable HLLSets
    - Confidence intervals via standard error propagation
    - Content-addressed: every artifact has a SHA1 name

Usage:
    from core.algebra import HLLCatalog, RelAlgebra

    # Ingest once
    catalog = HLLCatalog.from_duckdb("data/finance_data.duckdb", p_bits=14)

    # Query forever — zero data access
    ra = RelAlgebra(catalog)

    # COUNT DISTINCT
    r = ra.count_distinct("orders", "customer_id")

    # JOIN size estimation
    r = ra.join_estimate("orders", "customer_id", "customers", "id")

    # FK discovery
    fks = ra.discover_foreign_keys(threshold=0.5)

    # Overlap matrix
    mat = ra.overlap_matrix(["orders.customer_id", "customers.id", ...])
"""

from __future__ import annotations

import json
import math
import hashlib
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from functools import reduce

from core.hllset import HLLSet
from core.constants import P_BITS, SHARED_SEED


# ═════════════════════════════════════════════════════════════════════════
#  Data classes
# ═════════════════════════════════════════════════════════════════════════

@dataclass
class ColumnProfile:
    """HLLSet fingerprint + metadata for one database column."""
    table_name: str
    column_name: str
    data_type: str
    row_count: int
    distinct_count: int          # exact (from ingestion query)
    null_count: int
    hllset: HLLSet               # probabilistic fingerprint
    hll_cardinality: float = 0.0 # estimated from HLLSet
    min_value: Optional[str] = None
    max_value: Optional[str] = None

    def __post_init__(self):
        self.hll_cardinality = self.hllset.cardinality()

    @property
    def fqn(self) -> str:
        """Fully-qualified name: table.column"""
        return f"{self.table_name}.{self.column_name}"

    @property
    def selectivity(self) -> float:
        """Fraction of distinct values: distinct / rows."""
        return self.distinct_count / self.row_count if self.row_count else 0.0

    @property
    def null_fraction(self) -> float:
        return self.null_count / self.row_count if self.row_count else 0.0

    def to_dict(self) -> dict:
        return {
            'table': self.table_name,
            'column': self.column_name,
            'type': self.data_type,
            'rows': self.row_count,
            'distinct': self.distinct_count,
            'hll_cardinality': round(self.hll_cardinality),
            'nulls': self.null_count,
            'null_frac': round(self.null_fraction, 4),
            'selectivity': round(self.selectivity, 4),
            'hll_name': self.hllset.short_name,
            'min': self.min_value,
            'max': self.max_value,
        }


@dataclass
class TableProfile:
    """Metadata for one database table."""
    table_name: str
    row_count: int
    column_count: int
    columns: List[str]
    table_hllset: HLLSet       # union of all column HLLSets

    @property
    def hll_cardinality(self) -> float:
        return self.table_hllset.cardinality()

    def to_dict(self) -> dict:
        return {
            'table': self.table_name,
            'rows': self.row_count,
            'columns': self.column_count,
            'hll_cardinality': round(self.hll_cardinality),
            'hll_name': self.table_hllset.short_name,
        }


@dataclass
class QueryResult:
    """
    Structured result from a relational algebra operation.

    Carries the estimated value, the HLLSet(s) that produced it,
    and an error-bound estimate so callers can decide whether to
    trust the approximation or fall back to exact computation.
    """
    operation: str                  # human-readable description
    estimate: float                 # primary result value
    confidence: float = 0.95       # confidence level
    error_bound: float = 0.0       # ± absolute error estimate
    error_pct: float = 0.0         # ± relative error %
    details: Dict[str, Any] = field(default_factory=dict)
    hllset: Optional[HLLSet] = None  # result HLLSet (composable)

    @property
    def low(self) -> float:
        return max(0.0, self.estimate - self.error_bound)

    @property
    def high(self) -> float:
        return self.estimate + self.error_bound

    def __repr__(self) -> str:
        if self.error_pct > 0:
            return (f"QueryResult({self.operation}: "
                    f"≈{self.estimate:,.0f} ±{self.error_pct:.1f}%)")
        return f"QueryResult({self.operation}: ≈{self.estimate:,.0f})"

    def to_dict(self) -> dict:
        d = {
            'operation': self.operation,
            'estimate': round(self.estimate),
            'error_bound': round(self.error_bound, 1),
            'error_pct': round(self.error_pct, 2),
            'range': [round(self.low), round(self.high)],
        }
        d.update(self.details)
        return d


# ═════════════════════════════════════════════════════════════════════════
#  HLLCatalog — the "shadow database"
# ═════════════════════════════════════════════════════════════════════════

class HLLCatalog:
    """
    In-memory catalog of HLLSet fingerprints for every column in a database.

    After construction the catalog holds ONLY HLLSets and metadata —
    zero raw data.  All subsequent analytics go through RelAlgebra
    which never touches the source database.

    Persistence:
        catalog.save("catalog.json")         # metadata only
        catalog = HLLCatalog.load("catalog.json")  # restore
    """

    def __init__(self, db_name: str = "unknown", p_bits: int = P_BITS):
        self.db_name = db_name
        self.p_bits = p_bits
        self.tables: Dict[str, TableProfile] = {}
        self.columns: Dict[str, ColumnProfile] = {}  # keyed by "table.column"

    # ------------------------------------------------------------------
    #  Factory: ingest from DuckDB
    # ------------------------------------------------------------------

    @classmethod
    def from_duckdb(cls, db_path: Union[str, Path],
                    p_bits: int = P_BITS) -> HLLCatalog:
        """
        Ingest a DuckDB database into an HLL catalog.

        Reads every column once, converts distinct values to HLLSets,
        then discards the connection.  After this call, zero data access.
        """
        import duckdb

        db_path = Path(db_path)
        catalog = cls(db_name=db_path.stem, p_bits=p_bits)

        # Use config to allow concurrent access when another process holds lock
        conn = duckdb.connect(str(db_path), read_only=True) #, config={'access_mode': 'read_only'})

        # Discover tables
        table_names = [r[0] for r in conn.execute("""
            SELECT table_name FROM duckdb_tables()
            WHERE schema_name = 'main'
            ORDER BY table_name
        """).fetchall()]

        print(f"Ingesting {len(table_names)} tables from {db_path.name} "
              f"(P={p_bits}, m={2**p_bits})")

        for tbl in table_names:
            # Table-level info
            row_count = conn.execute(
                f'SELECT COUNT(*) FROM "{tbl}"'
            ).fetchone()[0]

            col_rows = conn.execute(f"""
                SELECT column_name, data_type
                FROM duckdb_columns()
                WHERE table_name = '{tbl}' AND schema_name = 'main'
                ORDER BY column_index
            """).fetchall()

            col_names = [r[0] for r in col_rows]
            col_types = {r[0]: r[1] for r in col_rows}

            print(f"  {tbl:40s}  {row_count:>8,} rows  {len(col_names):>3} cols",
                  end="")

            table_hllset = HLLSet(p_bits=p_bits)  # start empty

            for col_name in col_names:
                # Distinct values → tokens
                vals = conn.execute(f"""
                    SELECT DISTINCT "{col_name}"::VARCHAR AS val
                    FROM "{tbl}"
                    WHERE "{col_name}" IS NOT NULL
                """).fetchall()
                tokens = [str(v[0]) for v in vals if v[0] is not None]

                # Stats
                stats = conn.execute(f"""
                    SELECT
                        COUNT(DISTINCT "{col_name}") AS dist,
                        COUNT(*) - COUNT("{col_name}") AS nulls
                    FROM "{tbl}"
                """).fetchone()
                distinct_count, null_count = stats

                # Min/Max (best effort)
                min_val = max_val = None
                try:
                    mm = conn.execute(f"""
                        SELECT MIN("{col_name}")::VARCHAR,
                               MAX("{col_name}")::VARCHAR
                        FROM "{tbl}"
                    """).fetchone()
                    min_val, max_val = mm
                except Exception:
                    pass

                # Build HLLSet
                if tokens:
                    col_hll = HLLSet.from_batch(tokens, p_bits=p_bits)
                else:
                    col_hll = HLLSet(p_bits=p_bits)

                col_profile = ColumnProfile(
                    table_name=tbl,
                    column_name=col_name,
                    data_type=col_types[col_name],
                    row_count=row_count,
                    distinct_count=distinct_count,
                    null_count=null_count,
                    hllset=col_hll,
                    min_value=min_val,
                    max_value=max_val,
                )
                catalog.columns[col_profile.fqn] = col_profile

                # Accumulate into table-level HLLSet
                table_hllset = table_hllset.union(col_hll)

            tbl_profile = TableProfile(
                table_name=tbl,
                row_count=row_count,
                column_count=len(col_names),
                columns=col_names,
                table_hllset=table_hllset,
            )
            catalog.tables[tbl] = tbl_profile
            print(f"  → |table|≈{table_hllset.cardinality():.0f}")

        conn.close()
        print(f"Catalog ready: {len(catalog.tables)} tables, "
              f"{len(catalog.columns)} columns")
        return catalog

    # ------------------------------------------------------------------
    #  Persistence
    # ------------------------------------------------------------------

    def save(self, path: Union[str, Path]):
        """
        Save catalog to directory: metadata JSON + per-column Roaring blobs.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Metadata
        meta = {
            'db_name': self.db_name,
            'p_bits': self.p_bits,
            'tables': {k: v.to_dict() for k, v in self.tables.items()},
            'columns': {k: v.to_dict() for k, v in self.columns.items()},
        }
        (path / "catalog.json").write_text(
            json.dumps(meta, indent=2, default=str)
        )

        # HLLSet blobs
        blob_dir = path / "blobs"
        blob_dir.mkdir(exist_ok=True)

        for fqn, col in self.columns.items():
            safe = fqn.replace(".", "__")
            (blob_dir / f"{safe}.roaring").write_bytes(col.hllset.dump_roaring())

        # Table-level HLLSets
        for tbl_name, tbl in self.tables.items():
            (blob_dir / f"_table_{tbl_name}.roaring").write_bytes(
                tbl.table_hllset.dump_roaring()
            )

        print(f"Catalog saved to {path}/  "
              f"({len(self.columns)} column blobs)")

    @classmethod
    def load(cls, path: Union[str, Path]) -> HLLCatalog:
        """Load catalog from directory previously created by save()."""
        path = Path(path)
        meta = json.loads((path / "catalog.json").read_text())

        catalog = cls(db_name=meta['db_name'], p_bits=meta['p_bits'])
        p_bits = meta['p_bits']
        blob_dir = path / "blobs"

        # Rebuild columns
        for fqn, col_meta in meta['columns'].items():
            safe = fqn.replace(".", "__")
            blob_path = blob_dir / f"{safe}.roaring"
            hll = HLLSet.from_roaring(blob_path.read_bytes(), p_bits=p_bits)

            catalog.columns[fqn] = ColumnProfile(
                table_name=col_meta['table'],
                column_name=col_meta['column'],
                data_type=col_meta['type'],
                row_count=col_meta['rows'],
                distinct_count=col_meta['distinct'],
                null_count=col_meta['nulls'],
                hllset=hll,
                min_value=col_meta.get('min'),
                max_value=col_meta.get('max'),
            )

        # Rebuild tables
        for tbl_name, tbl_meta in meta['tables'].items():
            blob_path = blob_dir / f"_table_{tbl_name}.roaring"
            tbl_hll = HLLSet.from_roaring(blob_path.read_bytes(), p_bits=p_bits)

            catalog.tables[tbl_name] = TableProfile(
                table_name=tbl_name,
                row_count=tbl_meta['rows'],
                column_count=tbl_meta['columns'],
                columns=list(meta['columns'].keys()),  # simplified
                table_hllset=tbl_hll,
            )
            # Fix: store only columns belonging to this table
            catalog.tables[tbl_name].columns = [
                c.column_name for c in catalog.columns.values()
                if c.table_name == tbl_name
            ]

        print(f"Catalog loaded from {path}/  "
              f"({len(catalog.tables)} tables, {len(catalog.columns)} columns)")
        return catalog

    # ------------------------------------------------------------------
    #  Lookup helpers
    # ------------------------------------------------------------------

    def get_column(self, table: str, column: str) -> ColumnProfile:
        """Lookup column profile by table + column name."""
        fqn = f"{table}.{column}"
        if fqn not in self.columns:
            raise KeyError(f"Column not found: {fqn}  "
                           f"(available: {list(self.columns.keys())[:10]}...)")
        return self.columns[fqn]

    def get_hllset(self, table: str, column: str) -> HLLSet:
        """Get the HLLSet for a specific column."""
        return self.get_column(table, column).hllset

    def get_table(self, table: str) -> TableProfile:
        """Lookup table profile."""
        if table not in self.tables:
            raise KeyError(f"Table not found: {table}  "
                           f"(available: {list(self.tables.keys())})")
        return self.tables[table]

    def all_columns(self) -> List[ColumnProfile]:
        """All column profiles, sorted by fqn."""
        return sorted(self.columns.values(), key=lambda c: c.fqn)

    def table_columns(self, table: str) -> List[ColumnProfile]:
        """All columns belonging to a table."""
        return [c for c in self.columns.values() if c.table_name == table]

    def summary(self) -> str:
        """Human-readable catalog summary."""
        lines = [f"HLLCatalog: {self.db_name}  (P={self.p_bits})"]
        lines.append(f"  Tables: {len(self.tables)}  "
                     f"Columns: {len(self.columns)}")
        for tbl_name, tbl in sorted(self.tables.items()):
            lines.append(f"  {tbl_name:40s}  "
                         f"{tbl.row_count:>8,} rows  "
                         f"{tbl.column_count:>3} cols  "
                         f"|table|≈{tbl.hll_cardinality:.0f}")
            for col in self.table_columns(tbl_name):
                lines.append(
                    f"    {col.column_name:36s}  "
                    f"{col.data_type:12s}  "
                    f"distinct={col.distinct_count:>8,}  "
                    f"|hll|≈{col.hll_cardinality:>8,.0f}"
                )
        return "\n".join(lines)


# ═════════════════════════════════════════════════════════════════════════
#  RelAlgebra — the query engine
# ═════════════════════════════════════════════════════════════════════════

class RelAlgebra:
    """
    Relational algebra operations on HLLSet catalogs.

    Every method returns a QueryResult with:
        - .estimate   – the numeric answer
        - .error_pct  – estimated relative error
        - .hllset     – composable HLLSet for further operations
        - .details    – operation-specific metadata

    All operations are O(m) time, O(1) space, zero data access.
    """

    def __init__(self, catalog: HLLCatalog):
        self.catalog = catalog
        self._std_err = 1.04 / math.sqrt(2 ** catalog.p_bits)

    @property
    def standard_error(self) -> float:
        """HLL standard error: 1.04/√m"""
        return self._std_err

    # ------------------------------------------------------------------
    #  Helpers
    # ------------------------------------------------------------------

    def _error_bound(self, estimate: float, n_ops: int = 1) -> Tuple[float, float]:
        """
        Compute error bound for an estimate after n_ops operations.

        Union:        error ≈ 1× std_err
        Intersection: error ≈ 3× std_err per operation (via I-E)
        Chain of k:   error ≈ k × 3 × std_err

        Returns (abs_error, pct_error).
        """
        # Each intersection roughly triples the relative error
        relative = self._std_err * (1 + 2 * n_ops)
        abs_error = estimate * relative * 1.96  # 95% CI
        pct = relative * 1.96 * 100
        return abs_error, pct

    def _resolve(self, spec: str) -> Tuple[str, str]:
        """Parse 'table.column' into (table, column)."""
        parts = spec.split(".", 1)
        if len(parts) != 2:
            raise ValueError(
                f"Column spec must be 'table.column', got: {spec!r}")
        return parts[0], parts[1]

    # ------------------------------------------------------------------
    #  § 1  COUNT DISTINCT
    # ------------------------------------------------------------------

    def count_distinct(self, table: str, column: str) -> QueryResult:
        """
        SELECT COUNT(DISTINCT column) FROM table

        Complexity: O(m)  vs  O(n log n) exact
        """
        col = self.catalog.get_column(table, column)
        est = col.hll_cardinality
        abs_err, pct_err = self._error_bound(est, n_ops=0)

        return QueryResult(
            operation=f"COUNT(DISTINCT {table}.{column})",
            estimate=est,
            error_bound=abs_err,
            error_pct=pct_err,
            hllset=col.hllset,
            details={
                'exact_distinct': col.distinct_count,
                'data_type': col.data_type,
                'row_count': col.row_count,
                'selectivity': round(col.selectivity, 4),
            },
        )

    # ------------------------------------------------------------------
    #  § 2  UNION
    # ------------------------------------------------------------------

    def union(self, table1: str, col1: str,
              table2: str, col2: str) -> QueryResult:
        """
        SELECT DISTINCT col FROM table1
        UNION
        SELECT DISTINCT col FROM table2
        """
        h1 = self.catalog.get_hllset(table1, col1)
        h2 = self.catalog.get_hllset(table2, col2)
        u = h1.union(h2)
        est = u.cardinality()
        abs_err, pct_err = self._error_bound(est, n_ops=0)

        return QueryResult(
            operation=f"UNION({table1}.{col1}, {table2}.{col2})",
            estimate=est,
            error_bound=abs_err,
            error_pct=pct_err,
            hllset=u,
            details={
                'left_card': h1.cardinality(),
                'right_card': h2.cardinality(),
            },
        )

    def union_all(self, *specs: str) -> QueryResult:
        """
        Union of multiple columns:  union_all("t1.c1", "t2.c2", "t3.c3")
        """
        hlls = []
        for spec in specs:
            t, c = self._resolve(spec)
            hlls.append(self.catalog.get_hllset(t, c))

        result = reduce(lambda a, b: a.union(b), hlls)
        est = result.cardinality()
        abs_err, pct_err = self._error_bound(est, n_ops=0)

        return QueryResult(
            operation=f"UNION({', '.join(specs)})",
            estimate=est,
            error_bound=abs_err,
            error_pct=pct_err,
            hllset=result,
            details={'input_count': len(specs)},
        )

    # ------------------------------------------------------------------
    #  § 3  INTERSECTION
    # ------------------------------------------------------------------

    def intersect(self, table1: str, col1: str,
                  table2: str, col2: str) -> QueryResult:
        """
        Equivalent to:
            SELECT DISTINCT a.col FROM t1 a
            INNER JOIN t2 b ON a.col = b.col
        """
        h1 = self.catalog.get_hllset(table1, col1)
        h2 = self.catalog.get_hllset(table2, col2)
        inter = h1.intersect(h2)
        est = inter.cardinality()
        abs_err, pct_err = self._error_bound(est, n_ops=1)

        return QueryResult(
            operation=f"INTERSECT({table1}.{col1}, {table2}.{col2})",
            estimate=est,
            error_bound=abs_err,
            error_pct=pct_err,
            hllset=inter,
            details={
                'left_card': h1.cardinality(),
                'right_card': h2.cardinality(),
            },
        )

    # ------------------------------------------------------------------
    #  § 4  DIFFERENCE
    # ------------------------------------------------------------------

    def difference(self, table1: str, col1: str,
                   table2: str, col2: str) -> QueryResult:
        """
        Values in table1.col1 NOT in table2.col2.

        Equivalent to:
            SELECT DISTINCT col FROM t1
            EXCEPT
            SELECT DISTINCT col FROM t2
        """
        h1 = self.catalog.get_hllset(table1, col1)
        h2 = self.catalog.get_hllset(table2, col2)
        diff = h1.diff(h2)
        est = diff.cardinality()
        abs_err, pct_err = self._error_bound(est, n_ops=1)

        return QueryResult(
            operation=f"EXCEPT({table1}.{col1}, {table2}.{col2})",
            estimate=est,
            error_bound=abs_err,
            error_pct=pct_err,
            hllset=diff,
            details={
                'left_card': h1.cardinality(),
                'right_card': h2.cardinality(),
            },
        )

    # ------------------------------------------------------------------
    #  § 5  SYMMETRIC DIFFERENCE
    # ------------------------------------------------------------------

    def symmetric_difference(self, table1: str, col1: str,
                             table2: str, col2: str) -> QueryResult:
        """Values in exactly one of the two columns (XOR)."""
        h1 = self.catalog.get_hllset(table1, col1)
        h2 = self.catalog.get_hllset(table2, col2)
        xor = h1.xor(h2)
        est = xor.cardinality()
        abs_err, pct_err = self._error_bound(est, n_ops=1)

        return QueryResult(
            operation=f"XOR({table1}.{col1}, {table2}.{col2})",
            estimate=est,
            error_bound=abs_err,
            error_pct=pct_err,
            hllset=xor,
            details={
                'left_card': h1.cardinality(),
                'right_card': h2.cardinality(),
            },
        )

    # ------------------------------------------------------------------
    #  § 6  JOIN SIZE ESTIMATION
    # ------------------------------------------------------------------

    def join_estimate(self, left_table: str, left_col: str,
                      right_table: str, right_col: str) -> QueryResult:
        """
        Estimate join cardinality:
            SELECT COUNT(*)
            FROM left_table l
            JOIN right_table r ON l.left_col = r.right_col

        Uses: join_size ≈ left_rows × (overlap / left_distinct)
        Assumption: uniform value distribution.
        """
        left_prof = self.catalog.get_column(left_table, left_col)
        right_prof = self.catalog.get_column(right_table, right_col)

        h_left = left_prof.hllset
        h_right = right_prof.hllset

        overlap = h_left.intersect(h_right)
        overlap_card = overlap.cardinality()
        left_card = h_left.cardinality()

        if left_card == 0:
            selectivity = 0.0
        else:
            selectivity = overlap_card / left_card

        join_size = left_prof.row_count * selectivity
        abs_err, pct_err = self._error_bound(join_size, n_ops=1)

        return QueryResult(
            operation=(f"JOIN({left_table}.{left_col} = "
                       f"{right_table}.{right_col})"),
            estimate=join_size,
            error_bound=abs_err,
            error_pct=pct_err,
            hllset=overlap,
            details={
                'left_rows': left_prof.row_count,
                'right_rows': right_prof.row_count,
                'left_distinct': left_card,
                'right_distinct': right_prof.hll_cardinality,
                'overlap_distinct': overlap_card,
                'selectivity': round(selectivity, 4),
            },
        )

    # ------------------------------------------------------------------
    #  § 7  SELECTIVITY ESTIMATION  (WHERE col IN (...))
    # ------------------------------------------------------------------

    def selectivity_in(self, table: str, column: str,
                       values: List[str]) -> QueryResult:
        """
        Estimate:
            SELECT COUNT(*) FROM table WHERE column IN (values)

        Builds a transient HLLSet from the predicate values, intersects
        with the column fingerprint, and uses the cardinality ratio as
        selectivity.
        """
        col = self.catalog.get_column(table, column)
        query_hll = HLLSet.from_batch(values, p_bits=self.catalog.p_bits)
        match_hll = col.hllset.intersect(query_hll)

        match_card = match_hll.cardinality()
        col_card = col.hll_cardinality

        if col_card == 0:
            selectivity = 0.0
        else:
            selectivity = match_card / col_card

        estimated_rows = col.row_count * selectivity
        abs_err, pct_err = self._error_bound(estimated_rows, n_ops=1)

        return QueryResult(
            operation=f"SELECT FROM {table} WHERE {column} IN ({len(values)} values)",
            estimate=estimated_rows,
            error_bound=abs_err,
            error_pct=pct_err,
            hllset=match_hll,
            details={
                'query_values': len(values),
                'matching_distinct': match_card,
                'column_distinct': col_card,
                'selectivity': round(selectivity, 4),
                'row_count': col.row_count,
            },
        )

    # ------------------------------------------------------------------
    #  § 8  SIMILARITY & OVERLAP
    # ------------------------------------------------------------------

    def jaccard(self, table1: str, col1: str,
                table2: str, col2: str) -> QueryResult:
        """
        Jaccard similarity: |A ∩ B| / |A ∪ B|.

        High Jaccard (>0.7) suggests FK relationship.
        """
        h1 = self.catalog.get_hllset(table1, col1)
        h2 = self.catalog.get_hllset(table2, col2)

        inter = h1.intersect(h2)
        union = h1.union(h2)

        inter_card = inter.cardinality()
        union_card = union.cardinality()

        if union_card == 0:
            j = 0.0
        else:
            j = inter_card / union_card

        return QueryResult(
            operation=f"JACCARD({table1}.{col1}, {table2}.{col2})",
            estimate=j,
            details={
                'intersection': inter_card,
                'union': union_card,
                'left_card': h1.cardinality(),
                'right_card': h2.cardinality(),
            },
        )

    def containment(self, table1: str, col1: str,
                    table2: str, col2: str) -> QueryResult:
        """
        Containment: |A ∩ B| / |A|.

        Measures how much of column A's values appear in column B.
        Containment ≈ 1.0 means A is likely a FK referencing B.
        """
        h1 = self.catalog.get_hllset(table1, col1)
        h2 = self.catalog.get_hllset(table2, col2)
        inter = h1.intersect(h2)

        inter_card = inter.cardinality()
        h1_card = h1.cardinality()

        if h1_card == 0:
            c = 0.0
        else:
            c = inter_card / h1_card

        return QueryResult(
            operation=f"CONTAINMENT({table1}.{col1} ⊆ {table2}.{col2})",
            estimate=c,
            details={
                'intersection': inter_card,
                'left_card': h1_card,
                'right_card': h2.cardinality(),
            },
        )

    # ------------------------------------------------------------------
    #  § 9  DISCOVERY OPERATIONS
    # ------------------------------------------------------------------

    def discover_foreign_keys(self,
                              threshold: float = 0.5,
                              min_distinct: int = 2) -> List[Dict[str, Any]]:
        """
        Discover likely FK relationships across all table pairs.

        For each (colA, colB) pair with containment ≥ threshold,
        report as candidate FK.

        Args:
            threshold: Minimum containment to report (0.0–1.0)
            min_distinct: Skip columns with fewer distinct values

        Returns:
            Sorted list of FK candidates with containment scores.
        """
        candidates = []
        cols = [c for c in self.catalog.all_columns()
                if c.hll_cardinality >= min_distinct]

        for i, ca in enumerate(cols):
            for cb in cols[i + 1:]:
                if ca.table_name == cb.table_name:
                    continue  # skip same-table pairs

                # A → B containment
                inter = ca.hllset.intersect(cb.hllset)
                inter_card = inter.cardinality()

                if ca.hll_cardinality > 0:
                    c_ab = inter_card / ca.hll_cardinality
                    if c_ab >= threshold:
                        candidates.append({
                            'from': ca.fqn,
                            'to': cb.fqn,
                            'containment': round(c_ab, 3),
                            'overlap': round(inter_card),
                            'from_card': round(ca.hll_cardinality),
                            'to_card': round(cb.hll_cardinality),
                        })

                # B → A containment
                if cb.hll_cardinality > 0:
                    c_ba = inter_card / cb.hll_cardinality
                    if c_ba >= threshold:
                        candidates.append({
                            'from': cb.fqn,
                            'to': ca.fqn,
                            'containment': round(c_ba, 3),
                            'overlap': round(inter_card),
                            'from_card': round(cb.hll_cardinality),
                            'to_card': round(ca.hll_cardinality),
                        })

        candidates.sort(key=lambda x: x['containment'], reverse=True)
        return candidates

    def overlap_matrix(self, *specs: str) -> Dict[str, Dict[str, float]]:
        """
        Compute pairwise Jaccard overlap matrix for specified columns.

        Usage:
            mat = ra.overlap_matrix(
                "orders.customer_id", "customers.id", "payments.customer_id"
            )
        Returns nested dict: mat["orders.customer_id"]["customers.id"] = 0.85
        """
        # Resolve all specs to HLLSets
        entries = []
        for spec in specs:
            t, c = self._resolve(spec)
            entries.append((spec, self.catalog.get_hllset(t, c)))

        matrix: Dict[str, Dict[str, float]] = {}
        for i, (name_a, hll_a) in enumerate(entries):
            row: Dict[str, float] = {}
            for j, (name_b, hll_b) in enumerate(entries):
                if i == j:
                    row[name_b] = 1.0
                else:
                    inter = hll_a.intersect(hll_b)
                    union = hll_a.union(hll_b)
                    inter_c = inter.cardinality()
                    union_c = union.cardinality()
                    row[name_b] = round(
                        inter_c / union_c if union_c > 0 else 0.0, 4
                    )
            matrix[name_a] = row

        return matrix

    # ------------------------------------------------------------------
    #  § 10  COMPOSITION — chain operations
    # ------------------------------------------------------------------

    def compose(self, *operations) -> HLLSet:
        """
        Chain a sequence of set operations on HLLSets.

        Usage:
            result = ra.compose(
                ("intersect", "t1.c1", "t2.c2"),
                ("union", None, "t3.c3"),      # union with previous result
                ("intersect", None, "t4.c4"),
            )

        Each tuple: (operation, left_spec_or_None, right_spec)
        None for left means "use the result of the previous step".
        """
        current: Optional[HLLSet] = None

        for step in operations:
            op = step[0]
            left_spec = step[1]
            right_spec = step[2]

            # Resolve right
            t_r, c_r = self._resolve(right_spec)
            h_right = self.catalog.get_hllset(t_r, c_r)

            # Resolve left
            if left_spec is None:
                if current is None:
                    raise ValueError(
                        "First step in compose() must specify both operands")
                h_left = current
            else:
                t_l, c_l = self._resolve(left_spec)
                h_left = self.catalog.get_hllset(t_l, c_l)

            # Execute
            if op == "union":
                current = h_left.union(h_right)
            elif op == "intersect":
                current = h_left.intersect(h_right)
            elif op == "diff":
                current = h_left.diff(h_right)
            elif op == "xor":
                current = h_left.xor(h_right)
            else:
                raise ValueError(f"Unknown operation: {op}")

        return current

    # ------------------------------------------------------------------
    #  § 11  INCLUSION-EXCLUSION IDENTITY CHECK
    # ------------------------------------------------------------------

    def inclusion_exclusion(self, table1: str, col1: str,
                            table2: str, col2: str) -> QueryResult:
        """
        Verify the inclusion-exclusion identity:
            |A ∪ B| ≈ |A| + |B| - |A ∩ B|

        Returns the identity residual as a consistency check.
        """
        h1 = self.catalog.get_hllset(table1, col1)
        h2 = self.catalog.get_hllset(table2, col2)

        card_a = h1.cardinality()
        card_b = h2.cardinality()
        card_union = h1.union(h2).cardinality()
        card_inter = h1.intersect(h2).cardinality()

        ie_estimate = card_a + card_b - card_inter
        residual = abs(ie_estimate - card_union)

        return QueryResult(
            operation=(f"I-E check: {table1}.{col1} vs "
                       f"{table2}.{col2}"),
            estimate=residual,
            details={
                '|A|': round(card_a),
                '|B|': round(card_b),
                '|A∪B|': round(card_union),
                '|A∩B|': round(card_inter),
                '|A|+|B|-|A∩B|': round(ie_estimate),
                'residual': round(residual),
                'residual_pct': (round(residual / card_union * 100, 2)
                                 if card_union > 0 else 0),
            },
        )

    # ------------------------------------------------------------------
    #  § 12  DATABASE-LEVEL ANALYTICS
    # ------------------------------------------------------------------

    def database_summary(self) -> Dict[str, Any]:
        """
        Comprehensive database analytics from HLLSets alone.

        Returns high-level stats plus per-table summaries.
        """
        total_rows = sum(t.row_count for t in self.catalog.tables.values())
        total_cols = len(self.catalog.columns)

        # Union all table HLLSets for total distinct universe
        all_tables = list(self.catalog.tables.values())
        if all_tables:
            universe = reduce(
                lambda a, b: a.union(b),
                [t.table_hllset for t in all_tables]
            )
            universe_card = universe.cardinality()
        else:
            universe_card = 0

        return {
            'database': self.catalog.db_name,
            'precision': self.catalog.p_bits,
            'standard_error': f"{self.standard_error * 100:.2f}%",
            'tables': len(self.catalog.tables),
            'columns': total_cols,
            'total_rows': total_rows,
            'universe_cardinality': round(universe_card),
            'per_table': {
                name: tbl.to_dict()
                for name, tbl in sorted(self.catalog.tables.items())
            },
        }


# ═════════════════════════════════════════════════════════════════════════
#  Module exports
# ═════════════════════════════════════════════════════════════════════════

__all__ = [
    'HLLCatalog',
    'RelAlgebra',
    'QueryResult',
    'ColumnProfile',
    'TableProfile',
]
