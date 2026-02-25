"""
Manifold Algebra - Unified Processing Model

The (reg, zeros) Universal Identifier
=====================================

All structures (HLLSet, AM, W, Sheaf sections) use the same addressing:

    content → hash → (reg, zeros) → index

This invariant enables:
- Content-addressability: same content → same position everywhere
- Compatibility: sub-structures built anywhere use same addressing
- Idempotent merge: no index translation needed
- Sheaf gluing: sections at same (reg, zeros) across layers form global sections

Unified Pipeline
================

Every interaction (ingestion OR query) follows the same pipeline:

    INPUT → HLLSet → New HRT → Extend with Context → Merge → New Current

Properties:
- Sub-structure isolation: work on separate instance
- Idempotent merge: same input → same result
- Eventual consistency: parallel changes converge
- CRDT-like: commutative, associative, idempotent

Manifold Algebra Operations
===========================

Structure-agnostic operations that preserve (reg, zeros) addressing:

Projection (π):
    π_n(M)        - Extract layer n
    π_R(M)        - Extract rows R
    π_C(M)        - Extract columns C

Transform:
    T(M)          - Transpose
    N(M)          - Normalize rows
    S_α(M)        - Scale by α

Composition:
    M₁ + M₂       - Merge (add)
    M₁ ∘ M₂       - Chain (multiply)

Path:
    reach(M, S, k) - k-hop reachability
    M*            - Transitive closure

Lift/Lower:
    ↑_n(M)        - Lift 2D to layer n
    ↓(M)          - Lower 3D to 2D
"""

from __future__ import annotations
from typing import (
    List, Dict, Set, Optional, Tuple, FrozenSet, 
    Callable, Iterator, Any, Union, NamedTuple
)
from dataclasses import dataclass, field
from functools import reduce
from collections import defaultdict
import hashlib

# Internal imports
from .sparse_hrt_3d import (
    SparseHRT3D,
    Sparse3DConfig,
    SparseAM3D,
    SparseLattice3D,
    ImmutableSparseTensor3D,
    BasicHLLSet3D,
    Edge3D,
)
from .hllset import HLLSet, compute_sha1
from .constants import P_BITS

# Default hash bits (standard HLL uses 32-bit hash)
DEFAULT_H_BITS = 32


# ═══════════════════════════════════════════════════════════════════════════
# UNIVERSAL IDENTIFIER: (reg, zeros)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class UniversalID:
    """
    Universal identifier used across all structures.
    
    Computed from content hash:
        hash → reg (which register) + zeros (leading zeros)
    
    This is the "glue" that connects HLLSet, AM, W, and Sheaf sections.
    """
    reg: int      # Register index (0 to 2^p_bits - 1)
    zeros: int    # Leading zeros count (0 to h_bits - p_bits)
    layer: int    # Layer/n-gram level (0, 1, 2, ...)
    
    @classmethod
    def from_hash(cls, h: int, layer: int, p_bits: int, h_bits: int) -> 'UniversalID':
        """Compute (reg, zeros) from hash value."""
        reg = h & ((1 << p_bits) - 1)
        remaining = h >> p_bits
        zeros = 0
        max_zeros = h_bits - p_bits
        while zeros < max_zeros and (remaining & 1) == 0:
            zeros += 1
            remaining >>= 1
        return cls(reg=reg, zeros=zeros, layer=layer)
    
    @classmethod
    def from_content(cls, content: str, layer: int, p_bits: int = P_BITS, h_bits: int = DEFAULT_H_BITS) -> 'UniversalID':
        """Compute (reg, zeros) from content string."""
        h = int(hashlib.sha1(content.encode()).hexdigest()[:8], 16)
        return cls.from_hash(h, layer, p_bits, h_bits)
    
    def to_index(self, config: Sparse3DConfig) -> int:
        """Convert to matrix index."""
        return self.reg * (config.h_bits - config.p_bits + 1) + self.zeros
    
    def __repr__(self) -> str:
        return f"UID(reg={self.reg}, zeros={self.zeros}, L{self.layer})"


def content_to_index(content: str, layer: int, config: Sparse3DConfig) -> int:
    """
    Unified content → index mapping.
    
    This is the fundamental addressing function used everywhere.
    """
    uid = UniversalID.from_content(content, layer, config.p_bits, config.h_bits)
    return uid.to_index(config)


# ═══════════════════════════════════════════════════════════════════════════
# SPARSE MATRIX (2D)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class SparseMatrix:
    """
    Immutable sparse matrix.
    
    Entries stored as frozenset of (row, col, value) tuples.
    """
    entries: FrozenSet[Tuple[int, int, float]]
    shape: Tuple[int, int]
    
    @classmethod
    def empty(cls, dim: int) -> 'SparseMatrix':
        return cls(entries=frozenset(), shape=(dim, dim))
    
    @classmethod
    def from_edges(cls, edges: List[Tuple[int, int, float]], dim: int) -> 'SparseMatrix':
        return cls(entries=frozenset(edges), shape=(dim, dim))
    
    @classmethod
    def from_dict(cls, d: Dict[int, Dict[int, float]], dim: int) -> 'SparseMatrix':
        entries = frozenset(
            (row, col, val)
            for row, cols in d.items()
            for col, val in cols.items()
        )
        return cls(entries=entries, shape=(dim, dim))
    
    def to_dict(self) -> Dict[int, Dict[int, float]]:
        d: Dict[int, Dict[int, float]] = {}
        for row, col, val in self.entries:
            if row not in d:
                d[row] = {}
            d[row][col] = val
        return d
    
    def __iter__(self) -> Iterator[Tuple[int, int, float]]:
        return iter(self.entries)
    
    @property
    def nnz(self) -> int:
        return len(self.entries)


# ═══════════════════════════════════════════════════════════════════════════
# SPARSE 3D MATRIX (Layered)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class Sparse3DMatrix:
    """
    Immutable 3D sparse matrix (layered).
    
    Each layer is a SparseMatrix. Layers correspond to n-gram sizes.
    """
    layers: Tuple[SparseMatrix, ...]
    shape: Tuple[int, int, int]  # (n_layers, dim, dim)
    
    @classmethod
    def empty(cls, n_layers: int, dim: int) -> 'Sparse3DMatrix':
        layers = tuple(SparseMatrix.empty(dim) for _ in range(n_layers))
        return cls(layers=layers, shape=(n_layers, dim, dim))
    
    @classmethod
    def from_am(cls, am: SparseAM3D, config: Sparse3DConfig) -> 'Sparse3DMatrix':
        layer_matrices = []
        for n in range(config.max_n):
            edges = list(am.tensor.layer_edges(n))
            matrix = SparseMatrix.from_edges(edges, config.dimension)
            layer_matrices.append(matrix)
        return cls(layers=tuple(layer_matrices), shape=(config.max_n, config.dimension, config.dimension))
    
    @classmethod
    def from_w(cls, W: Dict[int, Dict[int, Dict[int, float]]], config: Sparse3DConfig) -> 'Sparse3DMatrix':
        layer_matrices = []
        for n in range(config.max_n):
            if n in W:
                matrix = SparseMatrix.from_dict(W[n], config.dimension)
            else:
                matrix = SparseMatrix.empty(config.dimension)
            layer_matrices.append(matrix)
        return cls(layers=tuple(layer_matrices), shape=(config.max_n, config.dimension, config.dimension))
    
    def to_edges(self) -> List[Edge3D]:
        edges = []
        for n, layer in enumerate(self.layers):
            for row, col, val in layer:
                edges.append(Edge3D(n=n, row=row, col=col, value=val))
        return edges
    
    @property
    def nnz(self) -> int:
        return sum(layer.nnz for layer in self.layers)


# ═══════════════════════════════════════════════════════════════════════════
# PROJECTION OPERATIONS (π)
# ═══════════════════════════════════════════════════════════════════════════

def project_layer(M: Sparse3DMatrix, layer: int) -> SparseMatrix:
    """π_n: Extract single layer."""
    if 0 <= layer < len(M.layers):
        return M.layers[layer]
    return SparseMatrix.empty(M.shape[1])


def project_rows(M: SparseMatrix, rows: Set[int]) -> SparseMatrix:
    """π_R: Extract subset of rows."""
    new_entries = frozenset(
        (row, col, val) for row, col, val in M.entries if row in rows
    )
    return SparseMatrix(entries=new_entries, shape=M.shape)


def project_cols(M: SparseMatrix, cols: Set[int]) -> SparseMatrix:
    """π_C: Extract subset of columns."""
    new_entries = frozenset(
        (row, col, val) for row, col, val in M.entries if col in cols
    )
    return SparseMatrix(entries=new_entries, shape=M.shape)


def project_submatrix(M: SparseMatrix, rows: Set[int], cols: Set[int]) -> SparseMatrix:
    """π_{R,C}: Extract submatrix."""
    new_entries = frozenset(
        (row, col, val) for row, col, val in M.entries 
        if row in rows and col in cols
    )
    return SparseMatrix(entries=new_entries, shape=M.shape)


# ═══════════════════════════════════════════════════════════════════════════
# TRANSFORM OPERATIONS
# ═══════════════════════════════════════════════════════════════════════════

def transpose(M: SparseMatrix) -> SparseMatrix:
    """T: Transpose (swap rows/cols)."""
    new_entries = frozenset(
        (col, row, val) for row, col, val in M.entries
    )
    return SparseMatrix(entries=new_entries, shape=(M.shape[1], M.shape[0]))


def transpose_3d(M: Sparse3DMatrix) -> Sparse3DMatrix:
    """T: Transpose all layers."""
    new_layers = tuple(transpose(layer) for layer in M.layers)
    return Sparse3DMatrix(layers=new_layers, shape=M.shape)


def normalize_rows(M: SparseMatrix) -> SparseMatrix:
    """N: Row-normalize (sum to 1). Converts AM → W."""
    row_sums: Dict[int, float] = {}
    for row, col, val in M.entries:
        row_sums[row] = row_sums.get(row, 0.0) + val
    
    new_entries = frozenset(
        (row, col, val / row_sums[row]) if row_sums.get(row, 0) > 0 else (row, col, 0.0)
        for row, col, val in M.entries
    )
    return SparseMatrix(entries=new_entries, shape=M.shape)


def normalize_3d(M: Sparse3DMatrix) -> Sparse3DMatrix:
    """N: Normalize all layers."""
    return Sparse3DMatrix(
        layers=tuple(normalize_rows(layer) for layer in M.layers),
        shape=M.shape
    )


def scale(M: SparseMatrix, factor: float) -> SparseMatrix:
    """S_α: Scale all values."""
    new_entries = frozenset(
        (row, col, val * factor) for row, col, val in M.entries
    )
    return SparseMatrix(entries=new_entries, shape=M.shape)


# ═══════════════════════════════════════════════════════════════════════════
# FILTER OPERATIONS (σ)
# ═══════════════════════════════════════════════════════════════════════════

def filter_threshold(M: SparseMatrix, min_val: float = 0.0) -> SparseMatrix:
    """σ_θ: Keep entries ≥ threshold."""
    new_entries = frozenset(
        (row, col, val) for row, col, val in M.entries if val >= min_val
    )
    return SparseMatrix(entries=new_entries, shape=M.shape)


def filter_predicate(M: SparseMatrix, pred: Callable[[int, int, float], bool]) -> SparseMatrix:
    """σ_P: Keep entries where predicate is True."""
    new_entries = frozenset(
        (row, col, val) for row, col, val in M.entries if pred(row, col, val)
    )
    return SparseMatrix(entries=new_entries, shape=M.shape)


# ═══════════════════════════════════════════════════════════════════════════
# COMPOSITION OPERATIONS
# ═══════════════════════════════════════════════════════════════════════════

def merge_add(M1: SparseMatrix, M2: SparseMatrix) -> SparseMatrix:
    """+ : Element-wise addition."""
    combined: Dict[Tuple[int, int], float] = {}
    for row, col, val in M1.entries:
        combined[(row, col)] = combined.get((row, col), 0.0) + val
    for row, col, val in M2.entries:
        combined[(row, col)] = combined.get((row, col), 0.0) + val
    
    new_entries = frozenset(
        (row, col, val) for (row, col), val in combined.items()
    )
    return SparseMatrix(entries=new_entries, shape=M1.shape)


def merge_max(M1: SparseMatrix, M2: SparseMatrix) -> SparseMatrix:
    """∨ : Element-wise maximum."""
    combined: Dict[Tuple[int, int], float] = {}
    for row, col, val in M1.entries:
        key = (row, col)
        combined[key] = max(combined.get(key, 0.0), val)
    for row, col, val in M2.entries:
        key = (row, col)
        combined[key] = max(combined.get(key, 0.0), val)
    
    return SparseMatrix(
        entries=frozenset((r, c, v) for (r, c), v in combined.items()),
        shape=M1.shape
    )


def compose_chain(M1: SparseMatrix, M2: SparseMatrix) -> SparseMatrix:
    """∘ : Matrix multiplication (path composition)."""
    m2_by_row: Dict[int, List[Tuple[int, float]]] = {}
    for row, col, val in M2.entries:
        if row not in m2_by_row:
            m2_by_row[row] = []
        m2_by_row[row].append((col, val))
    
    result: Dict[Tuple[int, int], float] = {}
    for i, j, v1 in M1.entries:
        if j in m2_by_row:
            for k, v2 in m2_by_row[j]:
                key = (i, k)
                result[key] = result.get(key, 0.0) + v1 * v2
    
    return SparseMatrix(
        entries=frozenset((r, c, v) for (r, c), v in result.items()),
        shape=M1.shape
    )


def merge_3d_add(M1: Sparse3DMatrix, M2: Sparse3DMatrix) -> Sparse3DMatrix:
    """+ : Merge 3D matrices with addition."""
    new_layers = tuple(
        merge_add(l1, l2) for l1, l2 in zip(M1.layers, M2.layers)
    )
    return Sparse3DMatrix(layers=new_layers, shape=M1.shape)


# ═══════════════════════════════════════════════════════════════════════════
# PATH OPERATIONS
# ═══════════════════════════════════════════════════════════════════════════

def reachable_from(M: SparseMatrix, sources: Set[int], hops: int = 1) -> Set[int]:
    """Reach_k: Find nodes reachable in k hops from sources."""
    current = sources
    for _ in range(hops):
        next_set = set()
        for row, col, _ in M.entries:
            if row in current:
                next_set.add(col)
        current = next_set
    return current


def path_closure(M: SparseMatrix, max_hops: int = 10) -> SparseMatrix:
    """M*: Transitive closure (all paths up to max_hops)."""
    result = M
    current = M
    for _ in range(max_hops - 1):
        current = compose_chain(current, M)
        if current.nnz == 0:
            break
        result = merge_add(result, current)
    return result


# ═══════════════════════════════════════════════════════════════════════════
# LIFT/LOWER OPERATIONS
# ═══════════════════════════════════════════════════════════════════════════

def lift_to_layer(M: SparseMatrix, target_layer: int, n_layers: int) -> Sparse3DMatrix:
    """↑_n: Lift 2D matrix to layer n of 3D matrix."""
    layers = []
    for n in range(n_layers):
        if n == target_layer:
            layers.append(M)
        else:
            layers.append(SparseMatrix.empty(M.shape[0]))
    return Sparse3DMatrix(layers=tuple(layers), shape=(n_layers, M.shape[0], M.shape[1]))


def lower_aggregate(M: Sparse3DMatrix, agg: str = 'sum') -> SparseMatrix:
    """↓: Lower 3D to 2D by aggregating layers."""
    if agg == 'sum':
        result = M.layers[0]
        for layer in M.layers[1:]:
            result = merge_add(result, layer)
        return result
    elif agg == 'max':
        result = M.layers[0]
        for layer in M.layers[1:]:
            result = merge_max(result, layer)
        return result
    else:
        raise ValueError(f"Unknown aggregation: {agg}")


# ═══════════════════════════════════════════════════════════════════════════
# CROSS-STRUCTURE OPERATIONS (AM ↔ W)
# ═══════════════════════════════════════════════════════════════════════════

def am_to_w(AM: Sparse3DMatrix) -> Sparse3DMatrix:
    """AM → W: Convert adjacency counts to transition probabilities."""
    return normalize_3d(AM)


def w_to_am(W: Sparse3DMatrix, scale_factor: float = 1.0) -> Sparse3DMatrix:
    """W → AM: Convert probabilities back to counts (approximate)."""
    new_layers = tuple(scale(layer, scale_factor) for layer in W.layers)
    return Sparse3DMatrix(layers=new_layers, shape=W.shape)


# ═══════════════════════════════════════════════════════════════════════════
# LOOKUP TABLE (LUT) - Token Recovery
# ═══════════════════════════════════════════════════════════════════════════

# Special boundary tokens
START = ("<START>",)
END = ("<END>",)


@dataclass
class LookupTable:
    """
    Lookup Table for n-token recovery.
    
    Uses (reg, zeros) addressing throughout.
    """
    config: Sparse3DConfig
    index_to_ntokens: Dict[int, Set[Tuple[int, Tuple[str, ...]]]] = field(default_factory=lambda: defaultdict(set))
    ntoken_to_index: Dict[Tuple[str, ...], int] = field(default_factory=dict)
    
    def add_ntoken(self, ntoken: Tuple[str, ...]) -> int:
        """Add n-token, return its index."""
        if ntoken in (START, END):
            layer = 0
        else:
            layer = len(ntoken) - 1
        
        content = " ".join(ntoken)
        idx = content_to_index(content, layer, self.config)
        
        self.index_to_ntokens[idx].add((layer, ntoken))
        self.ntoken_to_index[ntoken] = idx
        
        return idx
    
    def get_ntoken_index(self, ntoken: Tuple[str, ...]) -> Optional[int]:
        """Get index for n-token."""
        return self.ntoken_to_index.get(ntoken)
    
    def get_ntokens_at_index(self, idx: int) -> Set[Tuple[int, Tuple[str, ...]]]:
        """Get all (layer, ntoken) pairs at index."""
        return self.index_to_ntokens.get(idx, set())
    
    def get_1tokens_at_index(self, idx: int) -> Set[str]:
        """Get only 1-tokens (single words) at index."""
        result = set()
        for layer, nt in self.index_to_ntokens.get(idx, set()):
            if layer == 0 and nt not in (START, END):
                result.add(nt[0])
        return result


# ═══════════════════════════════════════════════════════════════════════════
# UNIFIED PROCESSING PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class ProcessingResult:
    """
    Result of unified processing.
    
    All intermediate states preserved for inspection/debugging.
    """
    input_hllset: HLLSet
    input_basics: Tuple[BasicHLLSet3D, ...]
    input_edges: Tuple[Edge3D, ...]  # Edges from input tokens
    sub_hrt: SparseHRT3D
    context_edges: Tuple[Edge3D, ...]  # Edges from W context extension
    merged_hrt: SparseHRT3D


def tokenize(text: str) -> List[str]:
    """Simple tokenization."""
    return text.lower().strip().split()


def generate_ntokens(tokens: List[str], max_n: int = 3) -> List[Tuple[str, ...]]:
    """
    Generate n-tokens with START/END boundaries.
    
    Pattern: (START) → (a) → (a,b) → (a,b,c) → (b) → ... → (END)
    """
    ntokens = [START]
    for i in range(len(tokens)):
        for n in range(1, min(max_n + 1, len(tokens) - i + 1)):
            ntokens.append(tuple(tokens[i:i + n]))
    ntokens.append(END)
    return ntokens


def input_to_hllset(
    input_data: str,
    config: Sparse3DConfig,
    lut: LookupTable,
    max_n: int = 3
) -> Tuple[HLLSet, List[BasicHLLSet3D], List[Edge3D]]:
    """
    STEP 1: Convert input to HLLSet.
    
    Same for ingestion AND query.
    """
    tokens = tokenize(input_data)
    ntokens = generate_ntokens(tokens, max_n)
    
    hll = HLLSet(p_bits=config.p_bits)
    basics: List[BasicHLLSet3D] = []
    
    for ntoken in ntokens:
        lut.add_ntoken(ntoken)
        ntoken_text = " ".join(ntoken)
        hll = HLLSet.add(hll, ntoken_text)
        
        # Use UniversalID to compute consistent indices with LUT
        layer = 0 if ntoken in (START, END) else len(ntoken) - 1
        uid = UniversalID.from_content(ntoken_text, layer, config.p_bits, config.h_bits)
        basic = BasicHLLSet3D(n=layer, reg=uid.reg, zeros=uid.zeros)
        basics.append(basic)
    
    # Generate edges
    edges = []
    for i in range(len(ntokens) - 1):
        row_idx = lut.get_ntoken_index(ntokens[i])
        col_idx = lut.get_ntoken_index(ntokens[i + 1])
        
        col_ntoken = ntokens[i + 1]
        layer = 0 if col_ntoken in (START, END) else len(col_ntoken) - 1
        
        if row_idx is not None and col_idx is not None and layer < config.max_n:
            edges.append(Edge3D(n=layer, row=row_idx, col=col_idx, value=1.0))
    
    return hll, basics, edges


def build_sub_hrt(edges: List[Edge3D], config: Sparse3DConfig) -> SparseHRT3D:
    """
    STEP 2: Build isolated HRT from edges.
    """
    if not edges:
        am = SparseAM3D.from_edges(config, [])
        lattice = SparseLattice3D.from_sparse_am(am)
        return SparseHRT3D(am=am, lattice=lattice, config=config, lut=frozenset(), step=0)
    
    edge_dict: Dict[Tuple[int, int, int], float] = {}
    for edge in edges:
        key = (edge.n, edge.row, edge.col)
        edge_dict[key] = edge_dict.get(key, 0.0) + edge.value
    
    aggregated = [Edge3D(n=k[0], row=k[1], col=k[2], value=v) for k, v in edge_dict.items()]
    am = SparseAM3D.from_edges(config, aggregated)
    lattice = SparseLattice3D.from_sparse_am(am)
    
    return SparseHRT3D(am=am, lattice=lattice, config=config, lut=frozenset(), step=0)


def extend_with_context(
    sub_hrt: SparseHRT3D,
    current_W: Dict[int, Dict[int, Dict[int, float]]],
    input_basics: List[BasicHLLSet3D],
    config: Sparse3DConfig
) -> Tuple[SparseHRT3D, List[Edge3D]]:
    """
    STEP 3: Extend sub-HRT with context from current W.
    """
    input_indices = {b.to_index(config) for b in input_basics}
    context_edges: List[Edge3D] = []
    
    for n in range(config.max_n):
        if n not in current_W:
            continue
        for row_idx in input_indices:
            if row_idx in current_W[n]:
                for col_idx, prob in current_W[n][row_idx].items():
                    context_edges.append(Edge3D(n=n, row=row_idx, col=col_idx, value=prob))
    
    if not context_edges:
        return sub_hrt, []
    
    new_am = sub_hrt.am
    for edge in context_edges:
        new_am = new_am.with_edge(edge.n, edge.row, edge.col, edge.value)
    
    new_lattice = SparseLattice3D.from_sparse_am(new_am)
    return SparseHRT3D(
        am=new_am, lattice=new_lattice, config=config,
        lut=sub_hrt.lut, step=sub_hrt.step
    ), context_edges


def extend_with_intersected_context(
    sub_hrt: SparseHRT3D,
    current_W: Dict[int, Dict[int, Dict[int, float]]],
    input_basics: List[BasicHLLSet3D],
    config: Sparse3DConfig
) -> Tuple[SparseHRT3D, List[Edge3D]]:
    """
    STEP 3 (IMPROVED): Extend sub-HRT with INTERSECTED context from current W.
    
    Extended context = row_union(query) ∩ col_union(query)
    
    Where:
        row_union(query) = {col | ∃n: W[n][query_idx][col] > 0}
        col_union(query) = {row | ∃n: W[n][row][query_idx] > 0}
    
    The intersection narrows context to indices that appear in BOTH
    row and column relationships with the query, reducing noise from
    indices that only appear in one direction.
    """
    input_indices = {b.to_index(config) for b in input_basics}
    
    # Collect row-related indices: where query appears as row
    row_related: Set[int] = set()
    # Collect col-related indices: where query appears as col
    col_related: Set[int] = set()
    
    for n in range(config.max_n):
        if n not in current_W:
            continue
        
        for row_idx in input_indices:
            # Query as row → collect all columns
            if row_idx in current_W[n]:
                row_related.update(current_W[n][row_idx].keys())
        
        # Query as column → collect all rows
        for row, cols in current_W[n].items():
            for col in cols.keys():
                if col in input_indices:
                    col_related.add(row)
    
    # Intersected context: indices that appear in BOTH directions
    intersected_context = row_related & col_related
    
    if not intersected_context:
        return sub_hrt, []
    
    # Build context edges only for intersected indices
    context_edges: List[Edge3D] = []
    
    for n in range(config.max_n):
        if n not in current_W:
            continue
        for row_idx in input_indices:
            if row_idx in current_W[n]:
                for col_idx, prob in current_W[n][row_idx].items():
                    if col_idx in intersected_context:
                        context_edges.append(Edge3D(n=n, row=row_idx, col=col_idx, value=prob))
    
    if not context_edges:
        return sub_hrt, []
    
    new_am = sub_hrt.am
    for edge in context_edges:
        new_am = new_am.with_edge(edge.n, edge.row, edge.col, edge.value)
    
    new_lattice = SparseLattice3D.from_sparse_am(new_am)
    return SparseHRT3D(
        am=new_am, lattice=new_lattice, config=config,
        lut=sub_hrt.lut, step=sub_hrt.step
    ), context_edges


def merge_hrt(
    current_hrt: SparseHRT3D,
    sub_hrt: SparseHRT3D,
    config: Sparse3DConfig
) -> SparseHRT3D:
    """
    STEP 4: Merge sub-HRT into current (idempotent).
    
    Properties: commutative, associative.
    """
    all_edges: Dict[Tuple[int, int, int], float] = {}
    
    for n in range(config.max_n):
        for row, col, val in current_hrt.am.tensor.layer_edges(n):
            key = (n, row, col)
            all_edges[key] = all_edges.get(key, 0.0) + val
        for row, col, val in sub_hrt.am.tensor.layer_edges(n):
            key = (n, row, col)
            all_edges[key] = all_edges.get(key, 0.0) + val
    
    merged_edges = [Edge3D(n=k[0], row=k[1], col=k[2], value=v) for k, v in all_edges.items()]
    am = SparseAM3D.from_edges(config, merged_edges)
    lattice = SparseLattice3D.from_sparse_am(am)
    
    return SparseHRT3D(
        am=am, lattice=lattice, config=config,
        lut=frozenset(), step=max(current_hrt.step, sub_hrt.step) + 1
    )


def unified_process(
    input_data: str,
    current_hrt: SparseHRT3D,
    current_W: Dict[int, Dict[int, Dict[int, float]]],
    config: Sparse3DConfig,
    lut: LookupTable,
    max_n: int = 3
) -> ProcessingResult:
    """
    UNIFIED PROCESSING PIPELINE
    
    Same for ingestion AND query:
    INPUT → HLLSet → Sub-HRT → Extend → Merge
    """
    input_hll, input_basics, input_edges = input_to_hllset(input_data, config, lut, max_n)
    sub_hrt = build_sub_hrt(input_edges, config)
    extended_hrt, context_edges = extend_with_context(sub_hrt, current_W, input_basics, config)
    merged_hrt = merge_hrt(current_hrt, extended_hrt, config)
    
    return ProcessingResult(
        input_hllset=input_hll,
        input_basics=tuple(input_basics),
        input_edges=tuple(input_edges),
        sub_hrt=extended_hrt,
        context_edges=tuple(context_edges),
        merged_hrt=merged_hrt
    )


# ═══════════════════════════════════════════════════════════════════════════
# W MATRIX BUILDER
# ═══════════════════════════════════════════════════════════════════════════

def build_w_from_am(am: SparseAM3D, config: Sparse3DConfig) -> Dict[int, Dict[int, Dict[int, float]]]:
    """
    Build W (transition probabilities) from AM.
    
    W[n][row][col] = AM[n, row, col] / Σ_c AM[n, row, c]
    """
    W: Dict[int, Dict[int, Dict[int, float]]] = {}
    
    for n in range(config.max_n):
        W[n] = {}
        edges = am.tensor.layer_edges(n)
        
        row_sums: Dict[int, float] = {}
        row_edges: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
        
        for row, col, val in edges:
            row_sums[row] = row_sums.get(row, 0.0) + val
            row_edges[row].append((col, val))
        
        for row, edges_list in row_edges.items():
            W[n][row] = {}
            row_sum = row_sums[row]
            for col, val in edges_list:
                W[n][row][col] = val / row_sum if row_sum > 0 else 0.0
    
    return W


# ═══════════════════════════════════════════════════════════════════════════
# LAYER HLLSETS (for Cascading Disambiguation)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class LayerHLLSets:
    """
    Layer-specific HLLSets for cascading disambiguation.
    
    Maintains 3 HLLSets (L0, L1, L2) corresponding to n-gram layers,
    plus START_HLLSet for tokens following START symbol.
    
    These are cheap to maintain (just 3 additional HLLSets) and enable
    O(1) layer classification via intersection.
    """
    L0: HLLSet  # Layer 0 (1-grams)
    L1: HLLSet  # Layer 1 (2-grams)
    L2: HLLSet  # Layer 2 (3-grams)
    START: HLLSet  # START followers
    p_bits: int = 10
    
    @classmethod
    def empty(cls, p_bits: int = 10) -> 'LayerHLLSets':
        """Create empty layer HLLSets."""
        return cls(
            L0=HLLSet(p_bits=p_bits),
            L1=HLLSet(p_bits=p_bits),
            L2=HLLSet(p_bits=p_bits),
            START=HLLSet(p_bits=p_bits),
            p_bits=p_bits
        )
    
    @classmethod
    def from_am(cls, am: SparseAM3D, p_bits: int = 10) -> 'LayerHLLSets':
        """Build from existing SparseAM3D."""
        result = cls.empty(p_bits)
        
        for n in range(min(3, am.config.max_n)):
            rows, cols = am.layer_active(n)
            for idx in rows:
                result.add_to_layer(n, idx)
            for idx in cols:
                result.add_to_layer(n, idx)
        
        return result
    
    def add_to_layer(self, layer: int, idx: int):
        """Add index to appropriate layer HLLSet."""
        # Convert idx to string token for HLLSet API
        token = str(idx)
        if layer == 0:
            self.L0 = HLLSet.add(self.L0, token)
        elif layer == 1:
            self.L1 = HLLSet.add(self.L1, token)
        elif layer == 2:
            self.L2 = HLLSet.add(self.L2, token)
    
    def mark_start(self, idx: int):
        """Mark index as START follower."""
        token = str(idx)
        self.START = HLLSet.add(self.START, token)
    
    def get_layer(self, layer: int) -> HLLSet:
        """Get HLLSet for layer."""
        return [self.L0, self.L1, self.L2][layer]
    
    def merge(self, other: 'LayerHLLSets') -> 'LayerHLLSets':
        """Merge (union) two LayerHLLSets."""
        return LayerHLLSets(
            L0=self.L0.union(other.L0),
            L1=self.L1.union(other.L1),
            L2=self.L2.union(other.L2),
            START=self.START.union(other.START),
            p_bits=self.p_bits
        )
    
    def extract_known(self, query_hll: HLLSet) -> HLLSet:
        """
        Extract known tokens from query via layer intersections.
        
        Known = (Q ∩ L0) ∪ (Q ∩ L1) ∪ (Q ∩ L2)
        
        O(1) operation - no lookup table scan needed.
        
        Note: query_hll must have same p_bits as LayerHLLSets.
        """
        if query_hll.p_bits != self.p_bits:
            raise ValueError(
                f"Query HLLSet p_bits ({query_hll.p_bits}) != LayerHLLSets p_bits ({self.p_bits}). "
                f"Use HLLSet.from_batch(tokens, p_bits={self.p_bits}) to create compatible query."
            )
        known = query_hll.intersect(self.L0)
        known = known.union(query_hll.intersect(self.L1))
        known = known.union(query_hll.intersect(self.L2))
        return known
    
    def extract_unknown(self, query_hll: HLLSet) -> HLLSet:
        """
        Extract unknown tokens from query.
        
        Unknown = Q - Known = Q - ((Q ∩ L0) ∪ (Q ∩ L1) ∪ (Q ∩ L2))
        
        If Unknown.cardinality() > 0, query contains unseen tokens.
        O(1) operation for detecting novel content.
        """
        known = self.extract_known(query_hll)
        return query_hll.diff(known)
    
    def classify_query(self, query_hll: HLLSet) -> Dict[str, float]:
        """
        O(1) layer classification of query HLLSet.
        
        Returns similarity scores for each layer + unknown detection.
        """
        known = self.extract_known(query_hll)
        unknown = query_hll.diff(known)
        
        return {
            "L0_sim": query_hll.similarity(self.L0),
            "L1_sim": query_hll.similarity(self.L1),
            "L2_sim": query_hll.similarity(self.L2),
            "START_sim": query_hll.similarity(self.START),
            "known_ratio": known.cardinality() / max(query_hll.cardinality(), 1),
            "unknown_count": int(unknown.cardinality()),
        }
    
    def summary(self) -> Dict[str, int]:
        """Get cardinality summary."""
        return {
            "L0": int(self.L0.cardinality()),
            "L1": int(self.L1.cardinality()),
            "L2": int(self.L2.cardinality()),
            "START": int(self.START.cardinality()),
        }


@dataclass
class DisambiguationResult:
    """Result of disambiguating one index."""
    index: int
    layer: int
    constituent_indices: Set[int]
    
    def __repr__(self) -> str:
        return f"Disamb(idx={self.index}, L{self.layer}, n={len(self.constituent_indices)})"


def update_layer_hllsets(
    edges: List[Edge3D],
    layer_hllsets: LayerHLLSets,
    start_indices: Optional[Set[int]] = None
) -> LayerHLLSets:
    """
    Update layer HLLSets from new edges.
    
    Call during build_sub_hrt or merge to keep in sync.
    """
    for edge in edges:
        layer_hllsets.add_to_layer(edge.n, edge.row)
        layer_hllsets.add_to_layer(edge.n, edge.col)
    
    if start_indices:
        for idx in start_indices:
            layer_hllsets.mark_start(idx)
    
    return layer_hllsets


def cascading_disambiguate(
    query_indices: Set[int],
    am: SparseAM3D,
    layer_hllsets: LayerHLLSets,
    W: Dict[int, Dict[int, Dict[int, float]]],
    lut: LookupTable
) -> List[DisambiguationResult]:
    """
    Full Cascading Disambiguation Algorithm (5 steps).
    
    Given query indices, reconstruct token sequences by:
    1. Slice by layer (H_0, H_1, H_2 via intersection with LayerHLLSets)
    2. Find START candidates (H_0 ∩ START_HLLSet)
    3. Follow W transitions (W[s] ∩ H_1, W[2-gram] ∩ H_2)
    4. Decompose n-grams to constituent hashes
    5. Remove processed, repeat until H empty
    
    Returns list of DisambiguationResult ordered by discovery.
    """
    results = []
    processed = set()
    remaining = set(query_indices)
    
    # STEP 1: Classify by layer
    # Use AM's layer_active for accurate layer membership
    layer0_active = am.layer_active(0)[0] | am.layer_active(0)[1]
    layer1_active = (am.layer_active(1)[0] | am.layer_active(1)[1]) if am.config.max_n > 1 else set()
    layer2_active = (am.layer_active(2)[0] | am.layer_active(2)[1]) if am.config.max_n > 2 else set()
    
    H_0 = remaining & layer0_active  # 1-grams in query
    H_1 = remaining & layer1_active  # 2-grams in query
    H_2 = remaining & layer2_active  # 3-grams in query
    
    # STEP 2: Find START candidates
    # Look up START index from LUT
    start_idx = lut.get_ntoken_index(START)
    start_followers = set()
    
    if start_idx is not None and 0 in W:
        # Get all indices that follow START
        if start_idx in W[0]:
            start_followers = set(W[0][start_idx].keys())
    
    start_candidates = H_0 & start_followers
    
    # STEP 3: Follow W transitions from START candidates
    for start_token in start_candidates:
        if start_token in processed:
            continue
        
        # Try to build sequence: start_token → 2-gram → 3-gram
        sequence_results = _follow_transitions(
            start_token, W, H_0, H_1, H_2, am, processed
        )
        
        for result in sequence_results:
            results.append(result)
            processed.add(result.index)
            processed.update(result.constituent_indices)
            remaining.discard(result.index)
            remaining -= result.constituent_indices
    
    # STEP 4: Process remaining 3-grams (not reached via START)
    remaining_H2 = H_2 - processed
    for idx in remaining_H2:
        if idx in processed:
            continue
        
        # Decompose 3-gram to constituents
        constituents = _get_constituents(idx, 2, am)
        results.append(DisambiguationResult(idx, 2, constituents))
        processed.add(idx)
        processed.update(constituents)
        remaining.discard(idx)
        remaining -= constituents
    
    # Process remaining 2-grams
    remaining_H1 = H_1 - processed
    for idx in remaining_H1:
        if idx in processed:
            continue
        
        constituents = _get_constituents(idx, 1, am)
        results.append(DisambiguationResult(idx, 1, constituents))
        processed.add(idx)
        processed.update(constituents)
        remaining.discard(idx)
        remaining -= constituents
    
    # STEP 5: Process standalone 1-grams
    remaining_H0 = H_0 - processed
    for idx in remaining_H0:
        if idx in processed:
            continue
        results.append(DisambiguationResult(idx, 0, {idx}))
        processed.add(idx)
        remaining.discard(idx)
    
    return results


def _follow_transitions(
    start_token: int,
    W: Dict[int, Dict[int, Dict[int, float]]],
    H_0: Set[int],
    H_1: Set[int],
    H_2: Set[int],
    am: SparseAM3D,
    processed: Set[int]
) -> List[DisambiguationResult]:
    """
    Follow W transitions from a start token.
    
    start_token → 2-grams → 3-grams
    
    Returns results for the discovered sequence.
    """
    results = []
    
    # Start token is a 1-gram
    if start_token in processed:
        return results
    
    results.append(DisambiguationResult(start_token, 0, {start_token}))
    
    # Look for 2-grams following this token
    if 0 not in W or start_token not in W[0]:
        return results
    
    followers_1 = set(W[0][start_token].keys()) & H_0
    
    for follower in followers_1:
        if follower in processed:
            continue
        
        # Check if (start_token, follower) forms a 2-gram in H_1
        # The 2-gram index would be found via the AM connections
        two_gram_candidates = set()
        if 1 in W:
            for tg_idx in W.get(1, {}).get(start_token, {}).keys():
                if tg_idx in H_1:
                    two_gram_candidates.add(tg_idx)
        
        for tg_idx in two_gram_candidates:
            if tg_idx in processed:
                continue
            
            constituents_2g = _get_constituents(tg_idx, 1, am)
            results.append(DisambiguationResult(tg_idx, 1, constituents_2g))
            
            # Look for 3-grams
            if 2 in W and tg_idx in W[2]:
                for three_gram_idx in W[2][tg_idx].keys():
                    if three_gram_idx in H_2 and three_gram_idx not in processed:
                        constituents_3g = _get_constituents(three_gram_idx, 2, am)
                        results.append(DisambiguationResult(three_gram_idx, 2, constituents_3g))
    
    return results


def _get_constituents(idx: int, layer: int, am: SparseAM3D) -> Set[int]:
    """
    Get constituent indices for an n-gram.
    
    For layer 2 (3-gram): returns 2 constituent indices
    For layer 1 (2-gram): returns 1 constituent index
    For layer 0 (1-gram): returns self
    """
    if layer == 0:
        return {idx}
    
    constituents = set()
    for row, col, _ in am.layer_edges(layer):
        if row == idx:
            constituents.add(col)
        if col == idx:
            constituents.add(row)
    
    # If no constituents found via AM, return self
    return constituents if constituents else {idx}


def resolve_disambiguation(
    results: List[DisambiguationResult],
    lut: LookupTable
) -> Dict[int, List[str]]:
    """
    Resolve DisambiguationResults to tokens using LUT.
    
    Returns {index: [tokens]}
    """
    resolved = {}
    
    for r in results:
        tokens = []
        for idx in r.constituent_indices:
            ntoken = lut.index_to_ntokens.get(idx)
            if ntoken:
                if isinstance(ntoken, tuple):
                    tokens.extend(ntoken)
                else:
                    tokens.append(str(ntoken))
            else:
                tokens.append(f"<{idx}>")
        resolved[r.index] = tokens
    
    return resolved


# ═══════════════════════════════════════════════════════════════════════════
# COMMIT STORE - Track Processing History
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class Commit:
    """
    A timestamped commit representing a processing step.
    
    Each ingestion/query creates a commit, enabling:
    - Time-travel: access any historical state
    - Provenance: trace where data came from
    - Incremental update: resume from any point
    """
    id: str              # SHA-1 of content
    timestamp: float     # UNIX timestamp
    source: str          # File path or query ID
    perceptron: str      # Which perceptron processed it
    hrt: SparseHRT3D     # State after this commit
    W: Dict[int, Dict[int, Dict[int, float]]]  # W matrix after commit
    parent_id: Optional[str] = None  # Previous commit
    
    @classmethod
    def create(
        cls,
        hrt: SparseHRT3D,
        W: Dict[int, Dict[int, Dict[int, float]]],
        source: str,
        perceptron: str,
        parent_id: Optional[str] = None
    ) -> 'Commit':
        """Create a new commit."""
        import time
        timestamp = time.time()
        content = f"{source}:{timestamp}:{hrt.nnz}"
        commit_id = compute_sha1(content)
        
        return cls(
            id=commit_id,
            timestamp=timestamp,
            source=source,
            perceptron=perceptron,
            hrt=hrt,
            W=W,
            parent_id=parent_id
        )


class CommitStore:
    """
    Store for tracking commits (processing history).
    
    Enables:
    - Linear history of all processing
    - Rollback to any point
    - Branching for experiments
    """
    
    def __init__(self):
        self.commits: Dict[str, Commit] = {}
        self.head: Optional[str] = None
        self.history: List[str] = []
    
    def commit(
        self,
        hrt: SparseHRT3D,
        W: Dict[int, Dict[int, Dict[int, float]]],
        source: str,
        perceptron: str
    ) -> Commit:
        """Create and store a new commit."""
        c = Commit.create(hrt, W, source, perceptron, self.head)
        self.commits[c.id] = c
        self.head = c.id
        self.history.append(c.id)
        return c
    
    def get(self, commit_id: str) -> Optional[Commit]:
        """Get commit by ID."""
        return self.commits.get(commit_id)
    
    def rollback(self, commit_id: str) -> Optional[Commit]:
        """Set HEAD to a previous commit."""
        if commit_id in self.commits:
            self.head = commit_id
            return self.commits[commit_id]
        return None
    
    def latest(self) -> Optional[Commit]:
        """Get the latest commit."""
        return self.commits.get(self.head) if self.head else None
    
    def log(self, limit: int = 10) -> List[Commit]:
        """Get recent commits."""
        return [self.commits[cid] for cid in self.history[-limit:] if cid in self.commits]
    
    def __len__(self) -> int:
        return len(self.commits)


# ═══════════════════════════════════════════════════════════════════════════
# PERCEPTRON BASE CLASS - Sense Phase
# ═══════════════════════════════════════════════════════════════════════════

from abc import ABC, abstractmethod
from pathlib import Path


class Perceptron(ABC):
    """
    Base class for perceptrons - sense input and convert to HLLSet.
    
    Each perceptron:
    1. Finds/accepts its input type
    2. Extracts text content
    3. Processes via unified pipeline
    4. Commits after processing
    
    Part of sense-process-act loop:
        Perceptron (sense) → Pipeline (process) → Actuator (act)
    """
    
    def __init__(self, name: str, extensions: List[str], config: Sparse3DConfig):
        self.name = name
        self.extensions = extensions
        self.config = config
        self.lut: Optional[LookupTable] = None
        self.files_processed = 0
        self.total_tokens = 0
    
    def initialize(self, lut: LookupTable):
        """Initialize with shared LUT."""
        self.lut = lut
    
    def find_files(self, root: Path, exclude_dirs: Set[str] = None) -> Iterator[Path]:
        """Find all files matching extensions."""
        exclude_dirs = exclude_dirs or {'__pycache__', '.git', 'build', '.ipynb_checkpoints', 'deprecated'}
        
        for path in root.rglob('*'):
            if path.is_file() and path.suffix in self.extensions:
                if not any(ex in path.parts for ex in exclude_dirs):
                    yield path
    
    @abstractmethod
    def extract_text(self, path: Path) -> str:
        """Extract text content from input."""
        pass
    
    def process_file(
        self,
        path: Path,
        current_hrt: SparseHRT3D,
        current_W: Dict[int, Dict[int, Dict[int, float]]],
        store: CommitStore,
        max_n: int = 3
    ) -> Tuple[SparseHRT3D, Dict[int, Dict[int, Dict[int, float]]], Optional[Commit]]:
        """
        Process a single file and commit.
        
        Returns (new_hrt, new_W, commit)
        """
        if not self.lut:
            raise RuntimeError("Perceptron not initialized - call initialize(lut) first")
        
        text = self.extract_text(path)
        if not text.strip():
            return current_hrt, current_W, None
        
        # Unified processing
        result = unified_process(
            text,
            current_hrt,
            current_W,
            self.config,
            self.lut,
            max_n
        )
        
        # Update state
        new_hrt = result.merged_hrt
        new_W = build_w_from_am(new_hrt.am, self.config)
        
        # Commit
        commit = store.commit(new_hrt, new_W, str(path), self.name)
        
        self.files_processed += 1
        self.total_tokens = len(self.lut.ntoken_to_index)
        
        return new_hrt, new_W, commit
    
    def process_all(
        self,
        root: Path,
        current_hrt: SparseHRT3D,
        current_W: Dict[int, Dict[int, Dict[int, float]]],
        store: CommitStore,
        max_files: Optional[int] = None,
        max_n: int = 3,
        verbose: bool = True
    ) -> Tuple[SparseHRT3D, Dict[int, Dict[int, Dict[int, float]]]]:
        """
        Process all files of this type.
        
        Args:
            root: Root directory to search
            max_files: Limit number of files (None = all)
            max_n: Maximum n-gram size
            verbose: Print progress
        """
        files = list(self.find_files(root))
        if max_files:
            files = files[:max_files]
        
        if verbose:
            print(f"[{self.name}] Processing {len(files)} files")
        
        for path in files:
            try:
                current_hrt, current_W, commit = self.process_file(
                    path, current_hrt, current_W, store, max_n
                )
                if verbose and commit:
                    print(f"  ✓ {path.name} [{current_hrt.nnz} edges]")
            except Exception as e:
                if verbose:
                    print(f"  ✗ {path.name}: {e}")
        
        return current_hrt, current_W


class PromptPerceptron(Perceptron):
    """
    Perceptron for user prompts/queries.
    
    Treats user input exactly like file input:
    - Goes through unified pipeline
    - Gets committed
    - Contributes to manifold (learning from queries!)
    """
    
    def __init__(self, config: Sparse3DConfig):
        super().__init__("p_prompt", [], config)
        self.prompt_history: List[str] = []
    
    def extract_text(self, path: Path) -> str:
        """Not used - prompts come directly as text."""
        return ""
    
    def process_prompt(
        self,
        prompt: str,
        current_hrt: SparseHRT3D,
        current_W: Dict[int, Dict[int, Dict[int, float]]],
        store: CommitStore,
        max_n: int = 3
    ) -> Tuple[SparseHRT3D, Dict[int, Dict[int, Dict[int, float]]], Optional[Commit], Optional[ProcessingResult]]:
        """
        Process a user prompt and commit.
        
        Returns (new_hrt, new_W, commit, processing_result)
        """
        if not self.lut:
            raise RuntimeError("Perceptron not initialized - call initialize(lut) first")
        
        if not prompt.strip():
            return current_hrt, current_W, None, None
        
        self.prompt_history.append(prompt)
        
        # Unified processing (same as files!)
        result = unified_process(
            prompt,
            current_hrt,
            current_W,
            self.config,
            self.lut,
            max_n
        )
        
        new_hrt = result.merged_hrt
        new_W = build_w_from_am(new_hrt.am, self.config)
        
        # Commit with prompt ID as source
        prompt_id = f"prompt_{len(self.prompt_history)}"
        commit = store.commit(new_hrt, new_W, prompt_id, self.name)
        
        self.files_processed += 1
        
        return new_hrt, new_W, commit, result


# ═══════════════════════════════════════════════════════════════════════════
# ACTUATOR BASE CLASS - Act Phase
# ═══════════════════════════════════════════════════════════════════════════

class Actuator(ABC):
    """
    Base class for actuators - turn processed data into action.
    
    Completes the sense-process-act loop:
        Perceptron (sense) → Pipeline (process) → Actuator (act)
    
    Key insight: Actuator output can feed back into the manifold!
    """
    
    def __init__(self, name: str):
        self.name = name
        self.actions_taken = 0
    
    @abstractmethod
    def act(self, commit: Commit, result: ProcessingResult, **kwargs) -> str:
        """
        Perform action based on processed result.
        
        Returns action summary string.
        """
        pass


class ResponseActuator(Actuator):
    """
    Actuator for query responses with FEEDBACK LOOP.
    
    The response itself is ingested back into the manifold!
    This creates co-adaptive learning:
        Query → Response → HLLSet → Commit → (shapes future responses)
    """
    
    def __init__(self):
        super().__init__("a_response")
        self.responses: List[Dict[str, Any]] = []
    
    def act(
        self,
        commit: Commit,
        result: ProcessingResult,
        query_results: List[Tuple[Any, float]] = None,
        hrt: SparseHRT3D = None,
        W: Dict[int, Dict[int, Dict[int, float]]] = None,
        store: CommitStore = None,
        lut: LookupTable = None,
        config: Sparse3DConfig = None,
        ingest_response: bool = True,
        max_n: int = 3,
        **kwargs
    ) -> Tuple[str, SparseHRT3D, Dict[int, Dict[int, Dict[int, float]]]]:
        """
        Generate response and optionally ingest it back.
        
        Returns:
            (response_text, updated_hrt, updated_W)
        """
        import time
        from datetime import datetime
        
        # Build response text
        lines = [
            f"Query: {commit.source}",
            f"Commit: {commit.id[:8]}",
            f"Results ({len(query_results or [])} found):",
        ]
        
        for i, (ntoken, score) in enumerate(query_results or [], 1):
            lines.append(f"  {i:2d}. [{score:5.1f}] {ntoken}")
        
        response_text = "\n".join(lines)
        
        # Track response
        response_record = {
            "timestamp": datetime.fromtimestamp(commit.timestamp).isoformat(),
            "prompt": commit.source,
            "commit_id": commit.id[:8],
            "response": response_text,
            "ingested": False,
        }
        
        new_hrt = hrt
        new_W = W
        
        # FEEDBACK LOOP: Ingest response back into manifold
        if ingest_response and hrt and store and lut and config:
            response_result = unified_process(
                response_text,
                hrt,
                W,
                config,
                lut,
                max_n
            )
            
            new_hrt = response_result.merged_hrt
            new_W = build_w_from_am(new_hrt.am, config)
            
            # Commit response as its own entry
            response_id = f"response_{len(self.responses) + 1}"
            store.commit(new_hrt, new_W, response_id, self.name)
            
            response_record["ingested"] = True
            response_record["response_commit"] = response_id
        
        self.responses.append(response_record)
        self.actions_taken += 1
        
        return response_text, new_hrt, new_W
    
    def history(self) -> List[Dict[str, Any]]:
        """Get response history."""
        return self.responses


# ═══════════════════════════════════════════════════════════════════════════
# QUERY INTERFACE - ask() function
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class QueryContext:
    """
    Mutable context for interactive querying.
    
    Holds the current state that gets updated with each query.
    """
    hrt: SparseHRT3D
    W: Dict[int, Dict[int, Dict[int, float]]]
    config: Sparse3DConfig
    lut: LookupTable
    store: CommitStore
    layer_hllsets: LayerHLLSets
    prompt_perceptron: PromptPerceptron
    response_actuator: ResponseActuator
    max_n: int = 3


def ask(
    prompt: str,
    ctx: QueryContext,
    top_k: int = 10,
    learn: bool = True
) -> Tuple[str, List[DisambiguationResult]]:
    """
    Interactive query with feedback loop.
    
    Full sense-process-act-feedback cycle:
    1. Query → HLLSet → HRT → Commit (SENSE)
    2. Find related concepts (PROCESS)
    3. Disambiguate to tokens (PROCESS)
    4. Generate response (ACT)
    5. Response → HLLSet → HRT → Commit (FEEDBACK!)
    
    The manifold learns from BOTH the question AND its own answer.
    
    Args:
        prompt: User query text
        ctx: QueryContext with current state
        top_k: Number of results to return
        learn: If True, ingest both query AND response
    
    Returns:
        (response_text, disambiguation_results)
    """
    # SENSE: Process query through prompt perceptron
    new_hrt, new_W, commit, result = ctx.prompt_perceptron.process_prompt(
        prompt,
        ctx.hrt,
        ctx.W,
        ctx.store,
        ctx.max_n
    )
    
    if not commit:
        return "No results (empty query)", []
    
    if learn:
        ctx.hrt = new_hrt
        ctx.W = new_W
    
    # PROCESS: Get query indices
    query_indices = set()
    if result:
        for basic in result.input_basics:
            query_indices.add(basic.to_index(ctx.config))
    
    # Find reachable concepts
    AM = Sparse3DMatrix.from_am(ctx.hrt.am, ctx.config)
    layer0 = project_layer(AM, 0)
    reachable = reachable_from(layer0, query_indices, hops=1)
    
    # Score by connectivity
    layer0_dict = layer0.to_dict()
    scores = {}
    for idx in reachable:
        if idx in layer0_dict:
            scores[idx] = sum(layer0_dict[idx].values())
    
    # Get top-k results
    top = sorted(scores.items(), key=lambda x: -x[1])[:top_k]
    query_results = []
    for idx, score in top:
        ntokens = ctx.lut.index_to_ntokens.get(idx, set())
        if ntokens:
            # Get first ntoken
            _, ntoken = next(iter(ntokens))
            query_results.append((ntoken, score))
        else:
            query_results.append((f"<idx:{idx}>", score))
    
    # DISAMBIGUATE: Full cascading disambiguation
    disamb_results = cascading_disambiguate(
        query_indices=query_indices,
        am=ctx.hrt.am,
        layer_hllsets=ctx.layer_hllsets,
        W=ctx.W,
        lut=ctx.lut
    )
    
    # ACT + FEEDBACK: Generate response and ingest it back
    response_text, final_hrt, final_W = ctx.response_actuator.act(
        commit,
        result,
        query_results=query_results,
        hrt=ctx.hrt,
        W=ctx.W,
        store=ctx.store,
        lut=ctx.lut,
        config=ctx.config,
        ingest_response=learn,
        max_n=ctx.max_n
    )
    
    if learn:
        ctx.hrt = final_hrt
        ctx.W = final_W
    
    return response_text, disamb_results


def create_query_context(
    config: Sparse3DConfig,
    lut: Optional[LookupTable] = None
) -> QueryContext:
    """
    Create a new QueryContext for interactive querying.
    
    Initializes all components needed for ask().
    """
    if lut is None:
        lut = LookupTable(config=config)
        lut.add_ntoken(START)
        lut.add_ntoken(END)
    
    # Empty initial structures
    empty_am = SparseAM3D.from_edges(config, [])
    empty_lattice = SparseLattice3D.from_sparse_am(empty_am)
    empty_hrt = SparseHRT3D(
        am=empty_am,
        lattice=empty_lattice,
        config=config,
        lut=frozenset(),
        step=0
    )
    
    empty_W: Dict[int, Dict[int, Dict[int, float]]] = {n: {} for n in range(config.max_n)}
    
    # Components
    store = CommitStore()
    layer_hllsets = LayerHLLSets.empty(config.p_bits)
    
    prompt_perceptron = PromptPerceptron(config)
    prompt_perceptron.initialize(lut)
    
    response_actuator = ResponseActuator()
    
    return QueryContext(
        hrt=empty_hrt,
        W=empty_W,
        config=config,
        lut=lut,
        store=store,
        layer_hllsets=layer_hllsets,
        prompt_perceptron=prompt_perceptron,
        response_actuator=response_actuator,
        max_n=config.max_n
    )


# ═══════════════════════════════════════════════════════════════════════════
# MODULE EXPORTS
# ═══════════════════════════════════════════════════════════════════════════

__all__ = [
    # Universal ID
    'UniversalID',
    'content_to_index',
    
    # Sparse Matrices
    'SparseMatrix',
    'Sparse3DMatrix',
    
    # Projection
    'project_layer',
    'project_rows',
    'project_cols',
    'project_submatrix',
    
    # Transform
    'transpose',
    'transpose_3d',
    'normalize_rows',
    'normalize_3d',
    'scale',
    
    # Filter
    'filter_threshold',
    'filter_predicate',
    
    # Composition
    'merge_add',
    'merge_max',
    'compose_chain',
    'merge_3d_add',
    
    # Path
    'reachable_from',
    'path_closure',
    
    # Lift/Lower
    'lift_to_layer',
    'lower_aggregate',
    
    # Cross-structure
    'am_to_w',
    'w_to_am',
    
    # LUT
    'START',
    'END',
    'LookupTable',
    
    # Unified Processing
    'ProcessingResult',
    'tokenize',
    'generate_ntokens',
    'input_to_hllset',
    'build_sub_hrt',
    'extend_with_context',
    'extend_with_intersected_context',
    'merge_hrt',
    'unified_process',
    'build_w_from_am',
    
    # Cascading Disambiguation
    'LayerHLLSets',
    'DisambiguationResult',
    'update_layer_hllsets',
    'cascading_disambiguate',
    'resolve_disambiguation',
    
    # Commit Store
    'Commit',
    'CommitStore',
    
    # Perceptrons
    'Perceptron',
    'PromptPerceptron',
    
    # Actuators
    'Actuator',
    'ResponseActuator',
    
    # Query Interface
    'QueryContext',
    'ask',
    'create_query_context',
]
