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
    sub_hrt: SparseHRT3D
    context_edges: Tuple[Edge3D, ...]
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
        
        h = int(hashlib.sha1(ntoken_text.encode()).hexdigest()[:8], 16)
        layer = 0 if ntoken in (START, END) else len(ntoken) - 1
        basic = BasicHLLSet3D.from_hash(h, n=layer, p_bits=config.p_bits, h_bits=config.h_bits)
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
    'merge_hrt',
    'unified_process',
    'build_w_from_am',
]
