"""
Unified Processing Pipeline

Every interaction (ingestion OR query) follows the same pipeline:

    INPUT → HLLSet → New HRT → Extend with Context → Merge → New Current

Properties:
- Sub-structure isolation: work on separate instance
- Idempotent merge: same input → same result
- Eventual consistency: parallel changes converge
- CRDT-like: commutative, associative, idempotent

Architecture Note:
    Uses Kernel for HLLSet operations to ensure consistent p_bits and SHARED_SEED.
    HLLSet is imported for type annotations only.
"""

from __future__ import annotations
from typing import List, Dict, Tuple, Set
from dataclasses import dataclass
from collections import defaultdict

# Type annotation (no runtime behavior)
from ..hllset import HLLSet
# Operations go through kernel for consistent configuration
from ..kernel import Kernel
from ..sparse_hrt_3d import (
    SparseHRT3D,
    Sparse3DConfig,
    SparseAM3D,
    SparseLattice3D,
    BasicHLLSet3D,
    Edge3D,
)

from .universal_id import UniversalID
from .lut import LookupTable, START, END

# Module-level kernel instance for operations
_kernel = Kernel()


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
    STEP 1: Convert input to HLLSet via kernel.
    
    Same for ingestion AND query.
    """
    tokens = tokenize(input_data)
    ntokens = generate_ntokens(tokens, max_n)
    
    # Use kernel for HLLSet operations
    kernel = Kernel(p_bits=config.p_bits)
    hll = kernel.absorb(set())  # Start with empty HLLSet
    basics: List[BasicHLLSet3D] = []
    
    for ntoken in ntokens:
        lut.add_ntoken(ntoken)
        ntoken_text = " ".join(ntoken)
        hll = kernel.add(hll, ntoken_text)
        
        # Use UniversalID to compute consistent indices with LUT
        layer = 0 if ntoken in (START, END) else len(ntoken) - 1
        uid = UniversalID.from_content(ntoken_text, layer)
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


__all__ = [
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
]
