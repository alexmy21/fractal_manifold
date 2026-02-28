"""
Manifold Algebra Operations

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
from typing import Dict, Set, Callable

from .sparse_matrices import SparseMatrix, Sparse3DMatrix


# ═══════════════════════════════════════════════════════════════════════════
# PROJECTION OPERATIONS
# ═══════════════════════════════════════════════════════════════════════════

def project_layer(M: Sparse3DMatrix, layer: int) -> SparseMatrix:
    """π_n: Extract layer n from 3D matrix."""
    if layer < 0 or layer >= M.n_layers:
        return SparseMatrix.empty(M.shape[1:])
    return M.layers[layer]


def project_rows(M: SparseMatrix, rows: Set[int]) -> SparseMatrix:
    """π_R: Extract only specified rows."""
    new_data = {r: M.data[r].copy() for r in rows if r in M.data}
    return SparseMatrix(data=new_data, shape=M.shape)


def project_cols(M: SparseMatrix, cols: Set[int]) -> SparseMatrix:
    """π_C: Extract only specified columns."""
    new_data: Dict[int, Dict[int, float]] = {}
    for row, row_data in M.data.items():
        new_row = {c: v for c, v in row_data.items() if c in cols}
        if new_row:
            new_data[row] = new_row
    return SparseMatrix(data=new_data, shape=M.shape)


def project_submatrix(M: SparseMatrix, rows: Set[int], cols: Set[int]) -> SparseMatrix:
    """π_{R,C}: Extract submatrix with given rows and columns."""
    return project_cols(project_rows(M, rows), cols)


# ═══════════════════════════════════════════════════════════════════════════
# TRANSFORM OPERATIONS
# ═══════════════════════════════════════════════════════════════════════════

def transpose(M: SparseMatrix) -> SparseMatrix:
    """T(M): Transpose matrix."""
    new_data: Dict[int, Dict[int, float]] = {}
    for row, cols in M.data.items():
        for col, val in cols.items():
            if col not in new_data:
                new_data[col] = {}
            new_data[col][row] = val
    return SparseMatrix(data=new_data, shape=(M.shape[1], M.shape[0]))


def transpose_3d(M: Sparse3DMatrix) -> Sparse3DMatrix:
    """T(M) for 3D: Transpose each layer."""
    new_layers = tuple(transpose(layer) for layer in M.layers)
    return Sparse3DMatrix(layers=new_layers, shape=(M.shape[0], M.shape[2], M.shape[1]))


def normalize_rows(M: SparseMatrix) -> SparseMatrix:
    """N(M): Normalize rows to sum to 1 (stochastic matrix)."""
    new_data: Dict[int, Dict[int, float]] = {}
    for row, cols in M.data.items():
        row_sum = sum(cols.values())
        if row_sum > 0:
            new_data[row] = {c: v / row_sum for c, v in cols.items()}
        else:
            new_data[row] = cols.copy()
    return SparseMatrix(data=new_data, shape=M.shape)


def normalize_3d(M: Sparse3DMatrix) -> Sparse3DMatrix:
    """N(M) for 3D: Normalize each layer."""
    new_layers = tuple(normalize_rows(layer) for layer in M.layers)
    return Sparse3DMatrix(layers=new_layers, shape=M.shape)


def scale(M: SparseMatrix, factor: float) -> SparseMatrix:
    """S_α(M): Scale all values by factor."""
    new_data: Dict[int, Dict[int, float]] = {}
    for row, cols in M.data.items():
        new_data[row] = {c: v * factor for c, v in cols.items()}
    return SparseMatrix(data=new_data, shape=M.shape)


# ═══════════════════════════════════════════════════════════════════════════
# FILTER OPERATIONS
# ═══════════════════════════════════════════════════════════════════════════

def filter_threshold(M: SparseMatrix, min_val: float = 0.0) -> SparseMatrix:
    """Filter out values below threshold."""
    new_data: Dict[int, Dict[int, float]] = {}
    for row, cols in M.data.items():
        filtered = {c: v for c, v in cols.items() if v > min_val}
        if filtered:
            new_data[row] = filtered
    return SparseMatrix(data=new_data, shape=M.shape)


def filter_predicate(M: SparseMatrix, pred: Callable[[int, int, float], bool]) -> SparseMatrix:
    """Filter by arbitrary predicate on (row, col, value)."""
    new_data: Dict[int, Dict[int, float]] = {}
    for row, cols in M.data.items():
        filtered = {c: v for c, v in cols.items() if pred(row, c, v)}
        if filtered:
            new_data[row] = filtered
    return SparseMatrix(data=new_data, shape=M.shape)


# ═══════════════════════════════════════════════════════════════════════════
# COMPOSITION OPERATIONS
# ═══════════════════════════════════════════════════════════════════════════

def merge_add(M1: SparseMatrix, M2: SparseMatrix) -> SparseMatrix:
    """
    M₁ + M₂: Add corresponding values.
    
    Commutative, associative, idempotent for CRDT-like merge.
    """
    new_data: Dict[int, Dict[int, float]] = {}
    
    # Copy M1
    for row, cols in M1.data.items():
        new_data[row] = cols.copy()
    
    # Add M2
    for row, cols in M2.data.items():
        if row not in new_data:
            new_data[row] = {}
        for col, val in cols.items():
            new_data[row][col] = new_data[row].get(col, 0.0) + val
    
    return SparseMatrix(data=new_data, shape=M1.shape)


def merge_max(M1: SparseMatrix, M2: SparseMatrix) -> SparseMatrix:
    """
    max(M₁, M₂): Take maximum of corresponding values.
    
    Alternative merge semantics.
    """
    new_data: Dict[int, Dict[int, float]] = {}
    
    all_rows = set(M1.data.keys()) | set(M2.data.keys())
    for row in all_rows:
        cols1 = M1.data.get(row, {})
        cols2 = M2.data.get(row, {})
        all_cols = set(cols1.keys()) | set(cols2.keys())
        
        new_data[row] = {}
        for col in all_cols:
            v1 = cols1.get(col, 0.0)
            v2 = cols2.get(col, 0.0)
            new_data[row][col] = max(v1, v2)
    
    return SparseMatrix(data=new_data, shape=M1.shape)


def compose_chain(M1: SparseMatrix, M2: SparseMatrix) -> SparseMatrix:
    """
    M₁ ∘ M₂: Matrix multiplication (chain composition).
    
    (M₁ ∘ M₂)[i,j] = Σ_k M₁[i,k] × M₂[k,j]
    """
    new_data: Dict[int, Dict[int, float]] = {}
    
    # For each row in M1
    for i, row1_data in M1.data.items():
        new_data[i] = {}
        # For each column k that row i touches
        for k, v1 in row1_data.items():
            # k must be a row in M2
            if k in M2.data:
                for j, v2 in M2.data[k].items():
                    new_data[i][j] = new_data[i].get(j, 0.0) + v1 * v2
    
    return SparseMatrix(data=new_data, shape=(M1.shape[0], M2.shape[1]))


def merge_3d_add(M1: Sparse3DMatrix, M2: Sparse3DMatrix) -> Sparse3DMatrix:
    """Add corresponding layers of two 3D matrices."""
    new_layers = tuple(
        merge_add(l1, l2) for l1, l2 in zip(M1.layers, M2.layers)
    )
    return Sparse3DMatrix(layers=new_layers, shape=M1.shape)


# ═══════════════════════════════════════════════════════════════════════════
# PATH OPERATIONS
# ═══════════════════════════════════════════════════════════════════════════

def reachable_from(M: SparseMatrix, sources: Set[int], hops: int = 1) -> Set[int]:
    """
    reach(M, S, k): Find all nodes reachable in k hops from sources.
    """
    current = sources.copy()
    for _ in range(hops):
        next_nodes: Set[int] = set()
        for node in current:
            if node in M.data:
                next_nodes.update(M.data[node].keys())
        current = next_nodes
    return current


def path_closure(M: SparseMatrix, max_hops: int = 10) -> SparseMatrix:
    """
    M*: Transitive closure (all reachable pairs).
    
    Computed iteratively up to max_hops.
    """
    result = M
    power = M
    for _ in range(max_hops - 1):
        power = compose_chain(power, M)
        result = merge_max(result, power)
    return result


# ═══════════════════════════════════════════════════════════════════════════
# LIFT / LOWER OPERATIONS
# ═══════════════════════════════════════════════════════════════════════════

def lift_to_layer(M: SparseMatrix, target_layer: int, n_layers: int) -> Sparse3DMatrix:
    """
    ↑_n(M): Lift 2D matrix to specific layer of 3D matrix.
    
    Creates a 3D matrix where only layer n contains M.
    """
    layers = [SparseMatrix.empty(M.shape) for _ in range(n_layers)]
    if 0 <= target_layer < n_layers:
        layers[target_layer] = M
    return Sparse3DMatrix(layers=tuple(layers), shape=(n_layers, M.shape[0], M.shape[1]))


def lower_aggregate(M: Sparse3DMatrix, agg: str = 'sum') -> SparseMatrix:
    """
    ↓(M): Lower 3D matrix to 2D by aggregating across layers.
    
    Aggregation modes:
    - 'sum': Add values across layers
    - 'max': Take maximum across layers
    """
    if M.n_layers == 0:
        return SparseMatrix.empty((0, 0))
    
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


__all__ = [
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
]
