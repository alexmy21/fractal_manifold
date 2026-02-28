"""
GraphBLAS Operations for Lattice Processing

Provides CPU-optimized graph algorithms using SuiteSparse:GraphBLAS.
Complements the GPU-based HRT operations in sparse_hrt_3d.py.

Architecture:
============
    ┌─────────────────────────────────────────────────────────────────┐
    │  GraphBLAS (CPU)           │  HRT / PyTorch (GPU)               │
    │  - Graph algorithms        │  - Ingestion & AM construction     │
    │  - Lattice exploration     │  - Batch tensor operations         │
    │  - Pattern matching        │  - N-gram processing               │
    │  - Transitive closure      │  - Sparse matrix multiply          │
    └─────────────────────────────────────────────────────────────────┘
                     ↑                           ↑
                     └───── SparseAM3D ──────────┘
                           (shared data structure)

Installation:
    pip install python-graphblas
    # or: conda install -c conda-forge python-graphblas

Usage:
    from core.graphblas_ops import (
        am_to_graphblas,
        graphblas_to_am,
        bfs,
        connected_components,
        pagerank,
        transitive_closure,
    )
    
    # Convert SparseAM3D layer to GraphBLAS matrix
    gb_matrix = am_to_graphblas(am, layer=0)
    
    # Run graph algorithms
    levels = bfs(gb_matrix, source=0)
    components = connected_components(gb_matrix)
    ranks = pagerank(gb_matrix)
"""

from __future__ import annotations
from typing import Optional, List, Tuple, Dict, Any, TYPE_CHECKING
from dataclasses import dataclass
import numpy as np

# Lazy import for GraphBLAS (optional dependency)
try:
    import graphblas as gb
    from graphblas import Matrix, Vector, Scalar
    from graphblas import binary, monoid, semiring, unary
    GRAPHBLAS_AVAILABLE = True
except ImportError:
    GRAPHBLAS_AVAILABLE = False
    gb = None
    Matrix = None
    Vector = None

if TYPE_CHECKING:
    from .sparse_hrt_3d import SparseAM3D, Sparse3DConfig, Edge3D


def _check_graphblas():
    """Raise error if GraphBLAS not available."""
    if not GRAPHBLAS_AVAILABLE:
        raise ImportError(
            "python-graphblas is required for graph operations.\n"
            "Install with: pip install 'python-graphblas[default]'\n"
            "Or: conda install -c conda-forge python-graphblas"
        )


# =============================================================================
# SECTION 1: Converters (numpy/SparseAM3D ↔ GraphBLAS)
# =============================================================================

def numpy_to_graphblas(
    arr: np.ndarray,
    dtype: Any = None
) -> 'Matrix':
    """
    Convert a numpy 2D array to a GraphBLAS Matrix.
    
    Args:
        arr: 2D numpy array (dense adjacency matrix)
        dtype: GraphBLAS dtype (default: FP64)
        
    Returns:
        GraphBLAS Matrix
    """
    _check_graphblas()
    
    if dtype is None:
        dtype = gb.dtypes.FP64
    
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array, got {arr.ndim}D")
    
    # Find non-zero entries
    rows, cols = np.nonzero(arr)
    vals = arr[rows, cols].astype(np.float64)
    
    if len(rows) == 0:
        return Matrix(dtype, nrows=arr.shape[0], ncols=arr.shape[1])
    
    M = Matrix.from_coo(
        rows.tolist(), cols.tolist(), vals.tolist(),
        dtype=dtype, nrows=arr.shape[0], ncols=arr.shape[1]
    )
    return M


def am_to_graphblas(
    am: 'SparseAM3D',
    layer: int = 0,
    dtype: Any = None
) -> 'Matrix':
    """
    Convert a SparseAM3D layer to a GraphBLAS Matrix.
    
    Args:
        am: SparseAM3D adjacency matrix
        layer: N-gram layer to convert (0=unigram, 1=bigram, etc.)
        dtype: GraphBLAS dtype (default: FP64)
        
    Returns:
        GraphBLAS Matrix
    """
    _check_graphblas()
    
    if dtype is None:
        dtype = gb.dtypes.FP64
    
    # Extract edges from the specified layer
    rows, cols, vals = [], [], []
    for row, col, val in am.tensor.layer_edges(layer):
        rows.append(row)
        cols.append(col)
        vals.append(float(val))
    
    if not rows:
        # Empty matrix
        dim = am.config.dimension
        return Matrix(dtype, nrows=dim, ncols=dim)
    
    # Build GraphBLAS matrix from COO format
    dim = am.config.dimension
    M = Matrix.from_coo(rows, cols, vals, dtype=dtype, nrows=dim, ncols=dim)
    return M


def am_all_layers_to_graphblas(
    am: 'SparseAM3D',
    dtype: Any = None
) -> List['Matrix']:
    """
    Convert all layers of SparseAM3D to GraphBLAS matrices.
    
    Returns:
        List of GraphBLAS Matrices, one per layer
    """
    _check_graphblas()
    
    matrices = []
    for layer in range(am.config.max_n):
        matrices.append(am_to_graphblas(am, layer=layer, dtype=dtype))
    return matrices


def graphblas_to_edges(
    M: 'Matrix',
    layer: int = 0
) -> List['Edge3D']:
    """
    Convert a GraphBLAS Matrix back to Edge3D list.
    
    Args:
        M: GraphBLAS Matrix
        layer: Layer index to assign to edges
        
    Returns:
        List of Edge3D
    """
    _check_graphblas()
    from .sparse_hrt_3d import Edge3D
    
    rows, cols, vals = M.to_coo()
    edges = []
    for r, c, v in zip(rows, cols, vals):
        edges.append(Edge3D(n=layer, row=int(r), col=int(c), value=float(v)))
    return edges


def graphblas_to_am(
    matrices: List['Matrix'],
    config: 'Sparse3DConfig'
) -> 'SparseAM3D':
    """
    Convert GraphBLAS matrices back to SparseAM3D.
    
    Args:
        matrices: List of GraphBLAS matrices (one per layer)
        config: Sparse3DConfig for the AM
        
    Returns:
        SparseAM3D
    """
    _check_graphblas()
    from .sparse_hrt_3d import SparseAM3D, Edge3D
    
    all_edges = []
    for layer, M in enumerate(matrices):
        all_edges.extend(graphblas_to_edges(M, layer=layer))
    
    return SparseAM3D.from_edges(all_edges, config)


# =============================================================================
# SECTION 2: Graph Algorithms
# =============================================================================

def bfs(
    M: 'Matrix',
    source: int,
    max_depth: Optional[int] = None
) -> 'Vector':
    """
    Breadth-First Search from source node.
    
    Returns a Vector where v[i] = depth from source to node i.
    Unreachable nodes have no entry (sparse).
    
    Args:
        M: Adjacency matrix (GraphBLAS Matrix)
        source: Starting node index
        max_depth: Maximum depth to explore (None = unlimited)
        
    Returns:
        GraphBLAS Vector of depths
    """
    _check_graphblas()
    
    n = M.nrows
    
    # Initialize: level vector with source at depth 0
    levels = Vector(gb.dtypes.INT64, size=n)
    levels[source] = 0
    
    # Frontier: nodes at current level
    frontier = Vector(gb.dtypes.BOOL, size=n)
    frontier[source] = True
    
    # For row-major adjacency M[i,j] = i->j, use M.T to follow outgoing edges
    MT = M.T
    
    depth = 0
    while frontier.nvals > 0:
        if max_depth is not None and depth >= max_depth:
            break
        
        depth += 1
        
        # Expand frontier: find all neighbors of current frontier
        # Using M.T because M[i,j]=1 means i->j, so M.T.mxv finds destinations
        new_frontier = Vector(gb.dtypes.BOOL, size=n)
        new_frontier(mask=~levels.S) << MT.mxv(frontier, semiring.any_pair)
        
        # Update levels for newly discovered nodes
        if new_frontier.nvals > 0:
            levels(mask=new_frontier.S) << depth
        
        frontier = new_frontier
    
    return levels


def connected_components(M: 'Matrix') -> 'Vector':
    """
    Find connected components using label propagation.
    
    Returns a Vector where v[i] = component ID for node i.
    Uses FastSV algorithm (fastest known for GraphBLAS).
    
    Args:
        M: Adjacency matrix (should be symmetric for undirected)
        
    Returns:
        GraphBLAS Vector of component IDs
    """
    _check_graphblas()
    
    n = M.nrows
    
    # Initialize: each node in its own component
    parent = Vector.from_coo(range(n), range(n), dtype=gb.dtypes.INT64)
    
    # Make matrix symmetric (undirected graph)
    A = M.dup()
    A << A.ewise_add(A.T, binary.any)
    
    changed = True
    while changed:
        # Hooking: parent[i] = min(parent[j]) for all neighbors j
        new_parent = Vector(gb.dtypes.INT64, size=n)
        new_parent << A.mxv(parent, semiring.min_second)
        
        # Min with current parent
        new_parent << new_parent.ewise_mult(parent, binary.min)
        
        # Check convergence
        diff = Vector(gb.dtypes.INT64, size=n)
        diff << new_parent.ewise_mult(parent, binary.minus)
        changed = diff.reduce(monoid.lor).value
        
        parent = new_parent
    
    return parent


def pagerank(
    M: 'Matrix',
    damping: float = 0.85,
    max_iters: int = 100,
    tol: float = 1e-6
) -> 'Vector':
    """
    Compute PageRank scores.
    
    Args:
        M: Adjacency matrix
        damping: Damping factor (default 0.85)
        max_iters: Maximum iterations
        tol: Convergence tolerance
        
    Returns:
        GraphBLAS Vector of PageRank scores
    """
    _check_graphblas()
    
    n = M.nrows
    
    # Compute out-degrees (row sums)
    out_degree = Vector(gb.dtypes.FP64, size=n)
    out_degree << M.reduce_rowwise(monoid.plus)
    
    # Handle dangling nodes: set 0 degrees to 1 to avoid div-by-zero
    # First, set all non-zero entries to themselves, then fill zeros with 1
    idx, vals = out_degree.to_coo()
    for i in range(n):
        if i not in set(idx):
            out_degree[i] = 1.0
    
    # Normalize: d_inv = 1/degree
    d_inv = Vector(gb.dtypes.FP64, size=n)
    d_inv << out_degree.apply(unary.minv)
    
    # Initialize ranks uniformly
    rank = Vector.from_coo(range(n), [1.0/n]*n, dtype=gb.dtypes.FP64)
    
    teleport = (1.0 - damping) / n
    
    for _ in range(max_iters):
        # new_rank = damping * (P^T @ rank) + teleport
        # First: weighted_rank = rank * d_inv (element-wise)
        weighted = Vector(gb.dtypes.FP64, size=n)
        weighted << rank.ewise_mult(d_inv, binary.times)
        
        # Then: new_rank = M^T @ weighted_rank
        new_rank = Vector(gb.dtypes.FP64, size=n)
        new_rank << M.T.mxv(weighted, semiring.plus_times)
        
        # Apply damping and teleport
        new_rank << new_rank.apply(binary.times, left=damping)
        new_rank << new_rank.apply(binary.plus, left=teleport)
        
        # Check convergence
        diff = Vector(gb.dtypes.FP64, size=n)
        diff << new_rank.ewise_add(rank, binary.minus)
        diff << diff.apply(unary.abs)
        max_diff_scalar = diff.reduce(monoid.max)
        max_diff = max_diff_scalar.value if max_diff_scalar.value is not None else 0.0
        
        rank = new_rank
        
        if max_diff < tol:
            break
    
    return rank


def transitive_closure(
    M: 'Matrix',
    max_depth: Optional[int] = None
) -> 'Matrix':
    """
    Compute transitive closure of the graph.
    
    Result[i,j] = True if there's a path from i to j.
    
    Args:
        M: Adjacency matrix
        max_depth: Maximum path length (None = unlimited)
        
    Returns:
        GraphBLAS Matrix (boolean)
    """
    _check_graphblas()
    
    n = M.nrows
    
    # Convert to boolean
    A = Matrix(gb.dtypes.BOOL, nrows=n, ncols=n)
    A << M.apply(unary.one)
    
    # Initialize closure with identity + A
    C = A.dup()
    
    # Warshall's algorithm using matrix multiply
    depth = 0
    while True:
        if max_depth is not None and depth >= max_depth:
            break
        
        prev_nvals = C.nvals
        
        # C = C | (C @ A)
        C << C.ewise_add(C.mxm(A, semiring.any_pair), binary.any)
        
        depth += 1
        
        # Check convergence
        if C.nvals == prev_nvals:
            break
    
    return C


def shortest_paths(
    M: 'Matrix',
    source: int
) -> 'Vector':
    """
    Single-source shortest paths (Bellman-Ford).
    
    Args:
        M: Weighted adjacency matrix
        source: Source node
        
    Returns:
        Vector of shortest distances from source
    """
    _check_graphblas()
    
    n = M.nrows
    
    # Initialize distances (infinity except source)
    dist = Vector(gb.dtypes.FP64, size=n)
    dist[source] = 0.0
    
    for _ in range(n - 1):
        # Relax edges: dist[j] = min(dist[j], dist[i] + M[i,j])
        new_dist = Vector(gb.dtypes.FP64, size=n)
        new_dist << M.T.mxv(dist, semiring.min_plus)
        
        # Keep minimum
        dist << dist.ewise_add(new_dist, binary.min)
    
    return dist


# =============================================================================
# SECTION 3: Lattice-Specific Operations
# =============================================================================

def multi_source_bfs(
    M: 'Matrix',
    source_indices: List[int],
    max_hops: int = 3
) -> Dict[int, int]:
    """
    BFS from multiple source nodes.
    
    Args:
        M: GraphBLAS Matrix
        source_indices: Starting node indices
        max_hops: Maximum hop count
        
    Returns:
        Dict mapping node index to minimum distance
    """
    _check_graphblas()
    
    n = M.nrows
    MT = M.T  # For outgoing edges
    
    # Initialize frontier with all sources
    frontier = Vector(gb.dtypes.BOOL, size=n)
    for s in source_indices:
        frontier[s] = True
    
    # Track visited with distances
    visited = Vector(gb.dtypes.INT64, size=n)
    for s in source_indices:
        visited[s] = 0
    
    for hop in range(1, max_hops + 1):
        # Expand frontier
        new_frontier = Vector(gb.dtypes.BOOL, size=n)
        new_frontier(mask=~visited.S) << MT.mxv(frontier, semiring.any_pair)
        
        if new_frontier.nvals == 0:
            break
        
        # Update distances
        visited(mask=new_frontier.S) << hop
        frontier = new_frontier
    
    # Convert to dict
    indices, values = visited.to_coo()
    return dict(zip(indices, values))


def layer_reachability(
    am: 'SparseAM3D',
    source_indices: List[int],
    layer: int = 0,
    max_hops: int = 3
) -> Dict[int, int]:
    """
    Find all nodes reachable from source nodes within max_hops.
    
    Args:
        am: SparseAM3D
        source_indices: Starting node indices
        layer: N-gram layer
        max_hops: Maximum hop count
        
    Returns:
        Dict mapping node index to minimum distance
    """
    _check_graphblas()
    
    M = am_to_graphblas(am, layer=layer)
    n = M.nrows
    
    # Initialize frontier with all sources
    frontier = Vector(gb.dtypes.BOOL, size=n)
    for s in source_indices:
        frontier[s] = True
    
    # Track visited with distances
    visited = Vector(gb.dtypes.INT64, size=n)
    for s in source_indices:
        visited[s] = 0
    
    for hop in range(1, max_hops + 1):
        # Expand frontier
        new_frontier = Vector(gb.dtypes.BOOL, size=n)
        new_frontier(mask=~visited.S) << M.mxv(frontier, semiring.any_pair)
        
        if new_frontier.nvals == 0:
            break
        
        # Update distances
        visited(mask=new_frontier.S) << hop
        frontier = new_frontier
    
    # Convert to dict
    indices, values = visited.to_coo()
    return dict(zip(indices, values))


def node_importance(
    M: 'Matrix',
    method: str = 'pagerank'
) -> 'Vector':
    """
    Compute node importance scores for a GraphBLAS matrix.
    
    Args:
        M: GraphBLAS Matrix
        method: 'pagerank' or 'degree'
        
    Returns:
        GraphBLAS Vector of importance scores
    """
    _check_graphblas()
    
    if method == 'pagerank':
        return pagerank(M)
    elif method == 'degree':
        scores = Vector(gb.dtypes.FP64, size=M.nrows)
        scores << M.reduce_rowwise(monoid.plus)
        return scores
    else:
        raise ValueError(f"Unknown method: {method}")


def layer_importance(
    am: 'SparseAM3D',
    layer: int = 0,
    method: str = 'pagerank'
) -> Dict[int, float]:
    """
    Compute node importance scores for a layer.
    
    Args:
        am: SparseAM3D
        layer: N-gram layer
        method: 'pagerank' or 'degree'
        
    Returns:
        Dict mapping node index to importance score
    """
    _check_graphblas()
    
    M = am_to_graphblas(am, layer=layer)
    
    if method == 'pagerank':
        scores = pagerank(M)
    elif method == 'degree':
        scores = Vector(gb.dtypes.FP64, size=M.nrows)
        scores << M.reduce_rowwise(monoid.plus)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    indices, values = scores.to_coo()
    return dict(zip(indices, values))


def cross_layer_paths(
    am: 'SparseAM3D',
    source: int,
    target: int
) -> List[Tuple[int, List[int]]]:
    """
    Find paths from source to target across all layers.
    
    Returns list of (layer, path) where path exists.
    
    Args:
        am: SparseAM3D
        source: Source node index
        target: Target node index
        
    Returns:
        List of (layer_index, path_length) tuples
    """
    _check_graphblas()
    
    results = []
    for layer in range(am.config.max_n):
        M = am_to_graphblas(am, layer=layer)
        levels = bfs(M, source=source)
        
        if target in levels:
            indices, values = levels.to_coo()
            idx_map = dict(zip(indices, values))
            results.append((layer, idx_map[target]))
    
    return results


# =============================================================================
# SECTION 4: Lattice Matching & Entanglement
# =============================================================================
# Match two lattices (graphs) by comparing node degrees and BSS metrics.
# This provides structural alignment between different graph representations,
# enabling "entanglement" - finding correspondences between lattices.

@dataclass
class NodeSignature:
    """Structural signature of a node for matching."""
    index: int
    in_degree: int
    out_degree: int
    total_degree: int
    # Directed neighbor sets (for BSS tau/rho)
    in_neighbors: frozenset    # Nodes with edges TO this node
    out_neighbors: frozenset   # Nodes with edges FROM this node
    # Extended context
    two_hop_out: frozenset     # 2-hop outgoing neighbors
    two_hop_in: frozenset      # 2-hop incoming neighbors
    
    def __hash__(self):
        return hash((self.in_degree, self.out_degree, 
                     len(self.in_neighbors), len(self.out_neighbors)))


@dataclass 
class LatticeMatch:
    """Result of matching two lattices."""
    node_mapping: Dict[int, int]        # A_idx -> B_idx
    confidence: float                    # Overall match confidence
    tau_similarity: float               # BSS tau-based similarity
    rho_similarity: float               # BSS rho-based similarity
    degree_similarity: float            # Degree distribution similarity
    matched_count: int                  # Number of matched nodes
    total_A: int                        # Total nodes in lattice A
    total_B: int                        # Total nodes in lattice B


def compute_node_signatures(
    M: 'Matrix',
    compute_two_hop: bool = True
) -> Dict[int, NodeSignature]:
    """
    Compute structural signatures for all nodes in a graph.
    
    Each signature captures directed neighborhood structure:
    - In-degree / Out-degree
    - In-neighbors / Out-neighbors (for directed BSS tau/rho)
    - Two-hop neighbors in both directions
    
    Args:
        M: Adjacency matrix (M[i,j]=1 means i→j)
        compute_two_hop: Whether to compute 2-hop neighborhoods
        
    Returns:
        Dict mapping node index to NodeSignature
    """
    _check_graphblas()
    
    n = M.nrows
    signatures = {}
    
    # Compute in-degrees and out-degrees
    out_deg = Vector(gb.dtypes.INT64, size=n)
    out_deg << M.reduce_rowwise(monoid.plus)
    
    in_deg = Vector(gb.dtypes.INT64, size=n)
    in_deg << M.reduce_columnwise(monoid.plus)
    
    # Extract as dicts
    out_idx, out_vals = out_deg.to_coo()
    in_idx, in_vals = in_deg.to_coo()
    
    out_map = dict(zip(out_idx, out_vals))
    in_map = dict(zip(in_idx, in_vals))
    
    # Get edges for neighbor computation
    rows, cols, _ = M.to_coo()
    
    # Build directed adjacency lists
    neighbors_out: Dict[int, set] = {i: set() for i in range(n)}
    neighbors_in: Dict[int, set] = {i: set() for i in range(n)}
    
    for r, c in zip(rows, cols):
        neighbors_out[int(r)].add(int(c))  # r → c
        neighbors_in[int(c)].add(int(r))   # c ← r
    
    # Compute 2-hop neighbors in both directions
    two_hop_out: Dict[int, set] = {i: set() for i in range(n)}
    two_hop_in: Dict[int, set] = {i: set() for i in range(n)}
    
    if compute_two_hop:
        # M² gives 2-hop outgoing paths
        M2 = Matrix(gb.dtypes.BOOL, nrows=n, ncols=n)
        M2 << M.mxm(M, semiring.any_pair)
        rows2, cols2, _ = M2.to_coo()
        
        for r, c in zip(rows2, cols2):
            if int(r) != int(c):
                two_hop_out[int(r)].add(int(c))
                two_hop_in[int(c)].add(int(r))
    
    # Build signatures
    for i in range(n):
        in_d = int(in_map.get(i, 0))
        out_d = int(out_map.get(i, 0))
        
        signatures[i] = NodeSignature(
            index=i,
            in_degree=in_d,
            out_degree=out_d,
            total_degree=in_d + out_d,
            in_neighbors=frozenset(neighbors_in[i]),
            out_neighbors=frozenset(neighbors_out[i]),
            two_hop_out=frozenset(two_hop_out[i]),
            two_hop_in=frozenset(two_hop_in[i])
        )
    
    return signatures


def bss_tau(set_a: frozenset, set_b: frozenset) -> float:
    """
    Compute BSS τ (directed inclusion measure): |A ∩ B| / |A|
    
    Measures how much of A is included in B.
    This is DIRECTED: tau(A,B) ≠ tau(B,A) in general.
    
    Matches kernel.py semantics:
        BSS_τ(A, B) = |A ∩ B| / |A|
    
    Returns:
        Float in [0, 1] where 1 = A ⊆ B (A fully contained in B)
    """
    if not set_a:
        return 1.0 if not set_b else 0.0  # Empty set: contained in everything
    
    intersection = len(set_a & set_b)
    return intersection / len(set_a)


def bss_rho(set_a: frozenset, set_b: frozenset) -> float:
    """
    Compute BSS ρ (directed exclusion measure): |A - B| / |A| = 1 - τ(A,B)
    
    Measures how much of A is excluded from B.
    This is DIRECTED: rho(A,B) ≠ rho(B,A) in general.
    
    Matches kernel.py semantics:
        BSS_ρ(A, B) = |A - B| / |A| = 1 - BSS_τ(A, B)
    
    Returns:
        Float in [0, 1] where 0 = A ⊆ B (nothing excluded)
    """
    return 1.0 - bss_tau(set_a, set_b)


def bss_symmetric(set_a: frozenset, set_b: frozenset) -> Tuple[float, float, float, float]:
    """
    Compute symmetric BSS measures in both directions.
    
    Returns:
        (tau_ab, rho_ab, tau_ba, rho_ba)
        
    Where:
        tau_ab = how much of A is in B
        rho_ab = how much of A is excluded from B  
        tau_ba = how much of B is in A
        rho_ba = how much of B is excluded from A
    """
    tau_ab = bss_tau(set_a, set_b)
    rho_ab = 1.0 - tau_ab
    tau_ba = bss_tau(set_b, set_a)
    rho_ba = 1.0 - tau_ba
    return (tau_ab, rho_ab, tau_ba, rho_ba)


def match_nodes_by_degree(
    sigs_A: Dict[int, NodeSignature],
    sigs_B: Dict[int, NodeSignature],
    degree_tolerance: int = 0
) -> List[Tuple[int, int, float]]:
    """
    Find candidate node matches based on degree similarity.
    
    Args:
        sigs_A: Signatures from lattice A
        sigs_B: Signatures from lattice B
        degree_tolerance: Allow degree difference up to this value
        
    Returns:
        List of (A_idx, B_idx, similarity) tuples
    """
    candidates = []
    
    for idx_a, sig_a in sigs_A.items():
        for idx_b, sig_b in sigs_B.items():
            # Check degree compatibility
            in_diff = abs(sig_a.in_degree - sig_b.in_degree)
            out_diff = abs(sig_a.out_degree - sig_b.out_degree)
            
            if in_diff <= degree_tolerance and out_diff <= degree_tolerance:
                # Compute degree similarity
                max_in = max(sig_a.in_degree, sig_b.in_degree, 1)
                max_out = max(sig_a.out_degree, sig_b.out_degree, 1)
                
                degree_sim = 1.0 - (in_diff / max_in + out_diff / max_out) / 2
                candidates.append((idx_a, idx_b, degree_sim))
    
    # Sort by similarity descending
    candidates.sort(key=lambda x: -x[2])
    return candidates


def match_lattices(
    M_A: 'Matrix',
    M_B: 'Matrix',
    degree_tolerance: int = 1,
    tau_threshold: float = 0.3,
    rho_threshold: float = 0.7,
    use_two_hop: bool = True
) -> LatticeMatch:
    """
    Match two lattices by node degrees and directed BSS metrics.
    
    Uses directed BSS (matching kernel.py semantics):
    - Compares in-neighbors and out-neighbors separately
    - tau(A,B) = |A∩B|/|A| measures inclusion of A in B
    - rho(A,B) = 1 - tau(A,B) measures exclusion
    
    Algorithm:
    1. Compute node signatures for both lattices
    2. Find candidate matches by degree similarity
    3. Score using directed BSS on in/out neighborhoods
    4. Greedy assignment for best mapping
    
    Args:
        M_A: Adjacency matrix of lattice A
        M_B: Adjacency matrix of lattice B
        degree_tolerance: Max degree difference for candidates
        tau_threshold: Minimum tau for valid match (higher = stricter)
        rho_threshold: Maximum rho for valid match (lower = stricter)
        use_two_hop: Use 2-hop neighborhoods for matching
        
    Returns:
        LatticeMatch with node mapping and similarity scores
    """
    _check_graphblas()
    
    # Step 1: Compute signatures
    sigs_A = compute_node_signatures(M_A, compute_two_hop=use_two_hop)
    sigs_B = compute_node_signatures(M_B, compute_two_hop=use_two_hop)
    
    # Step 2: Find degree-compatible candidates
    candidates = match_nodes_by_degree(sigs_A, sigs_B, degree_tolerance)
    
    # Step 3: Score candidates using directed BSS metrics
    scored_matches: List[Tuple[int, int, float, float, float]] = []
    
    for idx_a, idx_b, deg_sim in candidates:
        sig_a = sigs_A[idx_a]
        sig_b = sigs_B[idx_b]
        
        # Directed BSS on OUT-neighbors (forward direction)
        tau_out_ab = bss_tau(sig_a.out_neighbors, sig_b.out_neighbors)
        tau_out_ba = bss_tau(sig_b.out_neighbors, sig_a.out_neighbors)
        
        # Directed BSS on IN-neighbors (backward direction)
        tau_in_ab = bss_tau(sig_a.in_neighbors, sig_b.in_neighbors)
        tau_in_ba = bss_tau(sig_b.in_neighbors, sig_a.in_neighbors)
        
        # Combined tau: average of all directions
        # High tau in both directions suggests structural equivalence
        tau = (tau_out_ab + tau_out_ba + tau_in_ab + tau_in_ba) / 4
        
        # Rho is the complement
        rho = 1.0 - tau
        
        # Include 2-hop context if available
        if use_two_hop:
            if sig_a.two_hop_out and sig_b.two_hop_out:
                tau_2hop_out = (bss_tau(sig_a.two_hop_out, sig_b.two_hop_out) +
                               bss_tau(sig_b.two_hop_out, sig_a.two_hop_out)) / 2
                tau = 0.7 * tau + 0.15 * tau_2hop_out
            
            if sig_a.two_hop_in and sig_b.two_hop_in:
                tau_2hop_in = (bss_tau(sig_a.two_hop_in, sig_b.two_hop_in) +
                               bss_tau(sig_b.two_hop_in, sig_a.two_hop_in)) / 2
                tau = tau + 0.15 * tau_2hop_in
            
            rho = 1.0 - tau
        
        # Check thresholds (tau high enough, rho low enough)
        if tau >= tau_threshold and rho <= rho_threshold:
            combined = 0.3 * deg_sim + 0.5 * tau + 0.2 * (1.0 - rho)
            scored_matches.append((idx_a, idx_b, combined, tau, rho))
    
    # Step 5: Greedy assignment (best match first, no duplicates)
    scored_matches.sort(key=lambda x: -x[2])
    
    node_mapping: Dict[int, int] = {}
    used_B: set = set()
    tau_sum = 0.0
    rho_sum = 0.0
    deg_sum = 0.0
    
    for idx_a, idx_b, score, tau, rho in scored_matches:
        if idx_a not in node_mapping and idx_b not in used_B:
            node_mapping[idx_a] = idx_b
            used_B.add(idx_b)
            tau_sum += tau
            rho_sum += rho
            deg_sum += score
    
    matched = len(node_mapping)
    
    # Compute overall similarities
    tau_similarity = tau_sum / matched if matched > 0 else 0.0
    rho_similarity = rho_sum / matched if matched > 0 else 0.0
    degree_similarity = deg_sum / matched if matched > 0 else 0.0
    
    # Overall confidence based on coverage and quality
    coverage = matched / max(len(sigs_A), len(sigs_B))
    confidence = coverage * (0.3 * degree_similarity + 0.4 * tau_similarity + 0.3 * rho_similarity)
    
    return LatticeMatch(
        node_mapping=node_mapping,
        confidence=confidence,
        tau_similarity=tau_similarity,
        rho_similarity=rho_similarity,
        degree_similarity=degree_similarity,
        matched_count=matched,
        total_A=len(sigs_A),
        total_B=len(sigs_B)
    )


def find_entangled_subgraphs(
    M_A: 'Matrix',
    M_B: 'Matrix',
    min_size: int = 3,
    tau_threshold: float = 0.5
) -> List[Tuple[List[int], List[int], float]]:
    """
    Find entangled (structurally similar) subgraphs between two lattices.
    
    Uses connected component analysis + BSS matching to identify
    corresponding substructures.
    
    Args:
        M_A: Lattice A adjacency matrix
        M_B: Lattice B adjacency matrix
        min_size: Minimum subgraph size to consider
        tau_threshold: Minimum tau similarity
        
    Returns:
        List of (A_nodes, B_nodes, similarity) tuples
    """
    _check_graphblas()
    
    # Find connected components in each lattice
    comp_A = connected_components(M_A)
    comp_B = connected_components(M_B)
    
    # Group nodes by component
    def group_by_component(comp_vec):
        idx, vals = comp_vec.to_coo()
        groups: Dict[int, List[int]] = {}
        for i, c in zip(idx, vals):
            c = int(c)
            if c not in groups:
                groups[c] = []
            groups[c].append(int(i))
        return groups
    
    groups_A = group_by_component(comp_A)
    groups_B = group_by_component(comp_B)
    
    # Filter by minimum size
    groups_A = {k: v for k, v in groups_A.items() if len(v) >= min_size}
    groups_B = {k: v for k, v in groups_B.items() if len(v) >= min_size}
    
    # Compare component structures
    entangled = []
    
    for comp_id_a, nodes_a in groups_A.items():
        nodes_a_set = frozenset(nodes_a)
        
        # Get degree profile of component A
        sigs_A = compute_node_signatures(M_A, compute_two_hop=False)
        degrees_A = sorted([sigs_A[n].total_degree for n in nodes_a])
        
        for comp_id_b, nodes_b in groups_B.items():
            if abs(len(nodes_a) - len(nodes_b)) > len(nodes_a) // 2:
                continue  # Size too different
            
            nodes_b_set = frozenset(nodes_b)
            
            # Get degree profile of component B
            sigs_B = compute_node_signatures(M_B, compute_two_hop=False)
            degrees_B = sorted([sigs_B[n].total_degree for n in nodes_b])
            
            # Compare degree distributions (structural fingerprint)
            min_len = min(len(degrees_A), len(degrees_B))
            if min_len == 0:
                continue
                
            degree_diff = sum(abs(degrees_A[i] - degrees_B[i]) 
                            for i in range(min_len)) / min_len
            degree_sim = 1.0 / (1.0 + degree_diff)
            
            # Size similarity
            size_sim = min(len(nodes_a), len(nodes_b)) / max(len(nodes_a), len(nodes_b))
            
            # Combined similarity
            similarity = 0.5 * degree_sim + 0.5 * size_sim
            
            if similarity >= tau_threshold:
                entangled.append((nodes_a, nodes_b, similarity))
    
    # Sort by similarity
    entangled.sort(key=lambda x: -x[2])
    
    return entangled


def compute_entanglement_score(
    M_A: 'Matrix',
    M_B: 'Matrix'
) -> float:
    """
    Compute overall entanglement score between two lattices.
    
    High score indicates strong structural correspondence.
    
    Returns:
        Float in [0, 1] where 1 = isomorphic structures
    """
    match = match_lattices(M_A, M_B)
    
    # Coverage: what fraction of nodes are matched
    coverage = match.matched_count / max(match.total_A, match.total_B)
    
    # Quality: how good are the matches
    quality = (match.tau_similarity + match.rho_similarity) / 2
    
    # Entanglement = coverage * quality
    return coverage * quality


# =============================================================================
# SECTION 5: Token Reordering (Post-Disambiguation)
# =============================================================================
# After disambiguation extracts candidate tokens from HLLSet/AM structure,
# they come out as an unordered set. These functions use adjacency relationships
# to reconstruct sequential order by finding paths from START to END.

@dataclass
class TokenPath:
    """A path through the token graph."""
    tokens: List[int]          # Ordered token indices
    weight: float              # Path weight (sum of edge weights)
    hops: int                  # Number of edges in path
    
    def __repr__(self):
        return f"TokenPath(len={len(self.tokens)}, weight={self.weight:.3f})"


def find_all_paths(
    M: 'Matrix',
    start: int,
    end: int,
    max_depth: int = 50,
    max_paths: int = 10,
    candidate_set: Optional[set] = None
) -> List[TokenPath]:
    """
    Find all paths from start to end node, returning multiple options.
    
    Used after disambiguation to reconstruct token sequences.
    The adjacency matrix encodes token transitions (bigrams), so
    paths represent valid token orderings.
    
    Args:
        M: Adjacency matrix (token transitions)
        start: START token index
        end: END token index  
        max_depth: Maximum path length
        max_paths: Maximum number of paths to return
        candidate_set: If provided, only traverse nodes in this set
                      (the disambiguated token candidates)
        
    Returns:
        List of TokenPath objects, sorted by weight (ascending = shortest)
    """
    _check_graphblas()
    
    n = M.nrows
    
    # Extract adjacency as dict for path enumeration
    rows, cols, vals = M.to_coo()
    adj: Dict[int, List[Tuple[int, float]]] = {}
    for r, c, v in zip(rows, cols, vals):
        # Filter to candidate set if provided
        if candidate_set is not None:
            if r not in candidate_set or c not in candidate_set:
                continue
        if r not in adj:
            adj[r] = []
        adj[r].append((c, float(v)))
    
    # DFS to find all paths
    paths: List[TokenPath] = []
    
    def dfs(node: int, path: List[int], weight: float, visited: set):
        if len(paths) >= max_paths:
            return
        if len(path) > max_depth:
            return
        
        if node == end:
            paths.append(TokenPath(
                tokens=path.copy(),
                weight=weight,
                hops=len(path) - 1
            ))
            return
        
        for neighbor, edge_weight in adj.get(node, []):
            if neighbor not in visited:
                visited.add(neighbor)
                path.append(neighbor)
                dfs(neighbor, path, weight + edge_weight, visited)
                path.pop()
                visited.remove(neighbor)
    
    # Start DFS from start node
    dfs(start, [start], 0.0, {start})
    
    # Sort by weight (shortest/lightest first)
    paths.sort(key=lambda p: p.weight)
    
    return paths


def find_shortest_path(
    M: 'Matrix',
    start: int,
    end: int,
    candidate_set: Optional[set] = None
) -> Optional[TokenPath]:
    """
    Find the shortest (minimum weight) path from start to end.
    
    Convenience wrapper around find_all_paths for single-path retrieval.
    
    Args:
        M: Adjacency matrix
        start: START token index
        end: END token index
        candidate_set: Optional filter to only use these nodes
        
    Returns:
        TokenPath or None if no path exists
    """
    paths = find_all_paths(M, start, end, max_paths=1, candidate_set=candidate_set)
    return paths[0] if paths else None


def find_longest_path(
    M: 'Matrix',
    start: int,
    end: int,
    candidate_set: Optional[set] = None,
    max_depth: int = 100
) -> Optional[TokenPath]:
    """
    Find the longest path (most tokens) from start to end.
    
    For sequence reconstruction, we typically want the path that
    includes the most tokens from our candidate set.
    
    Args:
        M: Adjacency matrix
        start: START token index
        end: END token index
        candidate_set: Optional filter to only use these nodes
        max_depth: Maximum path length to search
        
    Returns:
        TokenPath with most tokens, or None if no path exists
    """
    paths = find_all_paths(
        M, start, end, 
        max_depth=max_depth, 
        max_paths=100,  # Get many paths to find longest
        candidate_set=candidate_set
    )
    if not paths:
        return None
    
    # Sort by number of tokens (descending) then by weight (ascending)
    paths.sort(key=lambda p: (-len(p.tokens), p.weight))
    return paths[0]
    return paths[0] if paths else None


def reorder_tokens(
    M: 'Matrix',
    token_indices: List[int],
    start_token: int,
    end_token: int,
    return_all: bool = False,
    max_paths: int = 5
) -> List[TokenPath]:
    """
    Reorder a set of disambiguated tokens using graph structure.
    
    Given an unordered set of token indices extracted during disambiguation,
    finds valid orderings by computing paths from START to END that
    traverse only the candidate tokens.
    
    Args:
        M: Adjacency matrix encoding token transitions
        token_indices: Unordered list of token indices from disambiguation
        start_token: Index of START token
        end_token: Index of END token
        return_all: If True, return all found paths; else just the best
        max_paths: Maximum paths to return when return_all=True
        
    Returns:
        List of TokenPath objects representing valid orderings.
        If return_all=False, returns list with single best path.
        
    Example:
        >>> # After disambiguation gives us tokens {5, 12, 7, 3}
        >>> paths = reorder_tokens(M, [5, 12, 7, 3], start_token=0, end_token=1)
        >>> best_order = paths[0].tokens  # e.g., [0, 5, 7, 12, 3, 1]
    """
    # Build candidate set including start and end
    candidate_set = set(token_indices)
    candidate_set.add(start_token)
    candidate_set.add(end_token)
    
    paths = find_all_paths(
        M, 
        start=start_token, 
        end=end_token,
        max_paths=max_paths if return_all else 1,
        candidate_set=candidate_set
    )
    
    return paths


def rank_paths_by_ngram(
    paths: List[TokenPath],
    M_unigram: 'Matrix',
    M_bigram: Optional['Matrix'] = None,
    M_trigram: Optional['Matrix'] = None
) -> List[Tuple[TokenPath, float]]:
    """
    Rank candidate paths using n-gram layer evidence.
    
    Higher-order n-grams provide stronger evidence that a path
    is the correct original sequence.
    
    Args:
        paths: Candidate paths from find_all_paths
        M_unigram: Layer 0 (unigram) adjacency matrix
        M_bigram: Layer 1 (bigram) adjacency matrix, optional
        M_trigram: Layer 2 (trigram) adjacency matrix, optional
        
    Returns:
        List of (path, score) tuples, sorted by score (descending)
    """
    _check_graphblas()
    
    scored_paths = []
    
    for path in paths:
        score = 0.0
        tokens = path.tokens
        
        # Score bigram continuity
        if M_bigram is not None and len(tokens) >= 2:
            for i in range(len(tokens) - 1):
                t1, t2 = tokens[i], tokens[i + 1]
                # Check if bigram exists in bigram layer
                try:
                    val = M_bigram[t1, t2].value
                    if val is not None and val > 0:
                        score += 1.0  # Bigram evidence
                except:
                    pass
        
        # Score trigram continuity (stronger evidence)
        if M_trigram is not None and len(tokens) >= 3:
            for i in range(len(tokens) - 2):
                t1, t2, t3 = tokens[i], tokens[i + 1], tokens[i + 2]
                # For trigrams, check consecutive bigram pairs exist
                try:
                    val1 = M_trigram[t1, t2].value
                    val2 = M_trigram[t2, t3].value
                    if val1 is not None and val2 is not None:
                        if val1 > 0 and val2 > 0:
                            score += 2.0  # Trigram evidence (weighted higher)
                except:
                    pass
        
        scored_paths.append((path, score))
    
    # Sort by score descending
    scored_paths.sort(key=lambda x: -x[1])
    
    return scored_paths


# =============================================================================
# SECTION 5: Utility Functions
# =============================================================================

@dataclass
class GraphStats:
    """Statistics about a graph/layer."""
    nrows: int
    ncols: int
    nnz: int
    density: float
    avg_degree: float
    max_degree: int
    is_symmetric: bool


def compute_stats(M: 'Matrix') -> GraphStats:
    """
    Compute basic statistics about a GraphBLAS matrix.
    """
    _check_graphblas()
    
    n = M.nrows
    nnz = M.nvals
    density = nnz / (n * n) if n > 0 else 0.0
    
    # Compute degrees
    out_deg = Vector(gb.dtypes.INT64, size=n)
    out_deg << M.reduce_rowwise(monoid.plus)
    
    indices, values = out_deg.to_coo()
    avg_degree = sum(values) / len(values) if values else 0.0
    max_degree = max(values) if values else 0
    
    # Check symmetry (approximate)
    diff = Matrix(gb.dtypes.FP64, nrows=n, ncols=n)
    diff << M.ewise_add(M.T, binary.minus)
    is_symmetric = diff.nvals == 0 or diff.reduce_scalar(monoid.plus).value == 0
    
    return GraphStats(
        nrows=n,
        ncols=M.ncols,
        nnz=nnz,
        density=density,
        avg_degree=avg_degree,
        max_degree=max_degree,
        is_symmetric=is_symmetric
    )


def visualize_matrix(M: 'Matrix', max_display: int = 20) -> str:
    """
    Create ASCII visualization of sparse matrix structure.
    """
    _check_graphblas()
    
    n = min(M.nrows, max_display)
    rows, cols, _ = M.to_coo()
    
    # Create grid
    grid = [['.' for _ in range(n)] for _ in range(n)]
    
    for r, c in zip(rows, cols):
        if r < n and c < n:
            grid[r][c] = '█'
    
    # Build string
    lines = [''.join(row) for row in grid]
    if M.nrows > max_display:
        lines.append(f'... ({M.nrows - max_display} more rows)')
    
    return '\n'.join(lines)


__all__ = [
    # Availability check
    'GRAPHBLAS_AVAILABLE',
    
    # Converters
    'numpy_to_graphblas',
    'am_to_graphblas',
    'am_all_layers_to_graphblas',
    'graphblas_to_edges',
    'graphblas_to_am',
    
    # Graph Algorithms
    'bfs',
    'connected_components',
    'pagerank',
    'transitive_closure',
    'shortest_paths',
    
    # Lattice Operations
    'multi_source_bfs',
    'node_importance',
    'layer_reachability',
    'layer_importance',
    'cross_layer_paths',
    
    # Lattice Matching & Entanglement
    'NodeSignature',
    'LatticeMatch',
    'compute_node_signatures',
    'bss_tau',
    'bss_rho',
    'bss_symmetric',
    'match_nodes_by_degree',
    'match_lattices',
    'find_entangled_subgraphs',
    'compute_entanglement_score',
    
    # Token Reordering (Post-Disambiguation)
    'TokenPath',
    'find_all_paths',
    'find_shortest_path',
    'find_longest_path',
    'reorder_tokens',
    'rank_paths_by_ngram',
    
    # Utilities
    'GraphStats',
    'compute_stats',
    'visualize_matrix',
]
