"""
Fractal Manifold Core Module

This module provides the core components of the Fractal Manifold system:
- hllset: HLLSet class with C/Cython backend
- immutable_tensor: Generic immutable tensor foundation with PyTorch backend
- kernel: Stateless transformation engine (pure morphisms)
- hrt: Hash Relational Tensor with three-state evolution model
- fractal_core: Fractal loop primitives (NEW)

================================================================================
IMPORTANT: HLLSets are NOT sets containing tokens!
================================================================================

HLLSets are probabilistic register structures ("anti-sets") that:
- ABSORB tokens (hash them into registers)
- DO NOT STORE tokens (only register states remain)
- BEHAVE LIKE sets (union, intersection, cardinality estimation)
- ARE NOT sets (no element retrieval, no membership test)

================================================================================
THREE-LAYER ARCHITECTURE
================================================================================

LAYER 1: HLLSet (Register Layer)
- Works with individual HLLSets (register arrays)
- Compares: Register states, estimated cardinalities
- Morphism: Register-level comparison (NOT entanglement)

LAYER 2: Lattice (Structure Layer) - TRUE ENTANGLEMENT
- Works with HLLSetLattice objects
- Compares STRUCTURE (degree distributions, graph topology)
- BSS(τ, ρ): Determines morphism existence (construction phase)
- LatticeMorphism: Structure-level comparison
- Two lattices can be entangled even from completely different inputs!

LAYER 3: Fractal Loop (Scale Hierarchy) - SELF-SIMILARITY
- Treats lattice edges as tokens for next iteration
- N-tokenization creates scale hierarchy
- Overlap measure analyzes topology (after BSS builds lattice)
- Entanglement number E(m,n) measures cross-scale correlation

Architecture:
1. HLLSet: Named, immutable register array (C/Cython backend)
2. Kernel: Pure operations (absorb, union, intersection, difference)
3. HRT: Operation data structure (AM, Lattice, Covers) - immutable
4. Fractal: N-tokenization, edge→token, scale hierarchy, entanglement
5. OS: Reality interface (evolution orchestration, persistent storage)

The Fractal Loop:
    T → H → P → R → Lattice → Edges → (tokenize) → T' → H' → P' → ...
    
This creates self-similar structure across scales.
"""

from .hllset import HLLSet, compute_sha1

# DEPRECATED: Dense tensor/HRT imports (moved to deprecated/)
# These are kept for backward compatibility but will be removed in v1.0
from .deprecated.immutable_tensor import (
    ImmutableTensor,
    TensorEvolution,
    TensorEvolutionTriple,
    compute_element_hash,
    compute_aggregate_hash,
    compute_structural_hash,
)

from .kernel import (
    Kernel,
    Operation,
    record_operation,
    Morphism,
    LatticeMorphism,
    SingularityReport,
)

# DEPRECATED: Dense HRT (4GB per HRT - OOM on 64GB!)
# Use SparseHRT instead (2MB per 100K edges)
from .deprecated.hrt import (
    HRT,
    HRTConfig,
    HRTEvolution,
    HRTEvolutionTriple,
    AdjacencyMatrix,
    HLLSetLattice,
    BasicHLLSet,
    Cover,
)

from .algebra import (
    HLLCatalog,
    RelAlgebra,
    QueryResult,
    ColumnProfile,
    TableProfile,
)

from .entanglement import (
    # Fragment-based (structural)
    EntanglementFragment,
    EntanglementSubgraph,
    ExtendedEntanglement,
    CommonSubgraphExtractor,
    extract_entanglement,
    # Morphism-based
    EntanglementMeasurement,
    EntanglementMorphism,
    EntanglementEngine,
    # N-Edge based (stochastic) - NEW
    EdgeSignature,
    NEdgePath,
    EdgeLUT,
    NEdgeEntanglement,
    NEdgeExtractor,
    compute_nedge_entanglement,
)

from .fractal_core import (
    # N-Tokenization (Scale Hierarchy)
    n_tokenize,
    multi_scale_tokenize,
    tokens_to_hllsets,
    # Overlap Measure (Lattice Topology)
    overlap,
    overlap_from_cardinalities,
    # Edge Tokenization (Lattice → Tokens)
    LatticeEdge,
    extract_chains,
    chains_to_tokens,
    # Entanglement Number (Cross-Scale)
    entanglement_number,
    is_entangled,
    entanglement_matrix,
    # Scale Hierarchy (Fractal Tower)
    ScaleLevel,
    FractalTower,
    build_token_tower,
)

# DEPRECATED: IICA on dense tensors (use SparseHRT instead)
from .deprecated.hrt_iica import (
    HRT_IICA,
    EmbeddedLUT,
    LUTEntry,
    HRTStack,
    CommitInfo,
    create_hrt_iica,
    create_stack,
)

from .hrt_store import (
    HRTPersistentStore,
    MemoryHRTStore,
    FileSystemHRTStore,
    HRTSerializer,
    CommitObject,
    Blob,
    create_memory_store,
    create_file_store,
    create_serializer,
)

# DuckDB Store (preferred backend for ManifoldOS)
from .duckdb_store import (
    DuckDBStore,
    create_duckdb_store,
    create_memory_store as create_duckdb_memory_store,
    create_file_store as create_duckdb_file_store,
)

# DEPRECATED: Use ManifoldOS from mf_os instead
# These imports are wrapped in try/except as the deprecated module has broken imports
try:
    from .deprecated.manifold_os_iica import (
        ManifoldOS_IICA,
        ManifoldIICAConfig,
        create_manifold_iica,
        create_parallel_manifold,
    )
except (ImportError, ModuleNotFoundError):
    # Deprecated module unavailable - define placeholders
    ManifoldOS_IICA = None
    ManifoldIICAConfig = None
    create_manifold_iica = None
    create_parallel_manifold = None

# ManifoldOS (mf_os) - Preferred orchestration layer
from .mf_os import (
    ManifoldOS,
    ManifoldOSConfig,
    TextFilePerceptron,
    create_manifold_os,
    create_memory_manifold,
    create_persistent_manifold,
)

# Device management (from sparse_hrt_3d - consolidated)
from .sparse_hrt_3d import (
    get_device,
    get_default_dtype,
)

# DEPRECATED: 2D Sparse Tensor - use ImmutableSparseTensor3D from sparse_hrt_3d
try:
    from .deprecated.sparse_tensor import (
        ImmutableSparseTensor,
        sparse_zeros,
        sparse_from_dense,
    )
except (ImportError, ModuleNotFoundError):
    ImmutableSparseTensor = None
    sparse_zeros = None
    sparse_from_dense = None

# DEPRECATED: 2D Sparse HRT - use sparse_hrt_3d instead
try:
    from .deprecated.sparse_hrt_2d import (
        SparseHRT,
        SparseHRTConfig,
        SparseAM,
        SparseLattice,
        BasicHLLSet as SparseBasicHLLSet,
        create_sparse_hrt,
    )
except (ImportError, ModuleNotFoundError):
    SparseHRT = None
    SparseHRTConfig = None
    SparseAM = None
    SparseLattice = None
    SparseBasicHLLSet = None
    create_sparse_hrt = None

# GraphBLAS Operations (CPU graph algorithms for lattice processing)
try:
    from .graphblas_ops import (
        GRAPHBLAS_AVAILABLE,
        # Converters
        numpy_to_graphblas,
        am_to_graphblas,
        graphblas_to_am,
        # Core algorithms (work on GraphBLAS Matrix)
        bfs,
        connected_components,
        pagerank,
        transitive_closure,
        shortest_paths,
        multi_source_bfs,
        node_importance,
        # SparseAM3D-aware algorithms
        layer_reachability,
        layer_importance,
        # Lattice Matching & Entanglement
        NodeSignature,
        LatticeMatch,
        compute_node_signatures,
        bss_tau,
        bss_rho,
        bss_symmetric,
        match_nodes_by_degree,
        match_lattices,
        find_entangled_subgraphs,
        compute_entanglement_score,
        # Token Reordering (Post-Disambiguation)
        TokenPath,
        find_all_paths,
        find_shortest_path,
        find_longest_path,
        reorder_tokens,
        rank_paths_by_ngram,
    )
except ImportError:
    GRAPHBLAS_AVAILABLE = False
    numpy_to_graphblas = None
    am_to_graphblas = None
    graphblas_to_am = None
    bfs = None
    connected_components = None
    pagerank = None
    transitive_closure = None
    shortest_paths = None
    multi_source_bfs = None
    node_importance = None
    layer_reachability = None
    layer_importance = None
    NodeSignature = None
    LatticeMatch = None
    compute_node_signatures = None
    bss_tau = None
    bss_rho = None
    bss_symmetric = None
    match_nodes_by_degree = None
    match_lattices = None
    find_entangled_subgraphs = None
    compute_entanglement_score = None
    TokenPath = None
    find_all_paths = None
    find_shortest_path = None
    find_longest_path = None
    reorder_tokens = None
    rank_paths_by_ngram = None

# Sparse 3D Architecture (N-gram layered AM) - via mf_algebra (separation of concerns)
# All HRT classes are now exported through mf_algebra for consistent layering

from .constants import (
    P_BITS,
    SHARED_SEED,
    HASH_FUNC,
    DEFAULT_TAU,
    DEFAULT_RHO,
    DEFAULT_N_TOKEN_SIZES,
    MIN_OVERLAP,
    MAX_CHAIN_LENGTH,
    ENTANGLEMENT_THRESHOLD,
)

# Manifold Algebra (mf_algebra) - Unified Processing Model
# Includes HRT classes for separation of concerns
from .mf_algebra import (
    # Core HRT Classes (canonical implementation)
    SparseHRT3D,
    Sparse3DConfig,
    SparseAM3D,
    SparseLattice3D,
    ImmutableSparseTensor3D,
    BasicHLLSet3D,
    Edge3D,
    create_sparse_hrt_3d,
    
    # Universal ID
    UniversalID,
    content_to_index,
    
    # Sparse Matrices (algebra wrappers)
    SparseMatrix,
    Sparse3DMatrix,
    
    # Projection
    project_layer,
    project_rows,
    project_cols,
    project_submatrix,
    
    # Transform
    transpose,
    transpose_3d,
    normalize_rows,
    normalize_3d,
    scale,
    
    # Filter
    filter_threshold,
    filter_predicate,
    
    # Composition
    merge_add,
    merge_max,
    compose_chain,
    merge_3d_add,
    
    # Path
    reachable_from,
    path_closure,
    
    # Lift/Lower
    lift_to_layer,
    lower_aggregate,
    
    # Cross-structure
    am_to_w,
    w_to_am,
    
    # LUT & Tokens
    START,
    END,
    LookupTable,
    tokenize,
    generate_ntokens,
    
    # Unified Processing
    ProcessingResult,
    input_to_hllset,
    build_sub_hrt,
    extend_with_context,
    merge_hrt,
    unified_process,
    build_w_from_am,
    
    # Fractal D/R/N Evolution (Perceptron Swarm Entanglement)
    DRN_CONVERGENCE_THRESHOLD,
    DRNDecomposition,
    FractalW,
    TemporalDRN,
    LayerHLLSets,
)

# Sign Tokenizer (Uninflected Sign Systems)
from .sign_tokenizer import (
    tokenize_signs,
    generate_sign_ngrams,
    ngram_to_string,
    build_cover_from_rows,
    build_cover_from_cols,
    build_cover_from_w,
    query_signs,
)

__all__ = [
    # HLLSet
    'HLLSet',
    'compute_sha1',
    
    # Immutable Tensor
    'ImmutableTensor',
    'TensorEvolution',
    'TensorEvolutionTriple',
    'compute_element_hash',
    'compute_aggregate_hash',
    'compute_structural_hash',
    
    # Kernel
    'Kernel',
    'Operation',
    'record_operation',
    'Morphism',
    'LatticeMorphism',
    'SingularityReport',
    
    # HRT
    'HRT',
    'HRTConfig',
    'HRTEvolution',
    'HRTEvolutionTriple',
    'AdjacencyMatrix',
    'HLLSetLattice',
    'BasicHLLSet',
    'Cover',
    
    # Algebra
    'HLLCatalog',
    'RelAlgebra',
    'QueryResult',
    'ColumnProfile',
    'TableProfile',
    
    # Entanglement (Common Subgraph Extraction)
    'EntanglementFragment',
    'EntanglementSubgraph',
    'ExtendedEntanglement',
    'CommonSubgraphExtractor',
    'extract_entanglement',
    'EntanglementMeasurement',
    'EntanglementMorphism',
    'EntanglementEngine',
    # N-Edge Entanglement (Stochastic)
    'EdgeSignature',
    'NEdgePath',
    'EdgeLUT',
    'NEdgeEntanglement',
    'NEdgeExtractor',
    'compute_nedge_entanglement',
    
    # Fractal Core (Scale Hierarchy)
    'n_tokenize',
    'multi_scale_tokenize',
    'tokens_to_hllsets',
    'overlap',
    'overlap_from_cardinalities',
    'LatticeEdge',
    'extract_chains',
    'chains_to_tokens',
    'entanglement_number',
    'is_entangled',
    'entanglement_matrix',
    'ScaleLevel',
    'FractalTower',
    'build_token_tower',
    
    # IICA Architecture (Current)
    'HRT_IICA',
    'EmbeddedLUT',
    'LUTEntry',
    'HRTStack',
    'CommitInfo',
    'create_hrt_iica',
    'create_stack',
    
    # HRT Persistent Store
    'HRTPersistentStore',
    'MemoryHRTStore',
    'FileSystemHRTStore',
    'HRTSerializer',
    'CommitObject',
    'Blob',
    'create_memory_store',
    'create_file_store',
    'create_serializer',
    
    # DuckDB Store (preferred)
    'DuckDBStore',
    'create_duckdb_store',
    'create_duckdb_memory_store',
    'create_duckdb_file_store',
    
    # ManifoldOS IICA (deprecated - use ManifoldOS)
    'ManifoldOS_IICA',
    'ManifoldIICAConfig',
    'create_manifold_iica',
    'create_parallel_manifold',
    
    # ManifoldOS (mf_os) - Preferred
    'ManifoldOS',
    'ManifoldOSConfig',
    'TextFilePerceptron',
    'create_manifold_os',
    'create_memory_manifold',
    'create_persistent_manifold',
    
    # Sparse GPU Architecture (CUDA COO)
    'ImmutableSparseTensor',
    'get_device',
    'sparse_zeros',
    'sparse_from_dense',
    'SparseHRT',
    'SparseHRTConfig',
    'SparseAM',
    'SparseLattice',
    'SparseBasicHLLSet',
    'create_sparse_hrt',
    
    # Sparse 3D Architecture (N-gram layered)
    'SparseHRT3D',
    'Sparse3DConfig',
    'SparseAM3D',
    'SparseLattice3D',
    'ImmutableSparseTensor3D',
    'BasicHLLSet3D',
    'Edge3D',
    'create_sparse_hrt_3d',
    
    # Constants
    'P_BITS',
    'SHARED_SEED',
    'HASH_FUNC',
    'DEFAULT_TAU',
    'DEFAULT_RHO',
    'DEFAULT_N_TOKEN_SIZES',
    'MIN_OVERLAP',
    'MAX_CHAIN_LENGTH',
    'ENTANGLEMENT_THRESHOLD',
    
    # Manifold Algebra (v0.7) - Unified Processing Model
    'UniversalID',
    'content_to_index',
    'SparseMatrix',
    'Sparse3DMatrix',
    'project_layer',
    'project_rows',
    'project_cols',
    'project_submatrix',
    'transpose',
    'transpose_3d',
    'normalize_rows',
    'normalize_3d',
    'scale',
    'filter_threshold',
    'filter_predicate',
    'merge_add',
    'merge_max',
    'compose_chain',
    'merge_3d_add',
    'reachable_from',
    'path_closure',
    'lift_to_layer',
    'lower_aggregate',
    'am_to_w',
    'w_to_am',
    'START',
    'END',
    'LookupTable',
    'tokenize',
    'generate_ntokens',
    'ProcessingResult',
    'input_to_hllset',
    'build_sub_hrt',
    'extend_with_context',
    'merge_hrt',
    'unified_process',
    'build_w_from_am',
    
    # Fractal D/R/N Evolution (Perceptron Swarm Entanglement)
    'DRN_CONVERGENCE_THRESHOLD',
    'DRNDecomposition',
    'FractalW',
    'TemporalDRN',
    'LayerHLLSets',
    
    # Sign Tokenizer (Uninflected Sign Systems)
    'tokenize_signs',
    'generate_sign_ngrams',
    'ngram_to_string',
    'build_cover_from_rows',
    'build_cover_from_cols',
    'build_cover_from_w',
    'query_signs',
    
    # GraphBLAS Operations (CPU graph algorithms)
    'GRAPHBLAS_AVAILABLE',
    'numpy_to_graphblas',
    'am_to_graphblas',
    'graphblas_to_am',
    'bfs',
    'connected_components',
    'pagerank',
    'transitive_closure',
    'shortest_paths',
    'multi_source_bfs',
    'node_importance',
    'layer_reachability',
    'layer_importance',
    
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
]

__version__ = "0.7.0"  # Manifold Algebra - Unified Processing Model
