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

from .manifold_os_iica import (
    ManifoldOS_IICA,
    ManifoldIICAConfig,
    create_manifold_iica,
    create_parallel_manifold,
)

# Sparse GPU Architecture (CUDA COO tensors)
from .sparse_tensor import (
    ImmutableSparseTensor,
    get_device,
    sparse_zeros,
    sparse_from_dense,
)

from .sparse_hrt import (
    SparseHRT,
    SparseHRTConfig,
    SparseAM,
    SparseLattice,
    BasicHLLSet as SparseBasicHLLSet,
    create_sparse_hrt,
)

# Sparse 3D Architecture (N-gram layered AM)
from .sparse_hrt_3d import (
    SparseHRT3D,
    Sparse3DConfig,
    SparseAM3D,
    SparseLattice3D,
    ImmutableSparseTensor3D,
    BasicHLLSet3D,
    Edge3D,
    create_sparse_hrt_3d,
)

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
    
    # ManifoldOS IICA
    'ManifoldOS_IICA',
    'ManifoldIICAConfig',
    'create_manifold_iica',
    'create_parallel_manifold',
    
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
]

__version__ = "0.6.0"  # 3D AM with n-gram layers
