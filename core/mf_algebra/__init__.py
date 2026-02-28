"""
Manifold Algebra Package - Modular Implementation.

This package provides the core algebraic operations for the HLLSet-based
fractal manifold system. It is organized into the following modules:

- identifier_schemes: Content → Index mapping schemes
- universal_id: Universal (reg, zeros) addressing
- sparse_matrices: Lightweight sparse matrix wrappers
- operations: Structure-agnostic algebraic operations
- lut: Lookup Table for token recovery
- processing: Unified processing pipeline
- disambiguation: Cascading disambiguation via layer HLLSets
- drn: Fractal D/R/N evolution tracking
- stores: Commit Store and Bounded Evolution Store
- perceptrons: Sense-Process-Act loop components

All public symbols are re-exported from this __init__.py for
backwards compatibility with the original mf_algebra.py module.
"""

# Identifier Schemes
from .identifier_schemes import (
    IdentifierScheme,
    HashIdentifierScheme,
    VocabularyIdentifierScheme,
    HashVocabularyScheme,
    DEFAULT_IDENTIFIER_SCHEME,
    DEFAULT_H_BITS,
)

# Universal ID
from .universal_id import (
    UniversalID,
    content_to_index,
)

# Sparse Matrices (algebra wrappers)
from .sparse_matrices import (
    SparseMatrix,
    Sparse3DMatrix,
)

# Core HRT Classes (re-exported from sparse_hrt_3d for separation of concerns)
# All modules should import these from mf_algebra, not directly from sparse_hrt_3d
from ..sparse_hrt_3d import (
    # Device management
    get_device,
    get_default_dtype,
    # Configuration
    Sparse3DConfig,
    Edge3D,
    BasicHLLSet3D,
    ImmutableSparseTensor3D,
    SparseAM3D,
    SparseLattice3D,
    SparseHRT3D,
    create_sparse_hrt_3d,
)

# Operations
from .operations import (
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
)

# LUT
from .lut import (
    START,
    END,
    LookupTable,
)

# Processing
from .processing import (
    ProcessingResult,
    tokenize,
    generate_ntokens,
    input_to_hllset,
    build_sub_hrt,
    extend_with_context,
    extend_with_intersected_context,
    merge_hrt,
    unified_process,
    build_w_from_am,
)

# Disambiguation
from .disambiguation import (
    LayerHLLSets,
    DisambiguationResult,
    update_layer_hllsets,
    cascading_disambiguate,
    resolve_disambiguation,
)

# DRN (primitives from kernel, higher-level from drn.py)
from ..kernel import (
    DRN_CONVERGENCE_THRESHOLD,
    DRNDecomposition,
)
from .drn import (
    FractalW,
    TemporalDRN,
)

# Stores
from .stores import (
    Commit,
    CommitStore,
    CommitFingerprint,
    FingerprintIndex,
    EvictionRecord,
    StateSnapshot,
    DeltaRecord,
    EvolutionState,
    BoundedEvolutionStore,
    estimate_cardinality,
)

# Perceptrons and Actuators
from .perceptrons import (
    Perceptron,
    PromptPerceptron,
    Actuator,
    ResponseActuator,
    QueryContext,
    ask,
    create_query_context,
)


__all__ = [
    # Identifier Schemes
    'IdentifierScheme',
    'HashIdentifierScheme',
    'VocabularyIdentifierScheme',
    'HashVocabularyScheme',
    'DEFAULT_IDENTIFIER_SCHEME',
    'DEFAULT_H_BITS',
    
    # Universal ID
    'UniversalID',
    'content_to_index',
    
    # Sparse Matrices (algebra wrappers)
    'SparseMatrix',
    'Sparse3DMatrix',
    
    # Device management
    'get_device',
    'get_default_dtype',
    
    # Core HRT Classes (canonical implementation)
    'Sparse3DConfig',
    'Edge3D',
    'BasicHLLSet3D',
    'ImmutableSparseTensor3D',
    'SparseAM3D',
    'SparseLattice3D',
    'SparseHRT3D',
    'create_sparse_hrt_3d',
    
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
    
    # DRN
    'DRN_CONVERGENCE_THRESHOLD',
    'DRNDecomposition',
    'FractalW',
    'TemporalDRN',
    
    # Commit Store
    'Commit',
    'CommitStore',
    
    # Bounded Evolution Store
    'EvictionRecord',
    'DeltaRecord',
    'EvolutionState',
    'StateSnapshot',
    'BoundedEvolutionStore',
    'estimate_cardinality',
    
    # Fingerprint Index (Commit LUT)
    'CommitFingerprint',
    'FingerprintIndex',
    
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
