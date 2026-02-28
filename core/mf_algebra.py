"""
Manifold Algebra - Unified Processing Model

BACKWARDS COMPATIBILITY MODULE
==============================

This module re-exports all symbols from the mf_algebra/ package.
The implementation has been split into separate modules for better
maintainability:

- mf_algebra/identifier_schemes.py  - Content → Index mapping
- mf_algebra/universal_id.py        - Universal (reg, zeros) addressing
- mf_algebra/sparse_matrices.py     - Lightweight sparse wrappers
- mf_algebra/operations.py          - Structure-agnostic operations
- mf_algebra/lut.py                 - Lookup Table
- mf_algebra/processing.py          - Unified processing pipeline
- mf_algebra/disambiguation.py      - Cascading disambiguation
- mf_algebra/drn.py                 - D/R/N evolution tracking
- mf_algebra/stores.py              - Commit & Evolution stores
- mf_algebra/perceptrons.py         - Sense-Process-Act components

Original API Preserved
======================

All original imports continue to work:

    from core.mf_algebra import UniversalID, LookupTable, ask
    from core.mf_algebra import project_layer, merge_add

Or use the package directly:

    from core.mf_algebra.processing import unified_process
    from core.mf_algebra.stores import BoundedEvolutionStore

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

# Re-export everything from the mf_algebra package
from .mf_algebra import (
    # Identifier Schemes
    IdentifierScheme,
    HashIdentifierScheme,
    VocabularyIdentifierScheme,
    HashVocabularyScheme,
    DEFAULT_IDENTIFIER_SCHEME,
    DEFAULT_H_BITS,
    
    # Universal ID
    UniversalID,
    content_to_index,
    
    # Sparse Matrices
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
    
    # LUT
    START,
    END,
    LookupTable,
    
    # Unified Processing
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
    
    # Cascading Disambiguation
    LayerHLLSets,
    DisambiguationResult,
    update_layer_hllsets,
    cascading_disambiguate,
    resolve_disambiguation,
    
    # DRN
    DRN_CONVERGENCE_THRESHOLD,
    DRNDecomposition,
    FractalW,
    TemporalDRN,
    
    # Commit Store
    Commit,
    CommitStore,
    
    # Bounded Evolution Store
    EvictionRecord,
    DeltaRecord,
    EvolutionState,
    StateSnapshot,
    BoundedEvolutionStore,
    estimate_cardinality,
    
    # Fingerprint Index (Commit LUT)
    CommitFingerprint,
    FingerprintIndex,
    
    # Perceptrons
    Perceptron,
    PromptPerceptron,
    
    # Actuators
    Actuator,
    ResponseActuator,
    
    # Query Interface
    QueryContext,
    ask,
    create_query_context,
)


# Re-export __all__ from the package
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
