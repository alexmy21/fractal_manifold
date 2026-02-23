"""
DEPRECATED: Dense HRT Implementation

These modules use dense tensors (4GB per 32K×32K HRT).
With 3 HRTs, they exhaust 64GB RAM.

Use the sparse GPU architecture instead:
- SparseHRT: 2MB per 100K edges (2000x smaller)
- CUDA acceleration on GPU
- Same IICA properties

Migration:
    # OLD (dense)
    from core import HRT, HRTConfig, HRT_IICA
    
    # NEW (sparse)
    from core import SparseHRT, SparseHRTConfig, create_sparse_hrt

These modules will be removed in v1.0.
"""

# Re-export hllset from parent
from ..hllset import HLLSet, compute_sha1

# For backward compatibility, re-export from this package
from .immutable_tensor import (
    ImmutableTensor,
    TensorEvolution,
    TensorEvolutionTriple,
    compute_element_hash,
    compute_aggregate_hash,
    compute_structural_hash,
)

from .hrt import (
    HRT,
    HRTConfig,
    HRTEvolution,
    HRTEvolutionTriple,
    AdjacencyMatrix,
    HLLSetLattice,
    BasicHLLSet,
    Cover,
)

from .hrt_iica import (
    HRT_IICA,
    EmbeddedLUT,
    LUTEntry,
    HRTStack,
    CommitInfo,
    create_hrt_iica,
    create_stack,
)
