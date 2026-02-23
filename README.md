# Fractal Manifold v0.6.0

**3D Sparse HRT** - N-Gram Layered Adjacency Matrix with Karoubi Envelope Structure

A semantic memory system built on Hash Relational Tensors with GPU-accelerated sparse representation.

---

## Introduction

Fractal Manifold is a **content-addressable semantic memory** implementing a computational **Karoubi idempotent category** with explicit splits. The system stores and retrieves structured knowledge using probabilistic data structures (HLLSets - HyperLogLog Sets), sparse GPU tensors (CUDA COO), and a novel 3D adjacency matrix that separates n-gram orders.

**Important!** Despite its name, HLLSet is not a HyperLogLog cardinality estimator, which is based on Flajolet-Martin algorithms. HLLSets support all set operations making it ideal for building a computational implementation of the **Karoubi envelope** (idempotent completion).

The key distinction:

| HyperLogLog (Flajolet-Martin) | HLLSet |
| --------- | ---- |
| Cardinality estimator only | Full set operations |
| Answers: "How many unique elements?" | Answers: union, intersection, difference, etc. |
| Single-purpose | Algebraic structure |
| — | Enables Karoubi envelope |

The naming honors the computational breakthrough (probabilistic hashing, register-based structure), but HLLSet is a proper set with operations—which is exactly what makes it the right foundation for the idempotent category:

- merge(A, B) = set union (idempotent: merge(A, A) = A)
- intersect(A, B) = set intersection (for edge weights)
- Content-addressable (hash-based identity)

### The Elegant Core Insight

The 3D dimension literally **adds structure** rather than complexity—the n-gram order becomes **topology** rather than metadata.

```text
┌─────────────────────────────────────────────────────────────────────────┐
│                      3D ADJACENCY MATRIX                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│    AM[n, row, col] = |BasicHLLSet(row) ∩ BasicHLLSet(col)|              │
│                                                                         │
│    n = 0: 1-grams    "the", "cat", "sat"                                │
│    n = 1: 2-grams    "the cat", "cat sat"                               │
│    n = 2: 3-grams    "the cat sat"                                      │
│                                                                         │
│    Each layer is SEPARATE → no n-gram mixing                            │
│    Edge weight = CONTEXT COVARIANCE (not term frequency!)               │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Karoubi Envelope Structure

The 3D HRT is a computational implementation of the **Karoubi envelope** (idempotent completion), where all idempotents split explicitly.

### Categorical Foundation

```text
┌─────────────────────────────────────────────────────────────────────────┐
│                    KAROUBI ENVELOPE kar(C)                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  For each idempotent e: A → A where e ∘ e = e, find the split:          │
│                                                                         │
│    • Object B (the "image")                                             │
│    • Projection p: A → B                                                │
│    • Inclusion i: B → A                                                 │
│    • p ∘ i = id_B                                                       │
│    • i ∘ p = e                                                          │
│                                                                         │
│  IMPLEMENTATION MAPPING                                                 │
│  ─────────────────────                                                  │
│                                                                         │
│    Karoubi Concept        │  3D HRT Implementation                      │
│    ──────────────────     │  ──────────────────────                     │
│    Idempotent e           │  merge(A, A) = A                            │
│    Split objects          │  N-gram layers AM[n,:,:]                    │
│    Projection p           │  layer_edges(n)                             │
│    Inclusion i            │  with_ngram_edge(n, ...)                    │
│    Image objects          │  BasicHLLSet3D                              │
│                                                                         │
│  THE EXPLICIT SPLIT                                                     │
│  ──────────────────                                                     │
│                                                                         │
│    AM (mixed) ───p───▶ AM[n] (layer) ───i───▶ AM                      │
│             └──────────── e = i ∘ p ────────────┘                       │
│                           (idempotent)                                  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### BasicHLLSets as Retracts

```text
┌─────────────────────────────────────────────────────────────────────────┐
│                    PULLBACK AS EDGE WEIGHT                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  basic_hllsets_for_row(i)  =  projection onto row's "image"             │
│  basic_hllsets_for_col(j)  =  projection onto col's "image"             │
│                                                                         │
│  Edge weight = |row_image ∩ col_image| = SIZE OF PULLBACK OBJECT        │
│                                                                         │
│  This makes AM[n,i,j] a CATEGORICAL INVARIANT measuring the             │
│  pullback in the Karoubi category!                                      │
│                                                                         │
│    Row context ────────┐                                                │
│                        ▼                                                │
│                   ┌─────────┐                                           │
│                   │ Pullback│ ← size = edge weight                      │
│                   └─────────┘                                           │
│                        ▲                                                │
│    Col context ────────┘                                                │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## IICA Principles

All operations maintain **IICA** (Immutability, Idempotence, Content Addressability):

```text
┌─────────────────────────────────────────────────────────────────────────┐
│                           IICA PRINCIPLES                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  IMMUTABILITY         Every operation returns a NEW object              │
│  ─────────────        Original state is never modified                  │
│                       Enables safe parallelism & time-travel            │
│                                                                         │
│  IDEMPOTENCE          merge(A, A) == A                                  │
│  ───────────          Repeated operations produce same result           │
│                       Ensures well-behaved idempotent category          │
│                                                                         │
│  CONTENT              Object identity = hash of content                 │
│  ADDRESSABILITY       Same content → same name → deduplication          │
│  ──────────────       All splits are CANONICAL (content-addressed)      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Architecture

### 3D Sparse HRT

```text
┌─────────────────────────────────────────────────────────────────────────┐
│                         SparseHRT3D                                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                     SparseAM3D                                  │    │
│  │                                                                 │    │
│  │   Layer 0 (1-grams)    Layer 1 (2-grams)    Layer 2 (3-grams)   │    │
│  │   ┌────────────────┐   ┌────────────────┐   ┌────────────────┐  │    │
│  │   │  AM[0,:,:]     │   │  AM[1,:,:]     │   │  AM[2,:,:]     │  │    │
│  │   │  32K × 32K     │   │  32K × 32K     │   │  32K × 32K     │  │    │
│  │   │  (sparse COO)  │   │  (sparse COO)  │   │  (sparse COO)  │  │    │
│  │   └────────────────┘   └────────────────┘   └────────────────┘  │    │
│  │                                                                 │    │
│  │   ImmutableSparseTensor3D: indices[3, nnz] + values[nnz]        │    │
│  │   CUDA GPU accelerated, content-addressed (SHA1)                │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                   SparseLattice3D                               │    │
│  │                                                                 │    │
│  │   Per-layer connections:    layer_row_connections(n, row)       │    │
│  │                             layer_col_connections(n, col)       │    │
│  │                                                                 │    │
│  │   Aggregated (for BasicHLLSet): all_row_connections(row)        │    │
│  │                                 all_col_connections(col)        │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                   BasicHLLSet3D                                 │    │
│  │                                                                 │    │
│  │   (n, reg, zeros) - includes n-gram layer!                      │    │
│  │                                                                 │    │
│  │   basic_hllsets_for_row(i) → List[BasicHLLSet3D]                │    │
│  │   basic_hllsets_for_col(j) → List[BasicHLLSet3D]                │    │
│  │   compute_edge_weight(i, j) → |row_basics ∩ col_basics|         │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │   LUT (Lookup Table)                                            │    │
│  │   FrozenSet of entries for token disambiguation                 │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Memory Efficiency

```text
┌─────────────────────────────────────────────────────────────────────────┐
│                    SPARSE vs DENSE COMPARISON                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   Dense 3D AM (3 × 32K × 32K × float32):          12.0 GB               │
│   Sparse 3D AM (100K edges × 28 bytes):            2.7 MB               │
│                                                                         │
│   Memory savings:  ~4,500× smaller!                                     │
│                                                                         │
│   3 Dense HRTs:  36 GB  →  Exhausts 64GB laptop RAM!                    │
│   3 Sparse HRTs:  8 MB  →  Easily fits in GPU VRAM!                     │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Core Concepts

### BasicHLLSet3D

The atomic unit: `(n, reg, zeros)` triple.

```python
BasicHLLSet3D(n=0, reg=42, zeros=7)   # 1-gram at position (42, 7)
BasicHLLSet3D(n=1, reg=42, zeros=7)   # 2-gram at position (42, 7) - DIFFERENT LAYER!
```

Same `(reg, zeros)` but different `n` → different positions in 3D AM.

### Edge3D

Each edge includes the n-gram layer:

```python
Edge3D(n=0, row=100, col=200, value=5.0)  # 1-gram edge
Edge3D(n=1, row=100, col=201, value=3.0)  # 2-gram edge
Edge3D(n=2, row=100, col=202, value=7.0)  # 3-gram edge
```

### Context Covariance (not TF!)

```text
AM[n, i, j] = |BasicHLLSet(row_i) ∩ BasicHLLSet(col_j)|
              ─────────────────────────────────────────
                        CONTEXT COVARIANCE

• High value = positions share many connections = related contexts
• Low value = positions share few connections = unrelated contexts
• This is SEMANTIC PROXIMITY, not term frequency!
```

### Sliding Window Algorithm

```text
Token processing: 1-gram → 2-gram → 3-gram → shift → repeat

"The quick brown fox jumps"

Step 1: "The"                   → Layer 0
Step 2: "The quick"             → Layer 1
Step 3: "The quick brown"       → Layer 2
Step 4: shift, "quick"          → Layer 0
Step 5: "quick brown"           → Layer 1
Step 6: "quick brown fox"       → Layer 2
...
```

---

## Quick Start

```python
from core import (
    SparseHRT3D,
    Sparse3DConfig,
    SparseAM3D,
    SparseLattice3D,
    Edge3D,
    create_sparse_hrt_3d,
    get_device,
    __version__
)

print(f"Fractal Manifold v{__version__}")
print(f"Device: {get_device()}")

# Create 3D HRT
hrt = create_sparse_hrt_3d(p_bits=10, h_bits=32, max_n=3)
print(f"Shape: {hrt.shape}")  # (3, 32770, 32770)

# Add edges at different n-gram layers
hrt1 = hrt.with_ngram_edge(1, 100, 200, 1.0)   # 1-gram
hrt2 = hrt1.with_ngram_edge(2, 100, 201, 2.0)  # 2-gram
hrt3 = hrt2.with_ngram_edge(3, 100, 202, 3.0)  # 3-gram

print(f"Layer stats: {hrt3.layer_stats()}")
# {0: 1, 1: 1, 2: 1}

# BasicHLLSets for BOTH rows and columns
row_basics = hrt3.basic_hllsets_for_row(100)
col_basics = hrt3.basic_hllsets_for_col(200)
print(f"Row 100: {row_basics}")
print(f"Col 200: {col_basics}")

# Compute edge weight (context covariance)
weight = hrt3.compute_edge_weight(100, 200)
print(f"Edge weight = |row ∩ col| = {weight}")

# Merge (idempotent)
merged = hrt3.merge(hrt3)
assert merged.am.name == hrt3.am.name  # merge(A, A) = A ✓
```

---

## Module Structure

```text
core/
├── __init__.py           # Exports, version 0.6.0
├── sparse_hrt_3d.py      # SparseHRT3D, SparseAM3D, SparseLattice3D ← NEW
├── sparse_tensor.py      # ImmutableSparseTensor (2D, CUDA COO)
├── sparse_hrt.py         # SparseHRT (2D, kept for compatibility)
├── hllset.py             # HLLSet probabilistic structures
├── constants.py          # Configuration constants
├── kernel.py             # Kernel (merge operations)
├── algebra.py            # Algebraic operations
└── deprecated/           # Dense implementations (moved)
    ├── hrt.py            # Dense HRT (4GB per HRT!)
    ├── hrt_iica.py       # Dense IICA
    └── immutable_tensor.py
```

---

## Version History

| Version | Milestone |
| --------- | ----------- |
| **0.6.0** | **3D Sparse HRT** - N-gram layered AM, Karoubi envelope structure |
| 0.5.1 | Dense HRT moved to deprecated/ |
| 0.5.0 | Sparse GPU Architecture (CUDA COO) |
| 0.4.0 | IICA Architecture - Immutability, Idempotence, Content Addressability |

---

## References

- **Karoubi Envelope**: Idempotent completion of a category, where all idempotents split
- **HyperLogLog**: Probabilistic cardinality estimation (Flajolet et al.)
- **COO Format**: Coordinate sparse tensor representation
- **CUDA**: GPU acceleration for sparse tensor operations

---

## License

See [LICENSE](LICENSE)
