# Fractal Manifold v0.7.0

**Manifold Algebra** - Unified Processing with CRDT Semantics

A semantic memory system where **ingestion = query = processing** through a single algebraic pipeline.

---

## Core Insight

Everything flows through the same pipeline:

```text
INPUT → HLLSet → [Sub-HRT] → Merge → Current'
                     ↑
              ephemeral workspace
              (never persisted)
```

**Ingestion** uses this pipeline. **Query** uses this pipeline. **Any interaction** uses this pipeline.

---

## The Universal Identifier: `(reg, zeros)`

All structures share the same addressing scheme:

```text
content → hash → (reg, zeros) → index
```

This **glues together**:
- **HLLSet**: Register positions
- **AM**: Row/column indices  
- **W**: Transition matrix indices
- **Sheaf sections**: Cross-layer identifiers

Same content → same index → everywhere.

```python
from core.manifold_algebra import UniversalID

uid = UniversalID.from_content("cat sat", layer=0, p_bits=10)
print(uid)  # UniversalID(reg=523, zeros=4, layer=0)

# Same content ALWAYS produces same (reg, zeros)
# This enables idempotent merge
```

### Pluggable Addressing

`(reg, zeros)` is the **default hash-based scheme**. The algebra doesn't care what the atoms are:

| Vocabulary | Addressing |
|------------|-----------|
| Hash-based | `content → SHA1 → (reg, zeros)` |
| Chinese | `character → lookup → (atom_id, context)` |
| Visual | `pattern → codebook → (shape_id, scale)` |
| Musical | `note → vocabulary → (pitch, harmonic)` |

The structures and operations are **co-adaptive** - they evolve together.

---

## Anti-Sets (HLLSet)

**Important!** HLLSet is NOT a HyperLogLog cardinality estimator.

HLLSets are **anti-sets**: they behave like sets but don't store elements.

| Property | Anti-Set Behavior |
|----------|------------------|
| **Absorb** | `H.absorb(token)` - element absorbed into registers |
| **Union** | `A \| B` - full set union |
| **Intersection** | `A & B` - full set intersection |
| **Difference** | `A - B` - set difference |
| **Cardinality** | `H.cardinality()` - estimated count |
| **Elements** | ❌ Not stored - absorbed, not contained |

```python
A = HLLSet()
B = HLLSet()
A.absorb("cat")
B.absorb("dog")

C = A | B           # union
D = A & B           # intersection
E = A - B           # difference
print(C.cardinality())  # ~2
```

---

## CRDT Properties

The unified pipeline guarantees **CRDT semantics**:

```text
┌─────────────────────────────────────────────────────────────────────────┐
│                         CRDT PROPERTIES                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  IDEMPOTENT     merge(A, A) = A                                         │
│                 Same input processed twice → same result                │
│                                                                         │
│  COMMUTATIVE    merge(A, B) = merge(B, A)                               │
│                 Order doesn't matter                                    │
│                                                                         │
│  ASSOCIATIVE    merge(merge(A,B), C) = merge(A, merge(B,C))             │
│                 Grouping doesn't matter                                 │
│                                                                         │
│  ═══════════════════════════════════════════════════════════════════    │
│                                                                         │
│  → EVENTUAL CONSISTENCY guaranteed                                      │
│  → Parallel/distributed processing is SAFE                              │
│  → No coordination needed                                               │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Manifold Algebra

Parametrized operations for structure manipulation (not task-oriented):

### Projection (π)
Extract substructures:
```python
project_layer(AM, n)        # Get layer n
project_rows(M, {1,2,3})    # Get specific rows
project_cols(M, {4,5,6})    # Get specific columns
project_submatrix(M, rows, cols)  # Get block
```

### Transform
Modify structures:
```python
transpose(M)                # Flip rows ↔ columns
transpose_3d(AM)            # Transpose all layers (backpropagation)
normalize_rows(M)           # Row stochastic (AM → W)
scale(M, α)                 # Multiply all values
```

### Filter (σ)
Conditional selection:
```python
filter_threshold(M, min_val=0.1)    # Keep entries ≥ threshold
filter_predicate(M, lambda r,c,v: v > 1.0)  # Custom predicate
```

### Composition
Combine structures:
```python
merge_add(A, B)             # A + B (element-wise)
merge_max(A, B)             # max(A, B) (element-wise)
compose_chain(A, B)         # A ∘ B (matrix multiply, path composition)
```

### Path
Graph operations:
```python
reachable_from(M, {start}, hops=2)  # Nodes reachable in ≤2 hops
path_closure(M, max_hops=3)         # Transitive closure (M*)
```

### Lift/Lower
Move between layers:
```python
lift_to_layer(M_2d, target_layer)   # 2D → 3D at specific layer
lower_aggregate(AM, agg='sum')      # 3D → 2D by aggregation
```

### Cross-Structure
Convert between representations:
```python
W = am_to_w(AM)             # Adjacency → Transition (normalize rows)
AM = w_to_am(W)             # Transition → Adjacency (denormalize)
```

---

## Unified Processing

```python
from core.manifold_algebra import unified_process, build_w_from_am

# Process ANY input (ingestion OR query - same code!)
result = unified_process(
    text,           # Input text
    current_hrt,    # Current HRT state
    current_W,      # Current transition matrix
    config,         # System config
    lut,            # Lookup table
)

# Result contains:
#   result.input_hllset   - HLLSet of input
#   result.sub_hrt        - Ephemeral workspace (discarded)
#   result.context_edges  - Extended context
#   result.merged_hrt     - New current state

# Update W from merged AM
new_W = build_w_from_am(result.merged_hrt.am, config)
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
│  │                     SparseAM3D (Adjacency Matrix)               │    │
│  │                                                                 │    │
│  │   Layer 0 (1-grams)    Layer 1 (2-grams)    Layer 2 (3-grams)   │    │
│  │   ┌────────────────┐   ┌────────────────┐   ┌────────────────┐  │    │
│  │   │  AM[0,:,:]     │   │  AM[1,:,:]     │   │  AM[2,:,:]     │  │    │
│  │   │  sparse COO    │   │  sparse COO    │   │  sparse COO    │  │    │
│  │   └────────────────┘   └────────────────┘   └────────────────┘  │    │
│  │                                                                 │    │
│  │   ImmutableSparseTensor3D: indices[3, nnz] + values[nnz]        │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                     W (Transition Matrix)                       │    │
│  │                                                                 │    │
│  │   W = normalize_rows(AM)                                        │    │
│  │   Each row sums to 1.0 → transition probabilities               │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                     LookupTable (LUT)                           │    │
│  │                                                                 │    │
│  │   (reg, zeros) → n-token (for retrieval)                        │    │
│  │   n-token → (reg, zeros) (for ingestion)                        │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### N-Token Model

```text
"The cat sat" → sliding window with START/END markers:

(START) → (the) → (the,cat) → (the,cat,sat) → (cat) → (cat,sat) → (sat) → (END)
   ↓        ↓          ↓             ↓           ↓          ↓         ↓       ↓
 Layer 0  Layer 0   Layer 1      Layer 2     Layer 0   Layer 1    Layer 0  Layer 0
```

Each n-token maps to `(reg, zeros)` via hash. Same n-token → same position → idempotent.

---

## Memory Efficiency

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
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Quick Start

```python
from core import (
    SparseHRT3D, Sparse3DConfig, SparseAM3D, SparseLattice3D,
    get_device, __version__
)
from core.manifold_algebra import (
    LookupTable, START, END,
    unified_process, build_w_from_am,
    project_layer, transpose_3d, am_to_w
)

print(f"Fractal Manifold v{__version__}")

# Configuration
config = Sparse3DConfig(p_bits=10, h_bits=32, max_n=3)
lut = LookupTable(config=config)
lut.add_ntoken(START)
lut.add_ntoken(END)

# Initialize empty HRT
empty_am = SparseAM3D.from_edges(config, [])
empty_lattice = SparseLattice3D.from_sparse_am(empty_am)
current_hrt = SparseHRT3D(am=empty_am, lattice=empty_lattice, config=config, lut=frozenset(), step=0)
current_W = {}

# Ingest corpus (uses unified pipeline)
corpus = ["The cat sat on the mat", "The dog ran in the park"]
for text in corpus:
    result = unified_process(text, current_hrt, current_W, config, lut)
    current_hrt = result.merged_hrt
    current_W = build_w_from_am(current_hrt.am, config)

print(f"HRT edges: {current_hrt.nnz}")
print(f"LUT entries: {len(lut.ntoken_to_index)}")

# Query (same pipeline!)
query = "The cat ran"
result = unified_process(query, current_hrt, current_W, config, lut)
print(f"Query added {result.merged_hrt.nnz - current_hrt.nnz} new edges")
```

---

## Module Structure

```text
core/
├── __init__.py           # Exports, version 0.7.0
├── manifold_algebra.py   # UniversalID, algebraic ops, unified pipeline ← NEW
├── sparse_hrt_3d.py      # SparseHRT3D, SparseAM3D, SparseLattice3D
├── sparse_tensor.py      # ImmutableSparseTensor (CUDA COO)
├── hllset.py             # HLLSet anti-set structures
├── constants.py          # Configuration constants
├── kernel.py             # Kernel operations
└── deprecated/           # Dense implementations (moved)
```

---

## Version History

| Version | Milestone |
|---------|-----------|
| **0.7.0** | **Manifold Algebra** - Unified processing, CRDT semantics, `(reg, zeros)` universal ID |
| 0.6.0 | 3D Sparse HRT - N-gram layered AM |
| 0.5.0 | Sparse GPU Architecture (CUDA COO) |
| 0.4.0 | IICA Architecture |

---

## Design Principles

1. **Unified Pipeline**: Ingestion = Query = Processing
2. **Sub-HRT is Ephemeral**: Workspace for merge, never persisted
3. **CRDT Semantics**: Idempotent, commutative, associative → eventual consistency
4. **Content Addressability**: `(reg, zeros)` is the universal glue
5. **Co-Adaptive**: Structures and operations evolve together
6. **Anti-Sets**: HLLSets support full set algebra (not just cardinality)

---

## License

See [LICENSE](LICENSE)

## References

1. [Solomonoff prior](https://medium.com/@swarnenduiitb2020/the-man-who-solved-learning-in-1964-and-why-we-ignored-him-for-60-years-602a23ddf956)
2. **CRDTs**: Conflict-free Replicated Data Types (Shapiro et al.)
3. **Karoubi Envelope**: Idempotent completion of a category
4. **HyperLogLog**: Probabilistic data structure (Flajolet et al.)
