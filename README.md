# Fractal Manifold v0.7.0

**Manifold Algebra** - Unified Processing with CRDT (Conflict-free Replicated Data Types) Semantics

A semantic memory system where **ingestion = query = processing** through a single algebraic pipeline.

---

## Anti-Sets (HLLSet)

HLLSet (HyperLogLog Set) is the building block for everything in Fractal Manifold.

**Important!** HLLSet is NOT a HyperLogLog cardinality estimator.

HLLSets are **anti-sets**: they behave like sets but don't store elements.

| Property | Anti-Set Behavior |
| ---------- | ------------------ |
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

### Register Format (uint32 Bitmap)

HLLSet uses a **bitmap register format** (`uint32`), not traditional HLL max-zeros. Each token that we are pushing into HLLSet converted in the pair (reg, zeros), where reg is register's index calculated from P leading bits of the hash 32-bit integer; and zeros - is the bit position that is number of trailing zeros in the token hash:

```python
from core.hllset import HLLSet, DEFAULT_HASH_CONFIG, REGISTER_DTYPE

# Hash configuration is centralized
print(DEFAULT_HASH_CONFIG)
# HashConfig(hash_type=MURMUR3, p_bits=10, seed=42, h_bits=64)

# Register format
print(REGISTER_DTYPE)  # numpy.uint32

# Each register is a 32-bit BITMAP where bit k is set
# when an element with k trailing zeros was observed
registers = hllset.dump_numpy()  # shape: (2^p_bits,), dtype: uint32

# Set operations use bitwise operations:
# - Union: registers_a | registers_b
# - Intersection: registers_a & registers_b
```

This format enables full set algebra (union, intersection, difference) at the register level.

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
from core.mf_algebra import UniversalID

uid = UniversalID.from_content("cat sat", layer=0)
print(uid)  # UniversalID(reg=523, zeros=4, layer=0)

# Same content ALWAYS produces same (reg, zeros)
# This enables idempotent merge
```

### Pluggable Addressing

`(reg, zeros)` is the **default hash-based scheme**. The algebra doesn't care what the atoms are:

| Vocabulary | Addressing |
| ------------ | ----------- |
| Hash-based | `content → MurmurHash64A → (reg, zeros)` |
| Chinese | `character → lookup → (atom_id, context)` |
| Visual | `pattern → codebook → (shape_id, scale)` |
| Musical | `note → vocabulary → (pitch, harmonic)` |

### Identifier Schemes

Three schemes are implemented:

**HashIdentifierScheme** (default):

```python
from core.mf_algebra import HashIdentifierScheme

scheme = HashIdentifierScheme(p_bits=10, h_bits=32)
idx = scheme.to_index("neural network")  # hash → (reg, zeros) → index
```

For inflected languages where vocabulary is open-ended. Handles typos, variations, and novel words naturally.

**VocabularyIdentifierScheme** (sequential indices):

```python
from core.mf_algebra import VocabularyIdentifierScheme

# Load Chinese vocabulary (~80K hieroglyphs)
scheme = VocabularyIdentifierScheme.from_file("chinese_vocab.txt")
idx = scheme.to_index("你")  # Direct lookup → sequential index (0, 1, 2...)

# Or build programmatically
scheme = VocabularyIdentifierScheme()
scheme.add_sign("你")  # 0
scheme.add_sign("好")  # 1
```

For uninflected languages with fixed sign systems. Uses sequential IDs.

**HashVocabularyScheme** (for compact vocabularies with exact addressing):

```python
from core.mf_algebra import HashVocabularyScheme

# sign → hash → index (full 32-bit hash, no compression)
scheme = HashVocabularyScheme()
scheme.add_sign("你")  # Full hash = exact index
scheme.add_sign("好")

# Each sign gets unique row/column in AM/W
# Union across row/column = context for THIS SPECIFIC SIGN
idx = scheme.to_index("你")  # 32-bit hash as index

# Get (reg, zeros) when needed for HLLSet operations
reg, zeros = scheme.to_reg_zeros("你")

# Reverse lookup
sign = scheme.get_sign(idx)  # "你"

# Analyze collisions (should be ~0 for 80K vocabulary)
report = scheme.collision_report()
```

**Comparing the schemes:**

| Scheme | Pipeline | Index Range | AM/W Precision |
| -------- | ---------- | ------------- | ---------------- |
| `HashIdentifierScheme` | token → MurmurHash64A → (reg,zeros) → index | ~32K | Cloud (compression) |
| `HashVocabularyScheme` | sign → hash → index | ~4B | **Exact** (per-sign) |
| `VocabularyIdentifierScheme` | sign → sequential lookup | vocab size | Exact but arbitrary |

**When to use each:**

- **HashIdentifierScheme** (default): Open vocabularies, HLLSet-native, probabilistic OK
- **HashVocabularyScheme**: Compact vocabularies (Chinese ~80K), exact context unions needed
- **VocabularyIdentifierScheme**: When you need specific index assignments

**Vocabulary as HLLSet Fingerprint:**

We can process Vocabulary as a normal input interpreting each sign in the vocabulary as a token. Result - HLLSet that could be used as a fingerprint of vocabulary and just because it's HLLSet you can compare given vocabulary with any other, you can monitor changes in vocabulary during the time and do it just by applying normal set operations: union, intersection, difference and so on.

```python
# Vocabulary grows lazily
scheme = HashVocabularyScheme()
scheme.add_sign("你")
scheme.add_sign("好")
# ... signs added over time

# Get vocabulary fingerprint (O(1) per comparison)
hllset = scheme.to_hllset()

# Compare vocabularies across installations
other_scheme = HashVocabularyScheme.from_file("other_user.txt")
similarity = scheme.tau(other_scheme)  # 0.0 to 1.0

# Social profiling: what vocabulary reveals
jaccard = scheme.jaccard(other_scheme)  # Domain overlap
shared = scheme.vocabulary_intersection(other_scheme)  # Exact overlap
unique = scheme.vocabulary_diff(other_scheme)  # What I have they don't

# Merge vocabularies from different sources
combined = scheme.merge_vocabulary(other_scheme)
```

**$\text{HVS} ↔ \text{HIS}$ Interoperability:**

With same hash function, p_bits, and seed, HVS and HIS are compatible:

```python
# HVS stores both representations
hvs_index = scheme.to_index("你")      # Full hash (precise)
his_index = scheme.to_his_index("你")  # (reg, zeros) encoded (compressed)
reg, zeros = scheme.to_reg_zeros("你") # Raw (reg, zeros)

# Projection map: HVS → HIS (many-to-one, lossy)
hvs_to_his = scheme.hvs_to_his_index_map()  # {hvs_idx: his_idx, ...}

# Reverse map: HIS → HVS (one-to-many, disambiguation set)
his_to_hvs = scheme.his_to_hvs_index_map()  # {his_idx: {hvs_idx1, hvs_idx2}, ...}

# Signs at a given HIS index (disambiguation within vocabulary)
signs = scheme.signs_at_his_index(his_index)  # ["你", ...]

# Project AM from HVS to HIS (merges edges)
hvs_edges = [(0, hvs_idx1, hvs_idx2, 1.0), ...]
his_edges = scheme.project_am_to_his(hvs_edges)

# Create compatible HIS scheme
his_scheme = scheme.to_his_scheme()  # Same p_bits, h_bits
```

This enables **mixing precise and compressed data**:

- Use HVS for exact AM/W row/column addressing
- Project to HIS for HLLSet operations
- Same sign → same (reg, zeros) in both schemes

**Using schemes with LookupTable:**

```python
from core.mf_algebra import LookupTable, HashVocabularyScheme

vocab = HashVocabularyScheme.from_file("chinese.txt")
lut = LookupTable(config, identifier_scheme=vocab)
```

The structures and operations are **co-adaptive** - they evolve together.

---

## HLLSet as Context

**Critical insight**: HLLSet IS context, not a container of elements.

### Basic Context

An HLLSet built from co-occurring elements represents **local context** — elements assumed related by their co-occurrence:

```python
C = HLLSet.from_batch(["neural", "network", "learning"])
# C is the CONTEXT where these tokens appeared together
```

### Compound Context

The union of all HLLSets sharing an element is **complete context**:

```bush
Context(t) = ⋃ { Cᵢ : t ∈ Cᵢ }
```

**But here's the key**: compound context is ALSO an HLLSet. You cannot distinguish between:

- A basic HLLSet from one document
- A compound HLLSet merged from thousands

This is a **feature, not a limitation**:

| Property | Implication |
| ---------- | ------------ |
| **Indistinguishability** | No artificial boundaries between "local" and "global" context |
| **Composition** | Accidental co-occurrence → Compositional certainty via merge |
| **Never tokens, always context** | We don't ask "what is token X?" but "what contexts contain X with measurable certainty?" |

### Measurable Certainty

Token X alone is meaningless. Token X in its accumulated context has measurable relationships:

```python
# How related are "neural" and "deep" across all contexts?
context_neural = get_accumulated_context("neural")
context_deep = get_accumulated_context("deep")

similarity = context_neural.similarity(context_deep)  # 0.0 to 1.0
# High similarity = strong compositional relationship
```

The more contexts merge, the more **accidental noise averages out** and **true relationships emerge**.

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
from core.mf_algebra import unified_process, build_w_from_am

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

### 3D Sparse HRT (Hash Relational Tensor)

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

Some vocabularies can be very limited (for example, the whole Chinese vocabulary is only 80K hieroglyphs). To increase vocabulary size we are applying bootstrapping with an approach similar to n-grams (we call them n-tokens):

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
from core.mf_algebra import (
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
├── mf_algebra.py         # UniversalID, algebraic ops, unified pipeline
├── mf_os.py              # ManifoldOS orchestration layer
├── duckdb_store.py       # DuckDB-backed persistence
├── sparse_hrt_3d.py      # SparseHRT3D, SparseAM3D, SparseLattice3D
├── sparse_tensor.py      # ImmutableSparseTensor (CUDA COO)
├── hllset.py             # HLLSet anti-set structures (MurmurHash64A, uint32 bitmap)
├── sign_tokenizer.py     # Uninflected sign system tokenization
├── constants.py          # Configuration constants
├── kernel.py             # Kernel operations
└── deprecated/           # Dense implementations (moved)
```

---

## Version History

| Version | Milestone |
| --------- | ----------- |
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
7. **HLLSet Early, Tokens Late**: Convert input to index sets ASAP, resolve tokens only at the end

### HLLSet Early, Tokens Late (Developer Guideline)

When implementing new processing functions, follow this pattern:

```text
INPUT → Indices HLLSets (ASAP) → Set Operations → ... → Token Resolution (LAST)
```

**Why:**

- **Efficiency**: HLLSet/index set operations are O(1) membership, O(n) union/intersection
- **Compact**: Index sets are fixed-size, regardless of content volume  
- **Composable**: Set operations are associative, commutative, idempotent
- **Deferred cost**: LUT lookups are expensive - do them only once at the end

**Correct pattern:**

```python
def process(query_text, lut, am):
    # Step 1: Convert to indices IMMEDIATELY
    tokens = tokenize(query_text)
    query_indices = {lut.get_ntoken_index(t) for t in tokens if lut.get_ntoken_index(t)}
    
    # Step 2-N: Pure set operations (no token info carried)
    cover = build_cover(query_indices, W)
    results = disambiguate(cover, am)  # Returns indices
    
    # LAST: Resolve to tokens only when returning to user
    return [lut.get_ntokens_at_index(idx) for idx in results]
```

**Anti-pattern (avoid):**

```python
def process(query_text, lut, am):
    tokens = tokenize(query_text)
    found_pairs = []
    for t in tokens:
        idx = lut.get_ntoken_index(t)
        found_pairs.append((t, idx))  # ❌ Carrying token info through processing
    # ... processing with (token, idx) pairs
```

---

## Bounded Evolution Store

For systems with vocabulary-based identification (uninflected sign systems), unbounded growth of AM/W/HRT can become problematic. The `BoundedEvolutionStore` implements a delta-eviction model based on:

```math
T(t+1) = (T(t) \cup N(t+1)) \setminus D(t)
```

### Conflict-Free by Design (CA Property)

**Content Addressable (CA) identification guarantees no conflicts between active state and archive:**

- Same content → same index (always, deterministically)
- Evicted index can be "reheated" without conflict
- Active + Archive = complete knowledge (no duplicates)
- Re-encountered content simply reactivates its index

This means:

- Query results are consistent regardless of eviction state
- Eviction is purely about memory bounds, not data loss
- Historical relationships are preserved and recoverable

### Store Architecture

Instead of full snapshots (Git model), we store:

- **Active state**: Current AM, W, HRT (bounded by capacity)
- **Archive**: Only evicted entries D(t)
- **Deltas**: N(t+1) for each evolution step

```python
from core.mf_algebra import BoundedEvolutionStore, Sparse3DConfig

config = Sparse3DConfig(p_bits=10, max_n=3)
store = BoundedEvolutionStore(
    config=config,
    capacity=100000,        # Maximum active indices
    eviction_policy='lru',  # 'lru', 'lfu', 'age', 'combined'
    eviction_batch=1000
)

# Evolution
n_added, n_evicted = store.evolve(new_edges, source="input")

# Query including archived data (conflict-free due to CA)
results = store.query_with_archive(query_indices, include_archived=True)

# Reheat archived indices back to active state
n_reheated, n_evicted = store.reheat({index1, index2})

# Check index status
status = store.index_status(idx)  # 'active', 'archived', 'active+archived', 'unknown'

# Conservation check (Noether-inspired stability)
conservation = store.conservation_check()
# {'total_new': 160, 'total_deleted': 60, 'stable': True/False}
```

### HLLSet Memory Archaeology

System states can be compressed into HLLSet snapshots for **O(1) similarity comparison**:

```python
# Configure snapshot interval
store = BoundedEvolutionStore(
    config=config,
    capacity=100000,
    snapshot_interval=10  # Snapshot every 10 evolutions
)

# Find historical states similar to query (O(1) per comparison)
matches = store.find_similar_memories(query_indices, top_k=5)
for similarity, snapshot in matches:
    print(f"tau={similarity:.3f}, era={snapshot.source}")

# Find the single best match ("deepest memory from childhood")
deepest = store.find_deepest_memory(query_indices)

# Manual snapshot before major operation
snapshot = store.take_manual_snapshot("before_migration")

# Get current state as HLLSet (for comparison without archiving)
current = store.current_state_hllset()
similarity = current.tau(historical_snapshot)
```

**Key insight**: HLLSet provides O(1) similarity regardless of state size. This enables:

- **Memory archaeology**: Instantly find which historical state best matches a query
- **Drift detection**: Compare current state to baseline snapshots
- **Semantic versioning**: Tag important states and find them later by similarity

**Trade-offs:**

| Full Snapshots (CommitStore) | Delta-Eviction (BoundedEvolutionStore) |
| ------------------------------ | ---------------------------------------- |
| ✓ Instant rollback | ✗ Requires replay for rollback |
| ✗ Unbounded growth | ✓ Bounded active state |
| ✓ Full history preserved | ✓ Archive preserves evicted data |
| Good for (reg, zeros) IDs | Good for vocabulary IDs |
| Conflicts possible | ✓ **Conflict-free (CA property)** |
| O(n) state comparison | ✓ **O(1) HLLSet similarity** |

---

### FingerprintIndex: HLLSet Commit LUT

The `FingerprintIndex` provides a **Bloom Filter Tower** - a two-level filter for history search:

```text
┌─────────────────────────────────────────┐
│     System Fingerprint (Union)          │  ← Top: "Has it EVER existed?"
│     = max(all commit HLLSets)           │     No false negatives
├─────────────────────────────────────────┤
│  Commit N   │  Commit N-1  │  ...       │  ← Levels: Individual commits
│  HLLSet     │  HLLSet      │            │     Each is a HLLSet (works as a bloom filter)
├─────────────────────────────────────────┤
│  Commit 2   │  Commit 1    │            │
│  HLLSet     │  HLLSet      │            │
└─────────────────────────────────────────┘
```

The tower follows the evolution equation: $\mathcal{T}(t+1) = (\mathcal{T}(t) \cup N(t+1)) \setminus D(t)$

```python
from core.mf_algebra import FingerprintIndex, CommitFingerprint

# Create index
index = FingerprintIndex(p_bits=10)

# Add commit fingerprints
index.add_commit("commit_001", {42, 137, 999}, timestamp=1.0)
index.add_commit("commit_002", {137, 500, 777}, timestamp=2.0)
index.add_commit("commit_003", {42, 500, 888}, timestamp=3.0)

# Level 1: Fast system-wide check (O(1))
# NO FALSE NEGATIVES - if False, index never existed anywhere
if not index.system_contains_maybe(42):
    print("Index 42 never existed - skip history search")
else:
    # Level 2: Find candidate commits (filtered search)
    candidates = index.find_candidate_commits(42)
    for fp in candidates:
        print(f"Index 42 might be in {fp.commit_id}")

# Find commits similar to a query
similar = index.find_similar_commits({42, 137}, top_k=3)
for similarity, fp in similar:
    print(f"tau={similarity:.3f}, commit={fp.commit_id}")

# Find commits that might contain ALL given indices
commits = index.commits_containing({42, 137})
```

**Key property**: The system fingerprint is the **union of all commit HLLSets**:

- **No false negatives**: If `system_contains_maybe(idx)` returns `False`, the index was **never** in any commit
- **May have false positives**: Hash collisions mean `True` requires checking individual commits
- **Perfect for filtering**: Skip expensive history searches when result is certain

This is **not good for analytical work** (hash collisions blur precision), but **excellent for decision making** ("should we dive into history at all?").

---

## License

See [Apache License Version 2.0](LICENSE)

## References

1. [Solomonoff prior](https://medium.com/@swarnenduiitb2020/the-man-who-solved-learning-in-1964-and-why-we-ignored-him-for-60-years-602a23ddf956)
2. **CRDTs**: Conflict-free Replicated Data Types (Shapiro et al.)
3. **Karoubi Envelope**: Idempotent completion of a category
4. **HyperLogLog**: Probabilistic data structure (Flajolet et al.)
