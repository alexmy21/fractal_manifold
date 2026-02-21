# Fractal Manifold v0.4.0

**IICA Architecture** - Immutability, Idempotence, Content Addressability

A semantic memory system built on Hash Relational Tensors (HRT) with HyperLogLog probabilistic structures.

---

## Introduction

Fractal Manifold is a **content-addressable semantic memory** that stores and retrieves structured knowledge using probabilistic data structures. The system is built on three foundational principles (**IICA**) that are enforced throughout every layer:

### Design Philosophy

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
│                       Enables exactly-once semantics                    │
│                                                                         │
│  CONTENT              Object identity = hash of content                 │
│  ADDRESSABILITY       Same content → same name → deduplication          │
│  ──────────────       Enables structural sharing & verification         │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### The Foundation: HLLSet

The IICA properties aren't imposed artificially—they emerge naturally from the **HyperLogLog Set (HLLSet)** primitive at the core of the system.

**Important!** Despite its name, HLLSet is not a HyperLogLog cardinality estimator, which is based on Flajolet-Martin algorithms. We use the HyperLogLog prefix to honor the inventors and developers of this outstanding computational breakthrough.

```text
┌─────────────────────────────────────────────────────────────────────────┐
│                     HLLSet: NATURAL IICA                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  TOKEN ENCODING                                                         │
│  ──────────────                                                         │
│                                                                         │
│    "apple"  ──hash──▶  0x7A3F...  ──split──▶  register │ value        │
│                                                  │         │            │
│                                         p_bits (15) │  leading zeros    │
│                                                  ▼         ▼            │
│                                              reg=29873  zeros=7         │
│                                                                         │
│  WHY IICA IS NATURAL                                                    │
│  ───────────────────                                                    │
│                                                                         │
│  ✓ CONTENT ADDRESSABLE                                                  │
│    • "apple" always hashes to same (reg, zeros)                         │
│    • Token identity IS its hash encoding                                │
│    • No separate ID assignment needed                                   │
│                                                                         │
│  ✓ IDEMPOTENT                                                           │
│    • add("apple") twice = add("apple") once                             │
│    • HLLSet stores each hash at (reg, zeros) position in HLLSet         │
│    • Duplicates have no effect: (reg, zeros) is the same for given hash │
│                                                                         │
│  ✓ MERGE = UNION                                                        │
│    • HLLSet merge is bit-wise OR                                        │
│    • union(A, A) = A  (idempotent)                                      │
│    • union(A, B) = union(B, A)  (commutative)                           │
│    • union(union(A,B), C) = union(A, union(B,C))  (associative)         │
│                                                                         │
│  The architecture just adds IMMUTABILITY discipline                     │
│  ─────────────────────────────────────────────────                      │
│    • Return new HLLSet instead of mutating                              │
│    • Enables history, rollback, parallel ops                            │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

**The key insight**: HLLSet's hash-based encoding makes content addressability and idempotence *intrinsic* properties. The architecture simply:

1. Preserves these natural properties (doesn't break them)
2. Adds immutability discipline (return new objects, never mutate)
3. Propagates IICA to higher layers (HRT, persistence, orchestration)

This is why the system "just works"—we're not fighting against the data structure, we're amplifying its natural algebraic properties.

### Layered Architecture

The system is organized in layers, each enforcing IICA:

```text
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  Layer 4: ORCHESTRATION                                                 │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  ManifoldOS_IICA                                                │    │
│  │  • Coordinates ingestion, merge, persistence                    │    │
│  │  • Parallel batch processing                                    │    │
│  │  • Rollback & branching                                         │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                              │                                          │
│  Layer 3: STORAGE            ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  HRTPersistentStore + HRTStack                                  │    │
│  │  • Git-like commits (blobs, refs, HEAD)                         │    │
│  │  • Content-addressed deduplication                              │    │
│  │  • In-memory history with rollback                              │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                              │                                          │
│  Layer 2: TENSORS            ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  HRT_IICA (AM + Lattice + LUT)                                  │    │
│  │  • AM: Directed relationship graph (intersection cardinality)   │    │
│  │  • Lattice: Row/column HLLSet structures                        │    │
│  │  • LUT: Embedded token disambiguation                           │    │
│  │  • Merge: Associative, Commutative, Idempotent                  │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                              │                                          │
│  Layer 1: PRIMITIVES         ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  HLLSet + BasicHLLSet + Kernel + ImmutableTensor                │    │
│  │  • HLLSet: Probabilistic cardinality estimation                 │    │
│  │  • BasicHLLSet: Single (register, zeros) pair                   │    │
│  │  • Kernel: Merge operations with IICA guarantees                │    │
│  │  • ImmutableTensor: Content-hashed PyTorch tensors              │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### IICA Enforcement by Module

| Module | Immutability | Idempotence | Content Addressable |
| -------- | -------------- | ------------- | --------------------- |
| **HLLSet** | `add()` returns new HLLSet | `union(A,A) == A` | Hash of registers |
| **BasicHLLSet** | Frozen dataclass | `merge(A,A) == A` | `(reg, zeros)` tuple |
| **ImmutableTensor** | Returns new tensor on ops | `max(A,A) == A` | SHA1 of data |
| **Kernel** | Pure functions, no state | All ops idempotent | N/A (stateless) |
| **HRT_IICA** | `merge()` returns new HRT | `merge(A,A) == A` | SHA1 of structure |
| **EmbeddedLUT** | FrozenSet of entries | Union idempotent | Hash of entries |
| **HRTStack** | Append-only history | Commit idempotent | Commit SHA |
| **ManifoldOS_IICA** | HEAD moves, history preserved | Re-ingest = no-op | Via HRT names |

### Key Data Flow

```text
Token Stream                    HLLSet Space                 AM Space
────────────                    ────────────                 ────────
                                    
"the cat sat"     ──hash──▶    BasicHLLSets          ──coords──▶   AM[i,j]
                               (reg, zeros)                     intersection
                                    │                          cardinality
                                    ▼                              │
                               Lattice                             ▼
                            row_basic[i]                      Directed
                            col_basic[j]                        Graph
                                    │                              │
                                    └──────────┬───────────────────┘
                                               ▼
                                          W Matrix
                                    (filtered topology)
                                               │
                                               ▼
                                      Path Navigation
                                    (HLLSet composition)
```

### Why This Matters

1. **Parallelism**: Immutability enables lock-free concurrent operations
2. **Reproducibility**: Content addressing ensures same input → same output
3. **Efficiency**: Idempotence allows safe retries and deduplication
4. **Time Travel**: Immutable history enables rollback and branching
5. **Verification**: Content hashes enable integrity checking

---

## Architecture Overview

```text
┌─────────────────────────────────────────────────────────────────────────┐
│                         FRACTAL MANIFOLD                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐                │
│  │   HRT_IICA  │───▶│  HRTStack   │───▶│ Persistence │                │
│  │  (core)     │     │  (history)  │     │  (storage)  │                │
│  └─────────────┘     └─────────────┘     └─────────────┘                │
│         │                                                               │
│         ▼                                                               │
│  ┌─────────────────────────────────────────────────────┐                │
│  │                    HRT Structure                    │                │
│  │  ┌────────┐  ┌─────────────┐  ┌──────────────────┐  │                │
│  │  │   AM   │  │   Lattice   │  │   Embedded LUT   │  │                │
│  │  │(edges) │  │ (HLLSets)   │  │ (token mapping)  │  │                │
│  │  └────────┘  └─────────────┘  └──────────────────┘  │                │
│  └─────────────────────────────────────────────────────┘                │
│                                                                         │
│  ┌─────────────────────────────────────────────────────┐                │
│  │                 ManifoldOS_IICA                     │                │
│  │           (orchestration + parallel ops)            │                │
│  └─────────────────────────────────────────────────────┘                │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Core Concepts

### HRT (Hash Relational Tensor)

HRT is a semantic memory structure with three components:

| Component | Purpose | Implementation |
| ----------- | --------- | ---------------- |
| **AM** (Adjacency Matrix) | Directed graph of token relationships | Sparse tensor, `AM[i,j] = \|row_i ∩ col_j\|` |
| **Lattice** | HLLSet-based probabilistic structure | Row/column basic HLLSets |
| **LUT** (Lookup Table) | Token disambiguation | `(reg, zeros) → tokens` |

### AM Semantics

**AM[i,j] = cardinality(basic_row[i] ∩ basic_col[j])**

This is not term frequency—it's **context covariance**.

```text
┌─────────────────────────────────────────────────────────────────────────┐
│                    CONTEXT COVARIANCE (not TF!)                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Each basic HLLSet accumulates ALL tokens that hash to that position:   │
│                                                                         │
│    basic_row[i] = { all tokens with row_hash → i }  = CONTEXT_i         │
│    basic_col[j] = { all tokens with col_hash → j }  = CONTEXT_j         │
│                                                                         │
│  The intersection measures SHARED CONTEXT:                              │
│                                                                         │
│    AM[i,j] = |CONTEXT_i ∩ CONTEXT_j|                                    │
│            = "how much context do positions i and j share?"             │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  TF (Term Frequency)         vs    Context Covariance           │    │
│  │  ───────────────────               ──────────────────           │    │
│  │  "how many times did              "how much context do          │    │
│  │   token X appear here?"            these positions share?"      │    │
│  │                                                                 │    │
│  │  Counts single token              Measures relationship         │    │
│  │  Unbounded accumulation           between accumulated sets      │    │
│  │  Sensitive to duplicates          Semantic similarity           │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                         │
│  WHY THIS MATTERS                                                       │
│  ────────────────                                                       │
│                                                                         │
│  • High AM[i,j] = positions i,j share many tokens = related contexts    │
│  • Low AM[i,j] = positions i,j share few tokens = unrelated contexts    │
│  • AM captures SEMANTIC PROXIMITY, not just co-occurrence               │
│  • Natural measure for "how related are these two positions?"           │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

**Properties:**

- **Bounded**: Value can't exceed `min(|row|, |col|)`
- **Deterministic**: Same HLLSets → same intersection → same value
- **IICA-safe**: Merge via element-wise max preserves idempotence
- **Directed**: `AM[i,j] ≠ AM[j,i]` (captures token order)
- **Not acyclic**: Allows duplicates ("the cat sat on the mat")

### W (Weighted Topology Matrix)

>**W = Filtered AM preserving navigable topology**

W's purpose is to preserve HLLSet topology:

- A path in W lattice represents an original (or derived) HLLSet as basic HLLSet composition
- Thresholds (tau, rho, epsilon) filter to keep topologically significant edges
- W enables navigation from any HLLSet to related content

```text
Original HLLSet → decompose → [basic_1, basic_2, ..., basic_n]
                                    ↓
                              Path in W lattice
                                    ↓
                   Compose basics along path → Reconstruct HLLSet
```

---

## Retrieval Flow

```text
┌─────────────────────────────────────────────────────────────────┐
│                        RETRIEVAL FLOW                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. QUERY           prompt → HLLSet (prompt_hll)                │
│       ↓                                                         │
│  2. W NAVIGATION    prompt_hll ↔ W lattice (topology)           │
│       ↓             find related basic HLLSets via paths        │
│  3. AM MAPPING      basics → (i,j) coords in AM                 │
│       ↓             AM[i,j] = |basic_i ∩ basic_j|               │
│  4. DISAMBIGUATION  (i,j) → LUT → candidate tokens              │
│       ↓             resolve collisions                          │
│  5. ORDER RESTORE   AM is DIRECTED: START → t₁ → t₂ → ... → END │
│                     follow edges, duplicates OK (not acyclic!)  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Key Properties

| Property | Why It Matters |
| ---------- | ---------------- |
| **AM is directed** | Captures `token_i → token_j` order (not symmetric) |
| **AM not acyclic** | Allows duplicates in reconstructed text |
| **START/END symbols** | Bound discovery, prevent infinite traversal |
| **W preserves topology** | Navigate from any HLLSet to related content |
| **Intersection cardinality** | Edge strength = how related two positions are |

---

## IICA Properties

>**Immutability, Idempotence, Content Addressability**

```python
# Immutability: All operations return new HRT
hrt_new = hrt.merge(other, kernel)  # hrt unchanged

# Idempotence: merge(A, A) == A
assert hrt.merge(hrt, kernel).name == hrt.name

# Content Addressability: name = SHA1(content)
# Same content → same hash → same name
```

### Merge Algebra

Merge is:

- **Associative**: `merge(merge(A,B), C) == merge(A, merge(B,C))`
- **Commutative**: `merge(A,B) == merge(B,A)`
- **Idempotent**: `merge(A,A) == A`

This enables **lossless parallel merge** via divide-and-conquer:

```text
[A, B, C, D, E, F, G, H]  →  8 HRTs
   ↓ parallel merge pairs
[AB, CD, EF, GH]          →  4 HRTs  
   ↓ parallel merge pairs
[ABCD, EFGH]              →  2 HRTs
   ↓ final merge
[ABCDEFGH]                →  1 HRT
```

---

## Persistence Model

Git-like content-addressable storage:

```text
┌─────────────────────────────────────────┐
│              Git Semantics              │
├─────────────────────────────────────────┤
│  Blob     │ Content-addressed data      │
│  Commit   │ Points to blobs + parent    │
│  Ref      │ Named pointer (branch)      │
│  HEAD     │ Current commit              │
└─────────────────────────────────────────┘
```

```python
# Push to persistent store
mos.push()

# Reload from store (simulates restart)
mos2 = ManifoldOS_IICA(store_path=store_path)
assert mos2.head.name == original_name  # Round-trip verified
```

---

## Quick Start

```python
from core import (
    create_manifold_iica, create_parallel_manifold,
    HRT_IICA, Kernel
)

# Create ManifoldOS orchestrator
mos = create_manifold_iica()

# Ingest tokens
mos.ingest(["hello", "world", "semantic", "memory"])

# Access HRT
hrt = mos.head
print(f"HRT: {hrt.name}")
print(f"LUT entries: {len(hrt.lut)}")

# Parallel ingestion
mos_parallel = create_parallel_manifold(workers=4)
batches = [["batch1"], ["batch2"], ["batch3"], ["batch4"]]
mos_parallel.parallel_ingest(batches)

# Persistence
mos_persistent = ManifoldOS_IICA(store_path="./hrt_store")
mos_persistent.ingest(["persistent", "data"])
mos_persistent.push()

# Rollback
mos.rollback(1)  # Go back one step
```

---

## Module Structure

```text
core/
├── __init__.py          # Clean exports, version 0.4.0
├── hrt.py               # HRT, HRTConfig, AdjacencyMatrix, HLLSetLattice
├── hrt_iica.py          # HRT_IICA, EmbeddedLUT, LUTEntry, HRTStack
├── hrt_store.py         # Persistence: HRTPersistentStore, Serializer
├── manifold_os_iica.py  # ManifoldOS_IICA orchestrator
├── immutable_tensor.py  # ImmutableTensor foundation
├── hllset.py            # HLLSet probabilistic structures
├── kernel.py            # Kernel (merge operations)
└── algebra.py           # Algebraic operations
```

---

## Design Decisions

### Why Intersection Cardinality for AM?

Previous approach (TF accumulation) had issues:

- Unbounded values (duplicates inflate)
- Not deterministic across merge orders
- Breaks IICA guarantees

Intersection cardinality:

- `AM[i,j] = |basic_row[i] ∩ basic_col[j]|`
- Bounded by min set size
- Deterministic (set operation)
- Idempotent (max merge preserves value)

### Why Directed AM?

Text has order: "cat sat" ≠ "sat cat"

- `AM[i,j]` = evidence that position i precedes position j
- Traverse from START to END to reconstruct
- Cycles allowed (repeated words)

### Why W Separate from AM?

- **AM** = ground truth (all intersection cardinalities)
- **W** = navigable topology (filtered by tau/rho/epsilon)
- W preserves paths that reconstruct original HLLSets
- Keeps navigation efficient while AM stores complete information

---

## Roadmap

- [x] IICA Architecture (v0.4.0)
- [x] HRT_IICA with embedded LUT
- [x] Git-like persistence
- [x] Parallel merge/ingestion
- [ ] GPU + Sparse tensors (CUDA)
- [ ] AM intersection semantics implementation
- [ ] START/END token handling
- [ ] Full retrieval pipeline

---

## Version History

| Version | Milestone |
| --------- | ----------- |
| 0.4.0 | IICA Architecture - Immutability, Idempotence, Content Addressability |
| 0.3.x | Previous manifold implementation (deprecated) |

---

## License

See [LICENSE](LICENSE)
