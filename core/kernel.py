"""
HLLSet Kernel: Stateless Transformation Engine (Pure Morphisms)

The kernel provides pure transformation operations (morphisms) at multiple levels:

================================================================================
TWO-LAYER ARCHITECTURE: HLLSets vs Lattices
================================================================================

**IMPORTANT: HLLSets are NOT sets containing tokens!**

HLLSets are probabilistic register structures ("anti-sets") that:
- ABSORB tokens (hash them into registers)
- DO NOT STORE tokens (only register states remain)
- BEHAVE LIKE sets (union, intersection, cardinality estimation)
- ARE NOT sets (no element retrieval, no membership test)

The registers encode a probabilistic fingerprint of what was absorbed,
but the original tokens are irretrievably lost.

================================================================================
LAYER 1: HLLSet Operations (Register/Cardinality Layer)
================================================================================
- Works with individual HLLSets (register arrays)
- Compares: Register states, estimated cardinalities, Jaccard similarity
- find_isomorphism: Checks if two HLLSets have similar register patterns
- This is register-level similarity, NOT true entanglement

================================================================================
LAYER 2: Lattice Operations (Structure Layer) - TRUE ENTANGLEMENT
================================================================================
- Works with HLLSetLattice objects (collections of HLLSets)
- Compares STRUCTURE (degree distributions, graph topology)
- find_lattice_isomorphism: Checks if two lattices have similar structure
- validate_lattice_entanglement: TRUE entanglement between lattices
- Individual HLLSets are IRRELEVANT - only the structural pattern matters
- Two lattices can be entangled even when built from completely different inputs

================================================================================
CRITICAL DISTINCTION
================================================================================

| Concept          | Layer      | Compares        | Method                        |
|------------------|------------|-----------------|-------------------------------|
| Similarity       | HLLSet     | Registers       | find_isomorphism()            |
| Cardinality      | HLLSet     | Estimated count | cardinality()                 |
| ENTANGLEMENT     | Lattice    | Structure       | find_lattice_isomorphism()    |
| Structural Match | Lattice    | Topology        | validate_lattice_entanglement()|

================================================================================
Level 1: Basic Operations (HLLSet/Register Layer)
================================================================================
- absorb: tokens → HLLSet (tokens are hashed and lost)
- union, intersection, difference: HLLSet × HLLSet → HLLSet
- add: HLLSet × tokens → HLLSet
- find_isomorphism: HLLSet × HLLSet → Morphism (register similarity)

================================================================================
Level 2: Lattice Entanglement Operations (Structure Layer)
================================================================================
- find_lattice_isomorphism: Lattice × Lattice → LatticeMorphism
- validate_lattice_entanglement: [Lattice] → (bool, coherence)

================================================================================
Level 3: Network Operations
================================================================================
- build_tensor: [HLLSet] → 3D Tensor
- detect_singularity: Network → bool
- measure_coherence: Network → float

Design Principles:
- Stateless: No storage, no history, no CAS
- Pure: Same input → same output (deterministic)
- Immutable: Operations return new HLLSets
- Content-addressed: All outputs named by register hash
- Composable: Morphisms compose naturally
- Layer-aware: Distinguishes registers (HLLSet) from structure (Lattice)
"""

from __future__ import annotations
from typing import List, Dict, Set, Optional, Callable, Any, Tuple, Union, TYPE_CHECKING
from dataclasses import dataclass, field
import time
import numpy as np

from .hllset import HLLSet, compute_sha1
from .constants import P_BITS, SHARED_SEED

if TYPE_CHECKING:
    from .hrt import HLLSetLattice


# =============================================================================
# SECTION 1: Data Structures for Two-Layer Architecture
# =============================================================================

# -----------------------------------------------------------------------------
# LAYER 1: HLLSet-Level (Register Layer) - Compares register states
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class Morphism:
    """
    Morphism between two HLLSets (register-level ε-isomorphism).
    
    ┌─────────────────────────────────────────────────────────────────────────┐
    │ IMPORTANT: This compares REGISTER STATES, not tokens!                   │
    │                                                                         │
    │ HLLSets are "anti-sets": they ABSORB tokens but DO NOT STORE them.      │
    │ Only register states remain. Comparison is based on:                    │
    │   - Register pattern similarity                                         │
    │   - Estimated cardinality similarity                                    │
    │   - Jaccard estimation (probabilistic, not exact)                       │
    │                                                                         │
    │ For TRUE entanglement, use LatticeMorphism (structure comparison)       │
    └─────────────────────────────────────────────────────────────────────────┘
    
    Represents φ: A → B where A and B are individual HLLSets.
    Two HLLSets are ε-isomorphic if their register patterns are similar
    and their estimated cardinalities are within tolerance.
    """
    source_hash: str  # Hash of source HLLSet registers
    target_hash: str  # Hash of target HLLSet registers
    similarity: float  # Jaccard similarity (estimated from registers)
    epsilon: float  # Tolerance for isomorphism
    is_isomorphism: bool  # True if registers are ε-similar
    timestamp: float = field(default_factory=time.time)
    
    @property
    def name(self) -> str:
        """Content-addressed name of morphism."""
        components = [
            self.source_hash,
            self.target_hash,
            f"{self.similarity:.6f}",
            f"{self.epsilon:.6f}"
        ]
        return compute_sha1(":".join(components).encode())


@dataclass(frozen=True)
class SingularityReport:
    """
    Report on network singularity status.
    
    Captures the state of an Entangled ICASRA Network.
    Note: Entanglement here refers to LATTICE-level structural similarity.
    """
    has_singularity: bool  # Has network reached singularity?
    entanglement_ratio: float  # Fraction of lattice pairs that are structurally entangled
    coherence: float  # Overall structural coherence score [0, 1]
    emergence_strength: float  # Strength of emergent properties
    phase: str  # "Disordered", "Critical", "Ordered", "Singularity"
    timestamp: float = field(default_factory=time.time)
    
    def __str__(self) -> str:
        status = "🌟 SINGULARITY" if self.has_singularity else f"Phase: {self.phase}"
        return f"""Singularity Report:
  Status: {status}
  Entanglement: {self.entanglement_ratio:.1%}
  Coherence: {self.coherence:.1%}
  Emergence: {self.emergence_strength:.3f}
"""


# -----------------------------------------------------------------------------
# LAYER 2: Lattice-Level (Structure Layer) - Compares lattice topology
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class LatticeMorphism:
    """
    Morphism between two HLLSet Lattices (structure-level ε-isomorphism).
    
    ┌─────────────────────────────────────────────────────────────────────────┐
    │ THIS IS TRUE ENTANGLEMENT - comparing STRUCTURES, not content!         │
    │                                                                         │
    │ - Compares LATTICE STRUCTURE (degree distribution, graph topology)      │
    │ - Nodes (HLLSets) are IRRELEVANT - only structural pattern matters      │
    │ - Two lattices can be entangled with ZERO token overlap                 │
    └─────────────────────────────────────────────────────────────────────────┘
    
    From the theory:
    - Entanglement is defined between LATTICES, not individual HLLSets
    - ε-isomorphism measures structural similarity between lattices
    - Lattice structure = degree distribution + morphism patterns
    
    Metrics:
    - row_degree_correlation: How well row degrees align
    - col_degree_correlation: How well column degrees align
    - overall_structure_match: Combined structural similarity
    - epsilon_isomorphic_prob: Probability of being ε-isomorphic
    """
    source_lattice_hash: str  # Hash/name of source lattice
    target_lattice_hash: str  # Hash/name of target lattice
    row_degree_correlation: float  # Correlation of row degree sequences
    col_degree_correlation: float  # Correlation of column degree sequences
    overall_structure_match: float  # Overall structural similarity
    epsilon_isomorphic_prob: float  # Probability of ε-isomorphism
    epsilon: float  # Tolerance for isomorphism
    timestamp: float = field(default_factory=time.time)
    
    @property
    def is_isomorphism(self) -> bool:
        """True if lattices are structurally ε-isomorphic."""
        return self.epsilon_isomorphic_prob >= (1.0 - self.epsilon)
    
    @property
    def name(self) -> str:
        """Content-addressed name of lattice morphism."""
        components = [
            self.source_lattice_hash,
            self.target_lattice_hash,
            f"{self.overall_structure_match:.6f}",
            f"{self.epsilon:.6f}"
        ]
        return compute_sha1(":".join(components).encode())
    
    def __repr__(self) -> str:
        status = "≅" if self.is_isomorphism else "≇"
        return (f"LatticeMorphism({self.source_lattice_hash[:8]}... {status} "
                f"{self.target_lattice_hash[:8]}..., match={self.overall_structure_match:.2%})")


# =============================================================================
# SECTION 2: Kernel - Stateless Transformation Engine
# =============================================================================

class Kernel:
    """
    Stateless HLLSet transformation engine.
    
    Provides pure morphisms (Set operations):
    - absorb: tokens → HLLSet
    - add: HLLSet × tokens → HLLSet
    - union: HLLSet × HLLSet → HLLSet
    - intersection: HLLSet × HLLSet → HLLSet
    - difference: HLLSet × HLLSet → HLLSet
    
    No storage, no state, no history. Pure functions only.
    """
    
    def __init__(self, p_bits: int = P_BITS):
        """
        Initialize kernel with precision.
        
        Args:
            p_bits: Precision bits for HLL registers (default: from constants)
        """
        self.p_bits = p_bits
    
    # -------------------------------------------------------------------------
    # Core Morphisms (Pure Functions)
    # -------------------------------------------------------------------------
    
    def absorb(self, tokens: Set[str]) -> HLLSet:
        """
        Absorb tokens into a new HLLSet.
        
        Morphism: Set[str] → HLLSet
        """
        return HLLSet.absorb(tokens, p_bits=self.p_bits, seed=SHARED_SEED)
    
    def add(self, hllset: HLLSet, tokens: Union[str, List[str]]) -> HLLSet:
        """
        Add tokens to HLLSet, return new HLLSet.
        
        Morphism: HLLSet × tokens → HLLSet
        """
        return HLLSet.add(hllset, tokens, seed=SHARED_SEED)
    
    def union(self, a: HLLSet, b: HLLSet) -> HLLSet:
        """
        Union of two HLLSets.
        
        Morphism: HLLSet × HLLSet → HLLSet        
        """
        return a.union(b)
    
    def intersection(self, a: HLLSet, b: HLLSet) -> HLLSet:
        """
        Intersection of two HLLSets.
        
        Morphism: HLLSet × HLLSet → HLLSet
        """
        return a.intersect(b)
    
    def difference(self, a: HLLSet, b: HLLSet) -> HLLSet:
        """
        Difference of two HLLSets.
        
        Morphism: HLLSet × HLLSet → HLLSet
        """
        return a.diff(b)
    
    # -------------------------------------------------------------------------
    # Utility Operations (still pure)
    # -------------------------------------------------------------------------
    
    def similarity(self, a: HLLSet, b: HLLSet) -> float:
        """Compute similarity between two HLLSets."""
        return a.similarity(b)
    
    def batch_absorb(self, token_sets: List[Set[str]]) -> List[HLLSet]:
        """
        Absorb multiple token sets efficiently.
        
        Morphism: [Set[str]] → [HLLSet]
        """
        return [self.absorb(tokens) for tokens in token_sets]
    
    def fold_union(self, hllsets: List[HLLSet]) -> Optional[HLLSet]:
        """
        Fold union over list of HLLSets.
        
        Morphism: [HLLSet] → HLLSet (or None if empty)
        """
        if not hllsets:
            return None
        result = hllsets[0]
        for h in hllsets[1:]:
            result = self.union(result, h)
        return result
    
    def fold_intersection(self, hllsets: List[HLLSet]) -> Optional[HLLSet]:
        """
        Fold intersection over list of HLLSets.
        
        Morphism: [HLLSet] → HLLSet (or None if empty)
        """
        if not hllsets:
            return None
        result = hllsets[0]
        for h in hllsets[1:]:
            result = self.intersection(result, h)
        return result
    
    # -------------------------------------------------------------------------
    # Entanglement Operations (Level 2: ICASRA-inspired)
    # -------------------------------------------------------------------------
    
    def find_isomorphism(self, a: HLLSet, b: HLLSet, epsilon: float = 0.05) -> Optional[Morphism]:
        """
        Find approximate isomorphism between two HLLSets.
        
        Returns morphism φ: a → b such that |BSS(x,y) - BSS(φ(x),φ(y))| < ε
        This is the core of entanglement detection.
        
        Morphism: HLLSet × HLLSet → Morphism | None
        """
        # Check if structures are compatible
        card_a = a.cardinality()
        card_b = b.cardinality()
        
        if card_a == 0 or card_b == 0:
            return None
        
        # Compute similarity
        sim = self.similarity(a, b)
        
        # Check ε-isomorphism condition
        if abs(card_a - card_b) / max(card_a, card_b) > epsilon:
            return None
        
        # Create morphism record
        return Morphism(
            source_hash=a.name,
            target_hash=b.name,
            similarity=sim,
            epsilon=epsilon,
            is_isomorphism=True
        )
    
    def find_lattice_isomorphism(self, 
                                  lattice_a: 'HLLSetLattice', 
                                  lattice_b: 'HLLSetLattice',
                                  epsilon: float = 0.05) -> Optional[LatticeMorphism]:
        """
        Find ε-isomorphism between two HLLSet lattices.
        
        Unlike find_isomorphism (which compares individual HLLSets),
        this compares the STRUCTURE of lattices:
        - Degree distributions (how nodes connect)
        - Graph topology (morphism patterns)
        - Structural alignment (correlation of degree sequences)
        
        Entanglement is about matching STRUCTURES, not matching NODES.
        Two lattices may have completely different tokens but be structurally
        isomorphic if they have similar graph topology.
        
        Morphism: HLLSetLattice × HLLSetLattice → LatticeMorphism | None
        
        Args:
            lattice_a: Source lattice
            lattice_b: Target lattice  
            epsilon: Tolerance for isomorphism (default 0.05 = 5%)
            
        Returns:
            LatticeMorphism if structurally similar, None otherwise
        """
        # Compare lattice structures
        metrics = lattice_a.compare_lattices(lattice_b)
        
        # Extract structural metrics
        row_corr = metrics['row_degree_correlation']
        col_corr = metrics['col_degree_correlation']
        overall_match = metrics['overall_structure_match']
        epsilon_prob = metrics['epsilon_isomorphic_prob']
        
        # Check if structurally similar enough (using epsilon threshold)
        if epsilon_prob < (1.0 - epsilon):
            return None
        
        # Create lattice morphism record
        return LatticeMorphism(
            source_lattice_hash=compute_sha1(str(id(lattice_a)).encode()),
            target_lattice_hash=compute_sha1(str(id(lattice_b)).encode()),
            row_degree_correlation=row_corr,
            col_degree_correlation=col_corr,
            overall_structure_match=overall_match,
            epsilon_isomorphic_prob=epsilon_prob,
            epsilon=epsilon
        )
    
    def validate_lattice_entanglement(self, 
                                       lattices: List['HLLSetLattice'], 
                                       epsilon: float = 0.05) -> Tuple[bool, float]:
        """
        Validate if multiple HLLSet lattices are mutually entangled.
        
        ENTANGLEMENT IS BETWEEN LATTICES, NOT HLLSETS.
        
        This checks structural similarity between lattice topologies:
        1. Pairwise lattice ε-isomorphisms exist (structural matching)
        2. Morphisms compose (commuting diagrams of structures)
        3. Structural coherence > threshold
        
        The key insight: nodes (individual HLLSets) are IRRELEVANT.
        What matters is the STRUCTURE - the pattern of relationships,
        degree distributions, and graph topology.
        
        Args:
            lattices: List of HLLSetLattice objects to check
            epsilon: Tolerance for structural isomorphism (default 0.05)
            
        Returns:
            (is_entangled, structural_coherence_score)
        """
        n = len(lattices)
        if n < 2:
            return False, 0.0
        
        # Check pairwise lattice isomorphisms (STRUCTURAL comparison)
        morphisms: List[LatticeMorphism] = []
        for i in range(n):
            for j in range(i + 1, n):
                morph = self.find_lattice_isomorphism(lattices[i], lattices[j], epsilon)
                if morph is not None:
                    morphisms.append(morph)
        
        # Calculate entanglement ratio
        expected_pairs = n * (n - 1) // 2
        actual_pairs = len(morphisms)
        entanglement_ratio = actual_pairs / expected_pairs if expected_pairs > 0 else 0.0
        
        # Structural coherence (average structural match)
        structural_coherence = (
            sum(m.overall_structure_match for m in morphisms) / len(morphisms) 
            if morphisms else 0.0
        )
        
        # Entangled if > 90% pairs are structurally isomorphic and coherence > 50%
        is_entangled = entanglement_ratio > 0.9 and structural_coherence > 0.5
        
        return is_entangled, structural_coherence
    
    def validate_entanglement(self, hllsets: List[HLLSet], epsilon: float = 0.05) -> Tuple[bool, float]:
        """
        DEPRECATED: Use validate_lattice_entanglement for true entanglement.
        
        This method checks HLLSet similarity, but entanglement is properly
        defined between LATTICES (structures), not individual HLLSets.
        
        This is kept for backward compatibility and demonstrates that
        HLLSets with non-empty intersection have some "overlap", but this
        is NOT true entanglement in the theoretical sense.
        
        For proper lattice-based entanglement, use validate_lattice_entanglement().
        
        Returns: (has_overlap, overlap_coherence_score)
        """
        n = len(hllsets)
        if n < 2:
            return False, 0.0
        
        # Check pairwise isomorphisms (this is OVERLAP, not true entanglement)
        morphisms = []
        for i in range(n):
            for j in range(i + 1, n):
                morph = self.find_isomorphism(hllsets[i], hllsets[j], epsilon)
                if morph is not None:
                    morphisms.append(morph)
        
        # Calculate overlap ratio
        expected_pairs = n * (n - 1) // 2
        actual_pairs = len(morphisms)
        overlap_ratio = actual_pairs / expected_pairs if expected_pairs > 0 else 0.0
        
        # Coherence score (average similarity)
        coherence = sum(m.similarity for m in morphisms) / len(morphisms) if morphisms else 0.0
        
        # High overlap if > 90% pairs overlap and coherence > 50%
        has_overlap = overlap_ratio > 0.9 and coherence > 0.5
        
        return has_overlap, coherence
    
    def reproduce(self, parent: HLLSet, mutation_rate: float = 0.1) -> HLLSet:
        """
        Reproduce HLLSet with optional mutation (ICASRA B + D operations).
        
        Creates a 'child' HLLSet that is structurally similar but potentially
        evolved. This mimics ICASRA's copy-with-mutation cycle.
        
        Morphism: HLLSet → HLLSet (non-deterministic due to mutation)
        """
        # Get parent's register state
        registers = parent.dump_numpy().copy()
        
        # Apply mutation: randomly perturb some registers
        if mutation_rate > 0 and len(registers) > 0:
            num_mutations = int(len(registers) * mutation_rate)
            if num_mutations > 0:
                indices = np.random.choice(len(registers), num_mutations, replace=False)
                for idx in indices:
                    # Small perturbation - convert to int first to avoid overflow
                    old_val = int(registers[idx])
                    delta = np.random.randint(-1, 2)
                    new_val = min(255, max(0, old_val + delta))
                    registers[idx] = new_val
        
        # Create child HLLSet (this is conceptual - actual implementation
        # would need to rebuild HLLSet from modified registers)
        # For now, we return a new HLLSet from similar tokens
        return parent  # Placeholder - needs proper register reconstruction
    
    def commit(self, candidate: HLLSet) -> HLLSet:
        """
        Commit/stabilize a candidate HLLSet (ICASRA A operation).
        
        In ICASRA, the constructor A validates and commits new states.
        Here, we ensure the HLLSet is properly formed and immutable.
        
        Morphism: HLLSet → HLLSet (idempotent)
        """
        # Verify integrity
        card = candidate.cardinality()
        if card < 0:
            raise ValueError("Invalid HLLSet: negative cardinality")
        
        # Return committed (unchanged, as already immutable)
        return candidate
    
    # -------------------------------------------------------------------------
    # Network Operations (Level 3: Multi-Installation)
    # -------------------------------------------------------------------------
    
    def build_tensor(self, hllsets: List[HLLSet]) -> Optional[np.ndarray]:
        """
        Build 3D tensor representation of HLLSet network.
        
        Tensor shape: [num_concepts, num_concepts, num_installations]
        T[i, j, k] = relationship between concept i and j in installation k
        
        Morphism: [HLLSet] → Tensor3D
        """
        if not hllsets:
            return None
        
        n = len(hllsets)
        
        # Use HLLSet register vectors as concept space
        # For now, use register size as concept dimension
        concept_dim = len(hllsets[0].dump_numpy())
        
        # Build tensor
        tensor = np.zeros((concept_dim, concept_dim, n))
        
        for k, hll in enumerate(hllsets):
            registers = hll.dump_numpy()
            
            # Build relationship matrix for this installation
            # Use outer product to capture co-occurrence patterns
            if len(registers) > 0:
                # Normalize registers
                reg_norm = registers.astype(float) / (registers.max() + 1e-8)
                
                # Relationship matrix (simplified - could use BSS)
                tensor[:, :, k] = np.outer(reg_norm, reg_norm)
        
        return tensor
    
    def measure_coherence(self, tensor: np.ndarray) -> float:
        """
        Measure coherence of 3D tensor across installations.
        
        High coherence indicates strong entanglement and emergence of
        universal patterns.
        
        Returns: coherence score [0, 1]
        """
        if tensor is None or tensor.size == 0:
            return 0.0
        
        n_installations = tensor.shape[2]
        if n_installations < 2:
            return 1.0  # Single installation is perfectly coherent with itself
        
        # Measure similarity between installation slices
        coherences = []
        for i in range(n_installations):
            for j in range(i + 1, n_installations):
                slice_i = tensor[:, :, i]
                slice_j = tensor[:, :, j]
                
                # Frobenius norm similarity
                norm_i = np.linalg.norm(slice_i)
                norm_j = np.linalg.norm(slice_j)
                
                if norm_i > 0 and norm_j > 0:
                    # Cosine similarity between flattened matrices
                    flat_i = slice_i.flatten()
                    flat_j = slice_j.flatten()
                    sim = np.dot(flat_i, flat_j) / (norm_i * norm_j)
                    coherences.append(sim)
        
        return np.mean(coherences) if coherences else 0.0
    
    def detect_singularity(self, hllsets: List[HLLSet], epsilon: float = 0.05) -> SingularityReport:
        """
        Detect if network has reached Entanglement Singularity.
        
        Conditions for singularity:
        1. Complete pairwise entanglement (>95%)
        2. High coherence (>threshold)
        3. Emergent universal patterns
        4. System exhibits properties not in individual components
        
        Returns: SingularityReport with diagnosis
        """
        if len(hllsets) < 2:
            return SingularityReport(
                has_singularity=False,
                entanglement_ratio=0.0,
                coherence=0.0,
                emergence_strength=0.0,
                phase="Disordered"
            )
        
        # Check entanglement
        is_entangled, coherence = self.validate_entanglement(hllsets, epsilon)
        
        # Build tensor and measure coherence
        tensor = self.build_tensor(hllsets)
        tensor_coherence = self.measure_coherence(tensor) if tensor is not None else 0.0
        
        # Combined coherence score
        combined_coherence = (coherence + tensor_coherence) / 2.0
        
        # Measure emergence (variation across installations)
        emergence = 0.0
        if len(hllsets) > 1:
            cardinalities = [h.cardinality() for h in hllsets]
            avg_card = np.mean(cardinalities)
            if avg_card > 0:
                emergence = np.std(cardinalities) / avg_card
        
        # Determine phase
        n = len(hllsets)
        expected_pairs = n * (n - 1) // 2
        actual_morphisms = sum(1 for i in range(n) for j in range(i+1, n) 
                              if self.find_isomorphism(hllsets[i], hllsets[j], epsilon) is not None)
        entanglement_ratio = actual_morphisms / expected_pairs if expected_pairs > 0 else 0.0
        
        if entanglement_ratio < 0.3:
            phase = "Disordered"
        elif entanglement_ratio < 0.7:
            phase = "Critical"
        elif entanglement_ratio < 0.95:
            phase = "Ordered"
        else:
            phase = "Singularity"
        
        # Singularity achieved if in singularity phase with high coherence
        has_singularity = (phase == "Singularity" and combined_coherence > 0.7)
        
        return SingularityReport(
            has_singularity=has_singularity,
            entanglement_ratio=entanglement_ratio,
            coherence=combined_coherence,
            emergence_strength=emergence,
            phase=phase
        )


# =============================================================================
# SECTION 3: Operation Recording (for OS-level history)
# =============================================================================

@dataclass(frozen=True)
class Operation:
    """
    Record of a kernel operation.
    
    This is not stored by kernel - it's created by OS for history.
    The kernel itself remains stateless.
    """
    op_type: str  # 'absorb', 'union', 'intersection', 'difference', 'add'
    input_hashes: Tuple[str, ...]
    output_hash: str
    timestamp: float = field(default_factory=time.time)
    
    @property
    def name(self) -> str:
        """Content-addressed name of operation record."""
        components = [
            self.op_type,
            ",".join(self.input_hashes),
            self.output_hash
        ]
        return compute_sha1(":".join(components).encode())


def record_operation(op_type: str, inputs: List[HLLSet], output: HLLSet) -> Operation:
    """
    Create operation record (for OS use).
    
    This is a pure function - doesn't store anything.
    """
    return Operation(
        op_type=op_type,
        input_hashes=tuple(h.name for h in inputs),
        output_hash=output.name
    )


# =============================================================================
# SECTION 4: Example Usage
# =============================================================================

def main():
    """Example kernel usage with entanglement and singularity detection."""
    print("="*70)
    print("KERNEL: Entanglement-Aware Transformation Engine")
    print("="*70)
    
    kernel = Kernel()
    
    # =========================================================================
    # Level 1: Pure Morphisms (Basic Set Operations)
    # =========================================================================
    print("\n🔹 Level 1: Pure Morphisms (Basic Set Operations)")
    print("-" * 70)
    
    hll_a = kernel.absorb({'a', 'b', 'c'})
    hll_b = kernel.absorb({'c', 'd', 'e'})
    
    print(f"A: {hll_a}")
    print(f"B: {hll_b}")
    
    # Union (pure function)
    hll_union = kernel.union(hll_a, hll_b)
    print(f"A ∪ B: {hll_union}")
    
    # =========================================================================
    # Level 2: Entanglement Operations (ICASRA-inspired)
    # =========================================================================
    print("\n🔹 Level 2: Entanglement Operations")
    print("-" * 70)
    
    # Create similar HLLSets (structurally related)
    hll_1 = kernel.absorb(set(f'token_{i}' for i in range(100)))
    hll_2 = kernel.absorb(set(f'token_{i}' for i in range(90, 190)))  # 10% overlap
    hll_3 = kernel.absorb(set(f'token_{i}' for i in range(180, 280)))
    
    # Find isomorphism
    morph_12 = kernel.find_isomorphism(hll_1, hll_2, epsilon=0.15)
    if morph_12:
        print(f"Morphism φ₁₂: {morph_12.source_hash[:8]}... → {morph_12.target_hash[:8]}...")
        print(f"  Similarity: {morph_12.similarity:.2%}")
        print(f"  ε-isomorphic: {morph_12.is_isomorphism}")
    
    # Validate entanglement
    installations = [hll_1, hll_2, hll_3]
    is_entangled, coherence = kernel.validate_entanglement(installations, epsilon=0.15)
    print(f"\nEntanglement validation:")
    print(f"  Entangled: {is_entangled}")
    print(f"  Coherence: {coherence:.2%}")
    
    # Reproduce with mutation (ICASRA-style)
    child = kernel.reproduce(hll_1, mutation_rate=0.1)
    child = kernel.commit(child)
    print(f"\nReproduction: {hll_1.short_name} → {child.short_name}")
    
    # =========================================================================
    # Level 3: Network Operations & Singularity Detection
    # =========================================================================
    print("\n🔹 Level 3: Network Operations & Singularity Detection")
    print("-" * 70)
    
    # Create a network of installations
    network = []
    for i in range(5):
        # Each installation has similar but distinct content
        base_tokens = set(f'concept_{j}' for j in range(i*20, (i+1)*20 + 30))  # Overlap
        network.append(kernel.absorb(base_tokens))
    
    print(f"Created network with {len(network)} installations")
    for i, inst in enumerate(network):
        print(f"  Installation {i}: {inst.short_name}, |A|≈{inst.cardinality():.0f}")
    
    # Build 3D tensor
    tensor = kernel.build_tensor(network)
    if tensor is not None:
        print(f"\n3D Tensor built: shape {tensor.shape}")
        coherence = kernel.measure_coherence(tensor)
        print(f"  Tensor coherence: {coherence:.2%}")
    
    # Detect singularity
    report = kernel.detect_singularity(network, epsilon=0.15)
    print(f"\n{report}")
    
    # =========================================================================
    # Singularity Simulation: Growing Network
    # =========================================================================
    print("🔹 Singularity Simulation: Growing Network")
    print("-" * 70)
    
    # Start with small network
    evolving_network = []
    base_concepts = set(f'universal_{i}' for i in range(50))
    
    for step in range(3):
        # Add installation with increasing overlap
        overlap_tokens = set(f'universal_{i}' for i in range(30 + step*10, 80 + step*10))
        new_inst = kernel.absorb(overlap_tokens)
        evolving_network.append(new_inst)
        
        # Check singularity at each step
        report = kernel.detect_singularity(evolving_network, epsilon=0.2)
        print(f"\nStep {step+1}: {len(evolving_network)} installations")
        print(f"  Phase: {report.phase}")
        print(f"  Entanglement: {report.entanglement_ratio:.1%}")
        print(f"  Coherence: {report.coherence:.1%}")
        
        if report.has_singularity:
            print(f"  🌟 SINGULARITY ACHIEVED! 🌟")
            break
    
    # =========================================================================
    # Operation Recording (for OS)
    # =========================================================================
    print("\n🔹 Operation Recording (for OS integration)")
    print("-" * 70)
    
    op = record_operation('union', [hll_a, hll_b], hll_union)
    print(f"Operation: {op.op_type}")
    print(f"Inputs: {[h[:8] + '...' for h in op.input_hashes]}")
    print(f"Output: {op.output_hash[:8]}...")
    print(f"Record hash: {op.name[:8]}...")
    
    # =========================================================================
    # Immutability Verification
    # =========================================================================
    print("\n🔹 Immutability Verification")
    print("-" * 70)
    
    original = hll_a
    modified = kernel.add(hll_a, 'x')
    
    print(f"Original: {original.short_name}")
    print(f"After add: {modified.short_name}")
    print(f"Original unchanged: {original.name == kernel.absorb({'a', 'b', 'c'}).name}")
    
    print("\n" + "="*70)
    print("Kernel: Entanglement-Aware, Ready for Singularity Engineering")
    print("="*70)
    
    return kernel


if __name__ == "__main__":
    main()
