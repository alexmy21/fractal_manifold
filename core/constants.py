# core/constants.py
"""
Fractal Manifold Constants

This module defines constants used throughout the fractal manifold system:

LAYER 1: HLLSet Constants (Register Layer)
- P_BITS: HLL precision bits (m = 2^P registers)
- SHARED_SEED: Hash seed for reproducibility
- HASH_FUNC: Standard hash function for tokens

LAYER 2: BSS Constants (Morphism Construction)
- DEFAULT_TAU: Inclusion threshold for BSS_τ
- DEFAULT_RHO: Exclusion threshold for BSS_ρ
- Morphism exists iff BSS_τ ≥ τ AND BSS_ρ ≤ ρ

LAYER 3: Fractal Loop Constants (Lattice Topology)
- DEFAULT_N_TOKEN_SIZES: Window sizes for n-tokenization
- MIN_OVERLAP: Minimum overlap for lattice edge consideration
- MAX_CHAIN_LENGTH: Maximum chain length for edge tokenization
"""
import hashlib


# =============================================================================
# LAYER 1: HLLSet Constants (Register Layer)
# =============================================================================

SHARED_SEED = 42
P_BITS = 10          # HLL precision (m = 2^P = 1024 registers)
HASH_FUNC = lambda s: int(hashlib.sha256(s.encode()).hexdigest()[:8], 16) & 0x7FFFFFFF


# =============================================================================
# LAYER 2: BSS Constants (Morphism Construction)
# =============================================================================

# BSS thresholds for morphism existence: BSS_τ ≥ τ AND BSS_ρ ≤ ρ
DEFAULT_TAU = 0.7    # Inclusion threshold (higher = stricter)
DEFAULT_RHO = 0.3    # Exclusion threshold (lower = stricter)

# Constraint: 0 ≤ ρ < τ ≤ 1
assert 0 <= DEFAULT_RHO < DEFAULT_TAU <= 1, "BSS thresholds must satisfy 0 ≤ ρ < τ ≤ 1"


# =============================================================================
# LAYER 3: Fractal Loop Constants (Lattice Topology Analysis)
# =============================================================================

# N-tokenization: sliding window sizes for scale hierarchy
# Tokens → 1-grams, 2-grams, 3-grams, etc.
DEFAULT_N_TOKEN_SIZES = (1, 2, 3)

# Minimum overlap for considering lattice edges in topology analysis
# After BSS builds the lattice, overlap is used for structure analysis
MIN_OVERLAP = 0.0    # Any intersection counts (can be raised for pruning)

# Maximum chain length for edge tokenization (n-edges in the lattice)
# Higher = deeper fractal recursion but more computation
MAX_CHAIN_LENGTH = 5

# Entanglement threshold: scales are entangled if E(m,n) > this
ENTANGLEMENT_THRESHOLD = 0.1