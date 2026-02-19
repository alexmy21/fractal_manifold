# File: core/manifold_os.py
"""
Manifold OS: Universal Constructor with Driver Management

The OS implements the ICASRA Universal Constructor pattern:
- **A (Constructor)**: Validates and commits states
- **B (Copier)**: Reproduces states with structural preservation
- **C (Controller)**: Coordinates driver lifecycle and resources
- **D (Interface)**: Manages external data ingestion

Design Principles:
- Immutability: All data structures are immutable
- Idempotence: Operations can be repeated safely
- Content Addressability: Everything identified by content hash
- No Scheduling: Immutability eliminates need for complex scheduling
- Driver Lifecycle: Wake → Active → Idle → Remove (or Restart)

Driver Management:
- Drivers are ephemeral, stateless workers
- OS manages driver lifecycle (wake, idle, restart, remove)
- Processes cannot harm each other (pure functional)
- First driver: IngestDriver (tokenizes external data)
"""

from __future__ import annotations
from typing import List, Dict, Set, Optional, Callable, Any, Tuple, Iterator
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum
import time
import json
import hashlib
from pathlib import Path
import threading

import numpy as np

from .kernel import Kernel, HLLSet, compute_sha1, Morphism, SingularityReport
from .constants import SHARED_SEED
from .hrt import (
    HRT, HRTConfig, AdjacencyMatrix, 
    HLLSetLattice, BasicHLLSet, Cover
)
from .entanglement import (
    EntanglementEngine, EntanglementMorphism,
    compute_hrt_entanglement, compute_entanglement_network
)
from .state_mapping import (
    StateMapping, StateMapper, HLLSetSwarm, TrajectoryTriple,
    map_mos_states, create_state_swarm
)


# =============================================================================
# SECTION 1: Driver Abstraction (Universal Constructor Pattern)
# =============================================================================

class DriverState(Enum):
    """Driver lifecycle states."""
    CREATED = "created"       # Just instantiated
    IDLE = "idle"             # Waiting for work
    ACTIVE = "active"         # Processing
    ERROR = "error"           # Failed, needs restart
    DEAD = "dead"             # Terminated, needs removal


@dataclass
class DriverStats:
    """Statistics for driver monitoring."""
    operations_count: int = 0
    errors_count: int = 0
    created_at: float = field(default_factory=time.time)
    last_active: Optional[float] = None
    total_active_time: float = 0.0
    
    def record_operation(self, duration: float):
        """Record successful operation."""
        self.operations_count += 1
        self.last_active = time.time()
        self.total_active_time += duration
    
    def record_error(self):
        """Record error."""
        self.errors_count += 1


class Driver:
    """
    Base driver class.
    
    Drivers are stateless, ephemeral workers that perform specific operations.
    They follow the lifecycle: CREATED → IDLE → ACTIVE → (IDLE | ERROR | DEAD)
    """
    
    def __init__(self, driver_id: str, driver_type: str):
        self.driver_id = driver_id
        self.driver_type = driver_type
        self.state = DriverState.CREATED
        self.stats = DriverStats()
        self._lock = threading.Lock()
    
    def wake(self):
        """Wake driver from idle state."""
        with self._lock:
            if self.state in (DriverState.CREATED, DriverState.IDLE):
                self.state = DriverState.IDLE
                return True
            return False
    
    def activate(self):
        """Activate driver for processing."""
        with self._lock:
            if self.state == DriverState.IDLE:
                self.state = DriverState.ACTIVE
                return True
            return False
    
    def idle(self):
        """Put driver to idle state."""
        with self._lock:
            if self.state == DriverState.ACTIVE:
                self.state = DriverState.IDLE
                return True
            return False
    
    def mark_error(self):
        """Mark driver as errored."""
        with self._lock:
            self.state = DriverState.ERROR
            self.stats.record_error()
    
    def mark_dead(self):
        """Mark driver as dead (needs removal)."""
        with self._lock:
            self.state = DriverState.DEAD
    
    def restart(self):
        """Restart driver (error → idle)."""
        with self._lock:
            if self.state == DriverState.ERROR:
                self.state = DriverState.IDLE
                return True
            return False
    
    def process(self, *args, **kwargs) -> Any:
        """Override in subclass to implement processing logic."""
        raise NotImplementedError("Subclass must implement process()")
    
    @property
    def is_alive(self) -> bool:
        """Check if driver is alive (not dead)."""
        return self.state != DriverState.DEAD
    
    @property
    def needs_restart(self) -> bool:
        """Check if driver needs restart."""
        return self.state == DriverState.ERROR


# =============================================================================
# SECTION 2: Ingest Driver (D - Interface to External Reality)
# =============================================================================

@dataclass
class TokenizationConfig:
    """Configuration for tokenization."""
    min_token_length: int = 1
    max_token_length: int = 100
    lowercase: bool = True
    remove_punctuation: bool = False
    split_on: str = " "
    
    # n-token disambiguation parameters
    use_n_tokens: bool = True  # Enable n-token representation
    n_token_groups: List[int] = field(default_factory=lambda: [1, 2, 3])  # 1-token, 2-token, 3-token
    maintain_order: bool = True  # Preserve token order in n-tokens
    
    # HLLSet extraction (IMPORTANT: AM is shared resource after commit)
    extract_hllset_from_am: bool = True  # Derive HLLSet from AM before commit
    # If True: Extract HLLSet from AM during ingestion (before AM is committed/merged)
    # If False: Skip HLLSet extraction (use when only AM is needed)


@dataclass
class LUTRecord:
    """
    Lookup Table Record for disambiguating tokens from HLLSet bits.
    
    Structure:
    - key: (reg, zeros) - register number and run of trailing zeros
    - hashes: list of hash values that contributed to this bit
    - tokens: list of token sequences that produced these hashes
    
    Purpose: Resolve one-to-many mapping from bits back to original tokens.
    """
    reg: int  # Register number
    zeros: int  # Run of trailing zeros
    hashes: List[int] = field(default_factory=list)
    tokens: List[Tuple[str, ...]] = field(default_factory=list)  # Token sequences
    
    def add_entry(self, hash_val: int, token_seq: Tuple[str, ...]):
        """Add hash and token sequence to this LUT record."""
        if hash_val not in self.hashes:
            self.hashes.append(hash_val)
            self.tokens.append(token_seq)
    
    def get_candidates(self) -> Set[str]:
        """Get all candidate tokens from this LUT entry."""
        candidates = set()
        for token_seq in self.tokens:
            candidates.update(token_seq)
        return candidates


@dataclass
class NTokenRepresentation:
    """
    Multiple representations of tokens using n-token groups.
    
    For tokens {a, b, c, d}:
    - 1-tokens: {a}, {b}, {c}, {d}
    - 2-tokens: {a,b}, {b,c}, {c,d}
    - 3-tokens: {a,b,c}, {b,c,d}
    
    Each group creates separate HLLSet with different hashes.
    Union of all groups should be equivalent.
    Order is implicitly preserved: {a} < {a,b} < {a,b,c}
    
    ARCHITECTURAL NOTE - AM as Source of Truth:
    - AM is built during ingestion and contains row/col HLLSets
    - complete_hllset is derived from AM BEFORE commit (AM becomes shared after)
    - This is optional: only extract if config.extract_hllset_from_am = True
    """
    original_tokens: List[str]  # Original ordered tokens
    n_token_groups: Dict[int, List[Tuple[str, ...]]] = field(default_factory=dict)  # n -> list of n-token tuples
    hllsets: Dict[int, HLLSet] = field(default_factory=dict)  # n -> HLLSet for that group
    luts: Dict[int, Dict[Tuple[int, int], LUTRecord]] = field(default_factory=dict)  # n -> LUT for that group
    complete_hllset: Optional[HLLSet] = None  # Complete HLLSet derived from AM (extracted before commit)
    
    def generate_n_tokens(self, n: int, maintain_order: bool = True) -> List[Tuple[str, ...]]:
        """
        Generate n-token groups from original tokens.
        
        Args:
            n: Size of token groups
            maintain_order: If True, preserve sequential order (sliding window)
                          If False, use all combinations
        
        Returns:
            List of n-token tuples
        """
        if n < 1 or n > len(self.original_tokens):
            return []
        
        if maintain_order:
            # Sliding window: preserves order
            # [a,b,c,d] with n=2 -> [(a,b), (b,c), (c,d)]
            result = []
            for i in range(len(self.original_tokens) - n + 1):
                result.append(tuple(self.original_tokens[i:i+n]))
            return result
        else:
            # All combinations: no order
            from itertools import combinations
            return [tuple(c) for c in combinations(self.original_tokens, n)]
    
    def build_n_token_groups(self, ns: List[int], maintain_order: bool = True):
        """Build all n-token groups for given n values."""
        for n in ns:
            self.n_token_groups[n] = self.generate_n_tokens(n, maintain_order)
    
    def get_implicit_order(self) -> List[Tuple[str, ...]]:
        """
        Get implicit order from n-token groups.
        
        Returns ordered list: {a} < {a,b} < {a,b,c} < {b} < {b,c} < ...
        """
        ordered = []
        for n in sorted(self.n_token_groups.keys()):
            ordered.extend(self.n_token_groups[n])
        return ordered
    
    def disambiguate_tokens(self, reg: int, zeros: int) -> Set[str]:
        """
        Disambiguate tokens from a specific bit (reg, zeros) using LUT intersection.
        
        Algorithm:
        1. For each n-token group, lookup candidates from LUT
        2. Intersect all candidate sets
        3. Result is narrowed set of possible tokens
        
        Returns:
            Set of candidate tokens (intersection across all n-token groups)
        """
        candidate_sets = []
        
        for n, lut in self.luts.items():
            key = (reg, zeros)
            if key in lut:
                candidates = lut[key].get_candidates()
                candidate_sets.append(candidates)
        
        if not candidate_sets:
            return set()
        
        # Intersection narrows down candidates
        result = candidate_sets[0]
        for candidates in candidate_sets[1:]:
            result = result.intersection(candidates)
        
        return result


# =============================================================================
# SECTION 2.5: Adjacency Matrix for Ingestion
# =============================================================================

@dataclass
class AMCell:
    """
    Cell in Adjacency Matrix.
    
    Tracks:
    - Frequency: how often column follows row
    - HLLSets: updated incrementally
    """
    row_id: Tuple[int, int]  # (reg, zeros) for row token
    col_id: Tuple[int, int]  # (reg, zeros) for col token
    frequency: int = 0
    
    def increment(self):
        """Increment frequency counter."""
        self.frequency += 1


@dataclass
class IngestionAdjacencyMatrix:
    """
    Adjacency Matrix built during ingestion using (reg, zeros) identifiers.
    
    TWO-PHASE USAGE:
    
    Phase 1 - INGESTION (Building AM):
    - Tokens processed through sliding window
    - Transition frequencies recorded in cells
    - Row/Column HLLSets aggregated
    - START/END markers establish boundaries
    
    Phase 2 - QUERY/PROMPT PROCESSING (Order Reconstruction):
    - Given: HLLSet from prompt + relevant retrieved HLLSets
    - Problem: HLLSets don't preserve order, only distinct tokens
    - Solution: Traverse AM to reconstruct original data order
    - Method: reconstruct_order() uses START→END traversal
    
    Structure:
    - Rows/Columns: (reg, zeros) identifiers from token hashes
    - Cell values: Frequency of transitions
    - Size: ~100K x 100K (vs millions of unique tokens)
    
    START/END are PERMANENT tokens:
    - START: (reg=-1, zeros=0) - Traversal entry point
    - END: (reg=-2, zeros=0) - Stop condition (with threshold)
    - AM holds order explicitly via transition patterns
    """
    
    # Matrix storage: {(row_id, col_id): AMCell}
    cells: Dict[Tuple[Tuple[int, int], Tuple[int, int]], AMCell] = field(default_factory=dict)
    
    # Row HLLSets: {row_id: HLLSet}
    row_hllsets: Dict[Tuple[int, int], HLLSet] = field(default_factory=dict)
    
    # Column HLLSets: {col_id: HLLSet}
    col_hllsets: Dict[Tuple[int, int], HLLSet] = field(default_factory=dict)
    
    # Special token identifiers
    START_ID: Tuple[int, int] = (-1, 0)
    END_ID: Tuple[int, int] = (-2, 0)
    
    def get_or_create_cell(self, row_id: Tuple[int, int], col_id: Tuple[int, int]) -> AMCell:
        """Get existing cell or create new one."""
        key = (row_id, col_id)
        if key not in self.cells:
            self.cells[key] = AMCell(row_id=row_id, col_id=col_id)
        return self.cells[key]
    
    def update_cell(self, row_id: Tuple[int, int], col_id: Tuple[int, int]):
        """
        Update cell frequency and corresponding row/column HLLSets.
        
        Args:
            row_id: (reg, zeros) for row token
            col_id: (reg, zeros) for column token
        """
        # Update cell frequency
        cell = self.get_or_create_cell(row_id, col_id)
        cell.increment()
    
    def update_row_hllset(self, row_id: Tuple[int, int], hllset: HLLSet, kernel: Kernel):
        """
        Update HLLSet for a row.
        
        Args:
            row_id: (reg, zeros) identifier
            hllset: HLLSet to merge into row
            kernel: Kernel for union operation
        """
        if row_id in self.row_hllsets:
            # Union with existing
            self.row_hllsets[row_id] = kernel.union(self.row_hllsets[row_id], hllset)
        else:
            # First time
            self.row_hllsets[row_id] = hllset
    
    def update_col_hllset(self, col_id: Tuple[int, int], hllset: HLLSet, kernel: Kernel):
        """
        Update HLLSet for a column.
        
        Args:
            col_id: (reg, zeros) identifier
            hllset: HLLSet to merge into column
            kernel: Kernel for union operation
        """
        if col_id in self.col_hllsets:
            # Union with existing
            self.col_hllsets[col_id] = kernel.union(self.col_hllsets[col_id], hllset)
        else:
            # First time
            self.col_hllsets[col_id] = hllset
    
    def get_frequency(self, row_id: Tuple[int, int], col_id: Tuple[int, int]) -> int:
        """Get frequency for cell (row, col)."""
        key = (row_id, col_id)
        return self.cells[key].frequency if key in self.cells else 0
    
    def get_row_ids(self) -> Set[Tuple[int, int]]:
        """Get all row identifiers."""
        rows = set()
        for (row_id, _) in self.cells.keys():
            rows.add(row_id)
        return rows
    
    def get_col_ids(self) -> Set[Tuple[int, int]]:
        """Get all column identifiers."""
        cols = set()
        for (_, col_id) in self.cells.keys():
            cols.add(col_id)
        return cols
    
    def get_size(self) -> Tuple[int, int]:
        """Get matrix dimensions (rows, cols)."""
        return (len(self.get_row_ids()), len(self.get_col_ids()))
    
    def get_nonzero_count(self) -> int:
        """Get number of non-zero cells."""
        return len(self.cells)
    
    def get_complete_hllset(self, kernel: Kernel, use_rows: bool = True) -> Optional[HLLSet]:
        """
        Reconstruct complete HLLSet from AM's row or column HLLSets.
        
        KEY INSIGHT: AM already contains all information needed to reconstruct
        the original HLLSet! No need to store HLLSets separately during ingestion.
        
        The union of all row_hllsets (or col_hllsets) gives you the complete
        HLLSet representing all tokens in the ingested data.
        
        ARCHITECTURE BENEFIT:
        - Ingestion: Build AM only (simpler, single source of truth)
        - Query time: Derive HLLSet from AM when needed
        - AM serves dual purpose: order preservation + set operations
        
        Args:
            kernel: Kernel for union operations
            use_rows: If True, union row_hllsets; if False, union col_hllsets
        
        Returns:
            Complete HLLSet (union of all row/col HLLSets), or None if empty
        
        Example:
            >>> am = build_am_from_tokens(tokens, kernel)
            >>> hllset = am.get_complete_hllset(kernel)
            >>> # Now you have both: AM (order) + HLLSet (cardinality/operations)
        """
        hllsets = self.row_hllsets if use_rows else self.col_hllsets
        
        if not hllsets:
            return None
        
        # Union all HLLSets
        result = None
        for hllset in hllsets.values():
            if result is None:
                result = hllset
            else:
                result = kernel.union(result, hllset)
        
        return result
    
    def to_dense_array(self) -> np.ndarray:
        """
        Convert to dense numpy array (for visualization/analysis).
        Warning: May be large (~100K x 100K).
        """
        row_ids = sorted(self.get_row_ids())
        col_ids = sorted(self.get_col_ids())
        
        # Create mapping
        row_map = {rid: i for i, rid in enumerate(row_ids)}
        col_map = {cid: i for i, cid in enumerate(col_ids)}
        
        # Allocate array
        array = np.zeros((len(row_ids), len(col_ids)), dtype=np.int32)
        
        # Fill array
        for (row_id, col_id), cell in self.cells.items():
            i = row_map[row_id]
            j = col_map[col_id]
            array[i, j] = cell.frequency
        
        return array
    
    def reconstruct_order(self, target_cardinality: int, threshold_ratio: float = 0.9) -> List[Tuple[int, int]]:
        """
        Reconstruct original token order from HLLSets using AM traversal.
        
        USE CASE - QUERY/PROMPT PROCESSING:
        1. User provides prompt → create HLLSet from prompt
        2. System retrieves relevant HLLSets (basic or compound)
        3. Problem: HLLSets only have distinct tokens, no order
        4. Solution: Use this method to traverse AM and restore original order
        
        ALGORITHM:
        1. Start at START_ID
        2. At each position, find highest frequency transition to next node
        3. Skip END_ID until reaching threshold (threshold_ratio * target_cardinality)
        4. Stop when:
           - Reached END_ID after threshold
           - No more valid transitions
           - Reached max steps (prevent infinite loops)
        
        Args:
            target_cardinality: Expected number of distinct tokens (from HLLSet.cardinality())
            threshold_ratio: Fraction of cardinality before allowing END (default 0.9)
        
        Returns:
            List of (reg, zeros) identifiers representing reconstructed order
            
        Example:
            # During query processing:
            prompt_hllset = kernel.absorb(tokenize("show me data about X"))
            relevant_hllsets = system.retrieve_relevant(prompt_hllset)
            cardinality = relevant_hllsets[0].cardinality()
            
            # Reconstruct original order:
            ordered_ids = am.reconstruct_order(cardinality, threshold_ratio=0.9)
            # If cardinality=15, skip END until step ~14
            # Returns: [(reg1, zeros1), (reg2, zeros2), ...] in original order
        
        Note: This is a placeholder for future implementation.
        """
        # TODO: Implement traversal logic
        # 1. current = START_ID
        # 2. path = []
        # 3. while len(path) < threshold and not at END:
        #    - Get all cells with row_id = current
        #    - Filter out END_ID if len(path) < threshold
        #    - Choose next based on highest frequency
        #    - path.append(next)
        #    - current = next
        # 4. return path
        raise NotImplementedError("Traversal not yet implemented")


# =============================================================================
# SECTION 3: Ingest Driver (Enhanced with AM construction)
# =============================================================================

class IngestDriver(Driver):
    """
    Ingest Driver: The gateway between external reality and the system.
    
    Enhanced with n-token disambiguation algorithm:
    
    Responsibilities:
    1. Ingest raw external data (text, events, observations)
    2. Tokenize data into meaningful units
    3. Generate n-token representations (1-token, 2-token, 3-token, ...)
    4. Build HLLSets for each n-token group with separate hashing
    5. Maintain LUT (Lookup Table) for token disambiguation
    6. Structure results in Adjacency Matrix (AM) of HRT
    7. Return NTokenRepresentation with all HLLSets and LUTs
    
    Algorithm:
    - For tokens {a,b,c,d}, create:
      - 1-tokens: {a}, {b}, {c}, {d}
      - 2-tokens: {a,b}, {b,c}, {c,d}
      - 3-tokens: {a,b,c}, {b,c,d}
    - Each group gets different hash -> different HLLSet
    - Union of all groups is equivalent
    - Order preserved: {a} < {a,b} < {a,b,c}
    - LUT enables disambiguation: intersection of candidates across groups
    
    This implements the D (Interface) component of ICASRA.
    """
    
    def __init__(self, driver_id: str, config: Optional[TokenizationConfig] = None):
        super().__init__(driver_id, "ingest")
        self.config = config or TokenizationConfig()
        self.ingested_count = 0
        
        # n-token representations storage
        self.n_token_representations: Dict[str, NTokenRepresentation] = {}  # data_hash -> representation
        
        # Adjacency Matrix for ingestion
        self.adjacency_matrix = IngestionAdjacencyMatrix()
    
    def tokenize(self, raw_data: str) -> List[str]:
        """
        Tokenize raw data into ORDERED list of tokens.
        
        Args:
            raw_data: Raw string data from external reality
        
        Returns:
            Ordered list of tokens (order preserved for n-token generation)
        """
        if not raw_data:
            return []
        
        # Apply transformations
        text = raw_data
        if self.config.lowercase:
            text = text.lower()
        
        if self.config.remove_punctuation:
            import string
            text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Split into tokens
        tokens = text.split(self.config.split_on)
        
        # Filter by length (preserve order)
        tokens = [
            t.strip() for t in tokens 
            if self.config.min_token_length <= len(t.strip()) <= self.config.max_token_length
        ]
        
        return tokens  # Return list, not set, to preserve order
    
    def _build_am_from_tokens(self, tokens: List[str], kernel: Kernel) -> IngestionAdjacencyMatrix:
        """
        Build Adjacency Matrix from tokens with START/END markers.
        
        INGESTION PHASE - Building the AM:
        1. Add <START> token at beginning, <END> token at end
        2. For each adjacent pair (token_i, token_i+1):
           - Compute (reg, zeros) identifiers using HLLSet (avoid double calculation)
           - Update AM[from_id, to_id] frequency
           - Accumulate row/column HLLSets
        
        START/END enable order reconstruction during QUERY phase:
        - During ingestion: Record transition patterns with START/END boundaries
        - During query: Traverse AM from START→END to restore original order
        - HLLSets lose order → AM preserves it via transition frequencies
        - Threshold logic prevents premature END during traversal
        
        OPTIMIZATION:
        - Uses HLLSet.compute_reg_zeros_batch() to get identifiers
        - Avoids duplicate hash calculations (computed once in C backend)
        
        Args:
            tokens: Ordered list of tokens
            kernel: Kernel for HLLSet operations
        
        Returns:
            IngestionAdjacencyMatrix with updated cells and HLLSets
        """
        if not tokens:
            return self.adjacency_matrix
        
        # 1. Add START and END special tokens
        START_TOKEN = "<START>"
        END_TOKEN = "<END>"
        tokens_with_markers = [START_TOKEN] + tokens + [END_TOKEN]
        
        # 2. Compute (reg, zeros) for all tokens at once (efficient batch operation)
        reg_zeros_pairs = HLLSet.compute_reg_zeros_batch(
            tokens_with_markers, 
            p_bits=kernel.p_bits,
            seed=SHARED_SEED
        )
        
        # 3. Build transitions: for each adjacent pair
        for i in range(len(tokens_with_markers) - 1):
            from_token = tokens_with_markers[i]
            to_token = tokens_with_markers[i + 1]
            
            # Get identifiers (already computed, no recalculation!)
            if from_token == START_TOKEN:
                from_id = self.adjacency_matrix.START_ID
            else:
                from_id = reg_zeros_pairs[i]
            
            if to_token == END_TOKEN:
                to_id = self.adjacency_matrix.END_ID
            else:
                to_id = reg_zeros_pairs[i + 1]
            
            # 4. Update AM cell (increment transition frequency)
            self.adjacency_matrix.update_cell(from_id, to_id)
            
            # 5. Update row and column HLLSets
            # Create HLLSets for individual tokens
            from_hllset = kernel.absorb({from_token})
            to_hllset = kernel.absorb({to_token})
            
            self.adjacency_matrix.update_row_hllset(from_id, from_hllset, kernel)
            self.adjacency_matrix.update_col_hllset(to_id, to_hllset, kernel)
        
        return self.adjacency_matrix
    
    def _build_lut_for_n_tokens(self, n_tokens: List[Tuple[str, ...]], 
                                 hllset: HLLSet, n: int) -> Dict[Tuple[int, int], LUTRecord]:
        """
        Build Lookup Table (LUT) for n-token group.
        
        For each bit (reg, zeros) in HLLSet, track which token sequences contributed.
        
        Args:
            n_tokens: List of n-token tuples
            hllset: HLLSet created from these n-tokens
            n: Size of n-token group
        
        Returns:
            LUT: dict mapping (reg, zeros) -> LUTRecord
        """
        lut = {}
        
        # Get precision bits - handle both real and mock HLLSets
        if hasattr(hllset, 'precision_bits'):
            p_bits = hllset.precision_bits
        elif hasattr(hllset, 'p_bits'):
            p_bits = hllset.p_bits
        else:
            # Default for mock/test HLLSets
            p_bits = 10
        
        # For each n-token sequence
        for token_seq in n_tokens:
            # Join tokens to create string for hashing
            token_str = f"__n{n}__" + "__".join(token_seq)  # e.g., ('a','b') -> '__n2__a__b'
            
            # Compute hash (same as HLLSet does internally)
            # Note: This is simplified - actual implementation should match kernel hashing
            hash_val = hash(token_str) & 0xFFFFFFFFFFFFFFFF  # Ensure 64-bit positive
            
            # Determine which register and trailing zeros
            reg = hash_val & ((1 << p_bits) - 1)  # Lower p_bits
            remaining = hash_val >> p_bits
            
            # Count trailing zeros
            zeros = 0
            if remaining > 0:
                while (remaining & 1) == 0 and zeros < (64 - p_bits):
                    zeros += 1
                    remaining >>= 1
            else:
                zeros = 64 - p_bits  # Max for 64-bit hash
            
            key = (reg, zeros)
            if key not in lut:
                lut[key] = LUTRecord(reg=reg, zeros=zeros)
            
            lut[key].add_entry(hash_val, token_seq)
        
        return lut
    
    def process(self, raw_data: str, kernel: Kernel) -> NTokenRepresentation:
        """
        Process raw data through enhanced n-token ingestion pipeline.
        
        Algorithm:
        1. Tokenize into ordered list
        2. Generate n-token groups (1-token, 2-token, 3-token)
        3. Create HLLSet for each group (different hashes)
        4. Build LUT for each group (for disambiguation)
        5. Return NTokenRepresentation with all data
        
        Args:
            raw_data: Raw string data
            kernel: Kernel for HLLSet creation
        
        Returns:
            NTokenRepresentation with HLLSets and LUTs for all n-token groups
        """
        if not self.activate():
            raise RuntimeError(f"Driver {self.driver_id} not in idle state")
        
        start_time = time.time()
        
        try:
            # 1. Tokenize into ordered list
            tokens = self.tokenize(raw_data)
            
            if not tokens:
                # Empty result
                self.idle()
                return NTokenRepresentation(original_tokens=[])
            
            # 2. Create NTokenRepresentation
            representation = NTokenRepresentation(original_tokens=tokens)
            
            if not self.config.use_n_tokens:
                # Simple mode: just create single HLLSet from tokens
                hllset = kernel.absorb(set(tokens))
                representation.hllsets[1] = hllset
                self.idle()
                return representation
            
            # 3. Generate n-token groups
            representation.build_n_token_groups(
                self.config.n_token_groups, 
                self.config.maintain_order
            )
            
            # 3.5. Build Adjacency Matrix from tokens (with START/END markers)
            self._build_am_from_tokens(tokens, kernel)
            
            # 3.6. CRITICAL: Extract HLLSet from AM BEFORE commit
            # After commit, AM is merged into shared resource - can't derive individual HLLSets
            if self.config.extract_hllset_from_am:
                representation.complete_hllset = self.adjacency_matrix.get_complete_hllset(kernel)
            
            # 4. Create HLLSet for each n-token group
            for n, n_token_list in representation.n_token_groups.items():
                # Flatten n-tokens to strings for hashing
                # Use join with separator to distinguish n-token groups
                token_strings = []
                for token_tuple in n_token_list:
                    # Join with special separator to make each n-group distinct
                    token_str = f"__n{n}__" + "__".join(token_tuple)
                    token_strings.append(token_str)
                
                # Create HLLSet from these strings
                hllset = kernel.absorb(set(token_strings))
                representation.hllsets[n] = hllset
                
                # 5. Build LUT for this n-token group
                lut = self._build_lut_for_n_tokens(n_token_list, hllset, n)
                representation.luts[n] = lut
            
            # Store representation
            data_hash = compute_sha1(raw_data)
            self.n_token_representations[data_hash] = representation
            
            # Record stats
            duration = time.time() - start_time
            self.stats.record_operation(duration)
            self.ingested_count += 1
            
            # Commit before moving to next batch
            # This finalizes current state and preserves AM for cumulative updates
            self.commit()
            
            return representation
            
        except Exception as e:
            self.mark_error()
            raise RuntimeError(f"Ingest driver {self.driver_id} failed: {e}")
    
    def batch_process(self, raw_data_list: List[str], kernel: Kernel) -> List[NTokenRepresentation]:
        """
        Process multiple raw data items in batch.
        
        For batch processing:
        1. Concatenate all raw data with START/END markers
        2. Build single AM for entire batch
        3. Return list of NTokenRepresentations (one per item)
        
        Returns list of NTokenRepresentations (each with multiple HLLSets and LUTs).
        """
        results = []
        
        # For batch, we process each item individually but share the same AM
        for raw_data in raw_data_list:
            try:
                representation = self.process(raw_data, kernel)
                results.append(representation)
            except RuntimeError:
                # Skip failed items, driver will be in ERROR state
                if self.needs_restart:
                    break
        
        return results
    
    def get_representation(self, data_hash: str) -> Optional[NTokenRepresentation]:
        """Retrieve stored n-token representation by data hash."""
        return self.n_token_representations.get(data_hash)
    
    def get_adjacency_matrix(self) -> IngestionAdjacencyMatrix:
        """Get the current Adjacency Matrix."""
        return self.adjacency_matrix
    
    def reset_adjacency_matrix(self):
        """Reset the Adjacency Matrix (for new batch)."""
        self.adjacency_matrix = IngestionAdjacencyMatrix()
    
    def commit(self) -> bool:
        """
        Commit current state before moving to next batch.
        
        This operation:
        1. Finalizes current AM state (marks as committed)
        2. Prepares for next batch (driver returns to IDLE)
        3. Preserves AM for cumulative updates across batches
        
        Note: Commit is automatically called at end of process().
        Can be called manually if needed.
        
        Returns:
            True if commit successful, False otherwise
        """
        with self._lock:
            if self.state == DriverState.ACTIVE:
                # Finalize current state - AM is preserved for next batch
                self.state = DriverState.IDLE
                return True
            return False


# =============================================================================
# SECTION 1: Perceptron - Data Source
# =============================================================================

@dataclass
class Perceptron:
    """
    A perceptron absorbs external reality and produces HLLSets.
    
    Perceptrons are the boundary between external reality and the system.
    They have semantic names (from external reality) but output
    content-addressed HLLSets.
    """
    perceptron_id: str
    source_name: str  # Semantic name from external reality
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Statistics
    absorption_count: int = 0
    last_absorption: Optional[float] = None
    
    def absorb(self, data: Set[str], kernel: Kernel) -> HLLSet:
        """
        Absorb data from external reality into HLLSet.
        Uses kernel for transformation.
        
        Note: p_bits is determined by kernel configuration, not passed here.
        """
        hllset = kernel.absorb(data)
        self.absorption_count += 1
        self.last_absorption = time.time()
        return hllset


# =============================================================================
# SECTION 2: Processing Pipeline Stage
# =============================================================================

@dataclass
class PipelineStage:
    """
    A stage in the processing pipeline.
    
    Each stage transforms HLLSets using kernel operations.
    """
    stage_id: str
    operation: str  # 'union', 'intersection', 'difference', 'filter', etc.
    config: Dict[str, Any] = field(default_factory=dict)
    
    # Statistics
    processed_count: int = 0
    errors_count: int = 0
    
    def process(self, inputs: List[HLLSet], kernel: Kernel) -> Optional[HLLSet]:
        """
        Process inputs through this stage.
        Returns output HLLSet or None on error.
        """
        if len(inputs) < 1:
            return None
        
        try:
            if self.operation == 'union':
                result = inputs[0]
                for inp in inputs[1:]:
                    result = kernel.union(result, inp)
            elif self.operation == 'intersection':
                result = inputs[0]
                for inp in inputs[1:]:
                    result = kernel.intersection(result, inp)
            elif self.operation == 'difference' and len(inputs) == 2:
                result = kernel.difference(inputs[0], inputs[1])
            elif self.operation == 'identity':
                result = inputs[0]
            else:
                self.errors_count += 1
                return None
            
            self.processed_count += 1
            return result
            
        except Exception as e:
            self.errors_count += 1
            return None


# =============================================================================
# SECTION 3: Persistent State (Git-like)
# =============================================================================

@dataclass
class OSState:
    """
    A snapshot of OS state.
    
    Contains:
    - Root HLLSet (union of all system HLLSets)
    - HRT (Hash Relational Tensor) - memory structure
    - Perceptron states
    - Pipeline configuration
    - Metadata
    
    Stored persistently (e.g., in Git).
    """
    state_hash: str  # SHA1 of this state
    root_hllset_hash: str  # Hash of root HLLSet
    hrt_hash: str  # Hash of HRT
    perceptron_states: Dict[str, Dict[str, Any]]  # perceptron_id -> state
    pipeline_config: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    parent_state: Optional[str] = None  # Hash of parent state
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def compute_hash(self) -> str:
        """Compute state hash from content."""
        content = json.dumps({
            'root_hllset_hash': self.root_hllset_hash,
            'hrt_hash': self.hrt_hash,
            'perceptron_states': self.perceptron_states,
            'pipeline_config': self.pipeline_config,
            'parent_state': self.parent_state,
        }, sort_keys=True)
        return compute_sha1(content)


class PersistentStore:
    """
    Persistent storage for OS states.
    
    Git-like interface:
    - commit: Save current state
    - checkout: Restore state
    - log: View history
    
    Also stores HRTs (Hash Relational Tensors).
    """
    
    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or Path(".manifold_storage")
        self.states: Dict[str, OSState] = {}
        self.hrts: Dict[str, HRT] = {}  # HRT hash -> HRT
        self.head: Optional[str] = None
        
        # In-memory cache of HLLSets
        self._hllset_cache: Dict[str, HLLSet] = {}
    
    def commit(self, state: OSState) -> str:
        """
        Commit state to persistent storage.
        Returns state hash.
        """
        state_hash = state.compute_hash()
        state.state_hash = state_hash
        self.states[state_hash] = state
        self.head = state_hash
        
        # Persist to disk if path exists
        if self.storage_path:
            self._persist_state(state)
        
        return state_hash
    
    def checkout(self, state_hash: str) -> Optional[OSState]:
        """Checkout state by hash."""
        if state_hash in self.states:
            self.head = state_hash
            return self.states[state_hash]
        return None
    
    def get_history(self, from_hash: Optional[str] = None) -> List[OSState]:
        """Get state history from given hash back to root."""
        if from_hash is None:
            from_hash = self.head
        
        history = []
        current = from_hash
        visited = set()
        
        while current and current not in visited:
            visited.add(current)
            state = self.states.get(current)
            if not state:
                break
            history.append(state)
            current = state.parent_state
        
        return list(reversed(history))
    
    def store_hllset(self, hllset: HLLSet):
        """Store HLLSet in cache."""
        self._hllset_cache[hllset.name] = hllset
    
    def get_hllset(self, hllset_hash: str) -> Optional[HLLSet]:
        """Get HLLSet from cache."""
        return self._hllset_cache.get(hllset_hash)
    
    def store_hrt(self, hrt: HRT) -> str:
        """Store HRT, return its hash."""
        h = hrt.name
        self.hrts[h] = hrt
        return h
    
    def get_hrt(self, hrt_hash: str) -> Optional[HRT]:
        """Get HRT by hash."""
        return self.hrts.get(hrt_hash)
    
    def store_artifact(self, data: bytes, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Store arbitrary artifact (bytes) with optional metadata.
        Returns SHA1 hash as artifact ID.
        
        Args:
            data: Raw bytes to store
            metadata: Optional metadata dictionary
            
        Returns:
            Artifact ID (SHA1 hash)
        """
        # Compute SHA1 hash as ID
        artifact_id = hashlib.sha1(data).hexdigest()
        
        # Store in persistent storage
        artifact_path = self.storage_path / 'artifacts' / artifact_id[:2] / artifact_id
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(artifact_path, 'wb') as f:
            f.write(data)
        
        # Store metadata if provided
        if metadata:
            meta_path = self.storage_path / 'artifacts' / artifact_id[:2] / f"{artifact_id}.meta.json"
            with open(meta_path, 'w') as f:
                json.dump(metadata, f, indent=2)
        
        return artifact_id
    
    def retrieve_artifact(self, artifact_id: str) -> bytes:
        """
        Retrieve artifact by ID.
        
        Args:
            artifact_id: SHA1 hash of artifact
            
        Returns:
            Raw bytes
            
        Raises:
            FileNotFoundError: If artifact not found
        """
        artifact_path = self.storage_path / 'artifacts' / artifact_id[:2] / artifact_id
        
        if not artifact_path.exists():
            raise FileNotFoundError(f"Artifact not found: {artifact_id}")
        
        with open(artifact_path, 'rb') as f:
            return f.read()
    
    def create_entanglement(self, source_id: str, target_id: str, strength: float = 1.0):
        """
        Create entanglement relationship between two artifacts.
        
        Stores entanglement metadata for tracking relationships between
        data and metadata, or between related artifacts.
        
        Args:
            source_id: Source artifact ID
            target_id: Target artifact ID  
            strength: Entanglement strength (0.0 to 1.0)
        """
        # Store entanglement in manifest
        entanglement_path = self.storage_path / 'entanglements'
        entanglement_path.mkdir(parents=True, exist_ok=True)
        
        entanglement_file = entanglement_path / f"{source_id}.json"
        
        # Load existing entanglements or create new
        if entanglement_file.exists():
            with open(entanglement_file, 'r') as f:
                entanglements = json.load(f)
        else:
            entanglements = []
        
        # Add new entanglement
        entanglements.append({
            'target_id': target_id,
            'strength': strength,
            'timestamp': time.time()
        })
        
        # Save
        with open(entanglement_file, 'w') as f:
            json.dump(entanglements, f, indent=2)
    
    def get_entanglements(self, artifact_id: str) -> Dict[str, float]:
        """
        Get all entanglements for an artifact.
        
        Args:
            artifact_id: Artifact ID to get entanglements for
            
        Returns:
            Dictionary mapping target_id -> strength
        """
        entanglement_path = self.storage_path / 'entanglements'
        entanglement_file = entanglement_path / f"{artifact_id}.json"
        
        if not entanglement_file.exists():
            return {}
        
        # Load entanglements
        with open(entanglement_file, 'r') as f:
            entanglements = json.load(f)
        
        # Convert to dict (most recent strength for each target)
        result = {}
        for ent in entanglements:
            result[ent['target_id']] = ent['strength']
        
        return result
    
    def _persist_state(self, state: OSState):
        """Persist state to disk."""
        # Create storage directory
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Save state metadata
        state_file = self.storage_path / f"{state.state_hash}.json"
        with open(state_file, 'w') as f:
            json.dump({
                'state_hash': state.state_hash,
                'root_hllset_hash': state.root_hllset_hash,
                'perceptron_states': state.perceptron_states,
                'pipeline_config': state.pipeline_config,
                'timestamp': state.timestamp,
                'parent_state': state.parent_state,
                'metadata': state.metadata
            }, f, indent=2)
    
    def merge_kernel_cas(self, kernel: Kernel) -> HLLSet:
        """
        Merge kernel CAS into persistent storage.
        
        Since everything is HLLSet and immutable, we can merge
        all HLLSets from kernel CAS into a single root HLLSet.
        """
        hllsets = list(kernel.cas.values())
        if not hllsets:
            # Return empty HLLSet
            return kernel.absorb(set())
        
        # Union all HLLSets from kernel CAS
        root = hllsets[0]
        for hll in hllsets[1:]:
            root = kernel.union(root, hll)
        
        # Store in persistent cache
        self.store_hllset(root)
        
        return root


# =============================================================================
# SECTION 4: Evolution Loop
# =============================================================================

@dataclass
class EvolutionConfig:
    """Configuration for evolution loop."""
    max_iterations: int = 100
    convergence_threshold: float = 0.99
    sleep_interval: float = 0.1
    enable_self_modification: bool = False


class EvolutionLoop:
    """
    Self-generating evolution loop.
    
    The loop:
    1. Ingest data from perceptrons
    2. Process through pipeline
    3. Update system state
    4. Check convergence
    5. Repeat or self-modify
    """
    
    def __init__(self, config: EvolutionConfig = None):
        self.config = config or EvolutionConfig()
        self.iteration_count = 0
        self.convergence_history: List[float] = []
        self.is_running = False
    
    def step(self,
             perceptrons: List[Perceptron],
             pipeline: List[PipelineStage],
             kernel: Kernel,
             store: PersistentStore,
             current_state: Optional[OSState] = None) -> Tuple[OSState, bool]:
        """
        Execute one evolution step.
        
        Returns:
            (new_state, converged)
        """
        self.iteration_count += 1
        
        # 1. Ingest from perceptrons
        ingested_hllsets = []
        for perceptron in perceptrons:
            # In real system, perceptron would have actual data
            # Here we simulate with empty sets
            data = set()  # Would come from external reality
            if data:
                hllset = perceptron.absorb(data, kernel)
                ingested_hllsets.append(hllset)
        
        # 2. Process through pipeline
        processed_hllsets = ingested_hllsets.copy()
        for stage in pipeline:
            if processed_hllsets:
                result = stage.process(processed_hllsets, kernel)
                if result:
                    processed_hllsets = [result]
        
        # 3. Merge with current state
        if current_state:
            current_root = store.get_hllset(current_state.root_hllset_hash)
            if current_root and processed_hllsets:
                new_root = kernel.union(current_root, processed_hllsets[0])
            elif processed_hllsets:
                new_root = processed_hllsets[0]
            elif current_root:
                new_root = current_root
            else:
                new_root = kernel.absorb(set())
        else:
            new_root = processed_hllsets[0] if processed_hllsets else kernel.absorb(set())
        
        # Store new root
        store.store_hllset(new_root)
        
        # 4. Check convergence
        converged = self._check_convergence(new_root, current_state, store)
        
        # 5. Create new state
        new_state = OSState(
            state_hash="",  # Will be computed
            root_hllset_hash=new_root.name,
            perceptron_states={p.perceptron_id: {
                'absorption_count': p.absorption_count,
                'last_absorption': p.last_absorption
            } for p in perceptrons},
            pipeline_config={'stages': [s.stage_id for s in pipeline]},
            parent_state=current_state.state_hash if current_state else None
        )
        
        return new_state, converged
    
    def _check_convergence(self,
                          new_root: HLLSet,
                          current_state: Optional[OSState],
                          store: PersistentStore) -> bool:
        """Check if evolution has converged."""
        if not current_state:
            return False
        
        current_root = store.get_hllset(current_state.root_hllset_hash)
        if not current_root:
            return False
        
        similarity = new_root.similarity(current_root)
        self.convergence_history.append(similarity)
        
        return similarity >= self.config.convergence_threshold
    
    def run(self,
           perceptrons: List[Perceptron],
           pipeline: List[PipelineStage],
           kernel: Kernel,
           store: PersistentStore,
           initial_state: Optional[OSState] = None) -> OSState:
        """
        Run evolution loop until convergence or max iterations.
        """
        self.is_running = True
        current_state = initial_state
        
        for i in range(self.config.max_iterations):
            if not self.is_running:
                break
            
            current_state, converged = self.step(
                perceptrons, pipeline, kernel, store, current_state
            )
            
            if converged:
                print(f"Converged at iteration {i+1}")
                break
            
            time.sleep(self.config.sleep_interval)
        
        self.is_running = False
        return current_state
    
    def stop(self):
        """Stop evolution loop."""
        self.is_running = False


# =============================================================================
# SECTION 5: Manifold OS
# =============================================================================

class ManifoldOS:
    """
    Manifold Operating System - Universal Constructor.
    
    Implements ICASRA pattern:
    - **A (Constructor)**: Validates and commits states (commit/validate)
    - **B (Copier)**: Reproduces states (reproduce via kernel)
    - **C (Controller)**: Manages driver lifecycle and resources
    - **D (Interface)**: Ingests external data via IngestDriver
    
    Orchestrates:
    - Drivers (managed lifecycle: wake, idle, restart, remove)
    - Perceptrons (data sources)
    - Processing pipeline
    - Evolution loop
    - Persistent storage (Git-like)
    - Kernel (stateless transformations)
    - HRT (Hash Relational Tensor) - memory structure
    - Entanglement (lattice-based morphisms)
    
    Design:
    - Immutability: No process can harm another
    - Idempotence: Operations can be repeated safely
    - Content-addressable: Everything identified by hash
    - No scheduling: Pure functional eliminates need
    """
    
    def __init__(self, storage_path: Optional[Path] = None, hrt_config: Optional[HRTConfig] = None,
                 lut_db_path: Optional[str] = None, extensions: Optional[Dict[str, Dict[str, Any]]] = None):
        """
        Initialize ManifoldOS with optional extensions.
        
        Args:
            storage_path: Path for persistent HLLSet storage
            hrt_config: Configuration for HRT creation
            lut_db_path: (DEPRECATED) Path to DuckDB file - use extensions param instead
            extensions: Extension configuration dictionary
                       Example: {'storage': {'type': 'duckdb', 'db_path': ':memory:'}}
        """
        self.kernel = Kernel()
        self.store = PersistentStore(storage_path)
        self.evolution = EvolutionLoop()
        
        # HRT configuration (for creating HRTs later)
        self.hrt_config = hrt_config or HRTConfig()
        self.current_hrt: Optional[HRT] = None  # Current HRT
        
        # Entanglement engine
        self.entanglement_engine = EntanglementEngine(self.kernel)
        self.entanglement_morphisms: Dict[str, EntanglementMorphism] = {}  # hrt_hash -> morphism
        
        # Extension system
        from core.extensions import ExtensionRegistry, DuckDBStorageExtension
        self.extensions = ExtensionRegistry()
        
        # Setup storage extension
        self._setup_storage_extension(storage_path, lut_db_path, extensions)
        
        # Setup storage extension
        self._setup_storage_extension(storage_path, lut_db_path, extensions)
        
        # Components
        self.perceptrons: Dict[str, Perceptron] = {}
        self.pipeline: List[PipelineStage] = []
        
        # Current state
        self.current_state: Optional[OSState] = None
        
        # Driver management (Universal Constructor pattern)
        self.drivers: Dict[str, Driver] = {}  # driver_id -> Driver
        self.driver_lock = threading.Lock()
        self.driver_health_check_interval = 5.0  # seconds
        self._driver_monitor_thread = None
        self._monitoring_active = False
        
        # Initialize ingest driver
        self._init_ingest_driver()
        
        # Statistics
        self.start_time = time.time()
        self.processing_cycles = 0
    
    # -------------------------------------------------------------------------
    # Artifact Storage Delegation (delegates to PersistentStore)
    # -------------------------------------------------------------------------
    
    def store_artifact(self, data: bytes, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Store arbitrary artifact (bytes) with optional metadata.
        Delegates to PersistentStore.
        
        Args:
            data: Raw bytes to store
            metadata: Optional metadata dictionary
            
        Returns:
            Artifact ID (SHA1 hash)
        """
        return self.store.store_artifact(data, metadata)
    
    def retrieve_artifact(self, artifact_id: str) -> bytes:
        """
        Retrieve artifact by ID.
        Delegates to PersistentStore.
        
        Args:
            artifact_id: SHA1 hash of artifact
            
        Returns:
            Raw bytes
        """
        return self.store.retrieve_artifact(artifact_id)
    
    def create_entanglement(self, source_id: str, target_id: str, strength: float = 1.0):
        """
        Create entanglement relationship between two artifacts.
        Delegates to PersistentStore.
        
        Args:
            source_id: Source artifact ID
            target_id: Target artifact ID  
            strength: Entanglement strength (0.0 to 1.0)
        """
        return self.store.create_entanglement(source_id, target_id, strength)
    
    def get_entanglements(self, artifact_id: str) -> Dict[str, float]:
        """
        Get all entanglements for an artifact.
        Delegates to PersistentStore.
        
        Args:
            artifact_id: Artifact ID to get entanglements for
            
        Returns:
            Dictionary mapping target_id -> strength
        """
        return self.store.get_entanglements(artifact_id)
    
    def _setup_storage_extension(self, storage_path, lut_db_path, extensions):
        """
        Setup storage extension with backward compatibility.
        
        Priority order:
          1. extensions parameter (new way)
          2. lut_db_path parameter (backward compatibility)
          3. Auto-detect from storage_path
        """
        from core.extensions import DuckDBStorageExtension
        
        # Determine configuration
        if extensions and 'storage' in extensions:
            # New way: explicit extension config
            storage_config = extensions['storage']
        elif lut_db_path is not None:
            # Backward compatibility: lut_db_path parameter
            storage_config = {'type': 'duckdb', 'db_path': lut_db_path}
        else:
            # Auto-detect: default location
            if storage_path:
                db_path = str(storage_path / "metadata.duckdb")
            else:
                db_path = "metadata.duckdb"
            storage_config = {'type': 'duckdb', 'db_path': db_path}
        
        # Register storage extension
        if storage_config.get('type') == 'duckdb':
            extension = DuckDBStorageExtension()
            config = {'db_path': storage_config.get('db_path', ':memory:')}
            self.extensions.register('storage', extension, config)
    
    @property
    def lut_store(self):
        """
        Get LUT store from storage extension.
        
        Maintains backward compatibility - code using self.lut_store still works.
        """
        storage = self.extensions.get('storage')
        if storage and storage.is_available():
            return storage.lut_store
        return None
    
    def _init_ingest_driver(self):
        """Initialize default ingest driver."""
        driver_id = "ingest_default"
        driver = IngestDriver(driver_id)
        self.register_driver(driver)
        driver.wake()  # Wake driver to idle state
    
    # -------------------------------------------------------------------------
    # Driver Management (Universal Constructor - C Controller)
    # -------------------------------------------------------------------------
    
    def register_driver(self, driver: Driver):
        """
        Register a driver with the OS.
        
        Args:
            driver: Driver instance to register
        """
        with self.driver_lock:
            self.drivers[driver.driver_id] = driver
    
    def unregister_driver(self, driver_id: str) -> bool:
        """
        Unregister and remove a driver.
        
        Args:
            driver_id: ID of driver to remove
        
        Returns:
            True if removed, False if not found
        """
        with self.driver_lock:
            if driver_id in self.drivers:
                driver = self.drivers[driver_id]
                driver.mark_dead()
                del self.drivers[driver_id]
                return True
            return False
    
    def get_driver(self, driver_id: str) -> Optional[Driver]:
        """Get driver by ID."""
        return self.drivers.get(driver_id)
    
    def list_drivers(self) -> Dict[str, Dict[str, Any]]:
        """
        List all drivers with their status.
        
        Returns:
            Dict of driver_id -> {state, type, stats}
        """
        result = {}
        for driver_id, driver in self.drivers.items():
            result[driver_id] = {
                'state': driver.state.value,
                'type': driver.driver_type,
                'is_alive': driver.is_alive,
                'needs_restart': driver.needs_restart,
                'operations': driver.stats.operations_count,
                'errors': driver.stats.errors_count,
                'uptime': time.time() - driver.stats.created_at
            }
        return result
    
    def wake_driver(self, driver_id: str) -> bool:
        """
        Wake a driver from idle state.
        
        Args:
            driver_id: ID of driver to wake
        
        Returns:
            True if woken, False otherwise
        """
        driver = self.get_driver(driver_id)
        if driver:
            return driver.wake()
        return False
    
    def idle_driver(self, driver_id: str) -> bool:
        """
        Put driver to idle state.
        
        Args:
            driver_id: ID of driver to idle
        
        Returns:
            True if idled, False otherwise
        """
        driver = self.get_driver(driver_id)
        if driver:
            return driver.idle()
        return False
    
    def restart_driver(self, driver_id: str) -> bool:
        """
        Restart a driver (error -> idle).
        
        Args:
            driver_id: ID of driver to restart
        
        Returns:
            True if restarted, False otherwise
        """
        driver = self.get_driver(driver_id)
        if driver:
            return driver.restart()
        return False
    
    def cleanup_dead_drivers(self) -> List[str]:
        """
        Remove all dead drivers.
        
        Returns:
            List of removed driver IDs
        """
        removed = []
        with self.driver_lock:
            dead_ids = [
                driver_id for driver_id, driver in self.drivers.items()
                if driver.state == DriverState.DEAD
            ]
            for driver_id in dead_ids:
                del self.drivers[driver_id]
                removed.append(driver_id)
        return removed
    
    def start_driver_monitoring(self):
        """
        Start background thread for driver health monitoring.
        
        Monitors drivers and:
        - Restarts errored drivers
        - Removes dead drivers
        - Puts idle drivers to sleep
        """
        if self._monitoring_active:
            return
        
        self._monitoring_active = True
        self._driver_monitor_thread = threading.Thread(
            target=self._driver_monitor_loop,
            daemon=True
        )
        self._driver_monitor_thread.start()
    
    def stop_driver_monitoring(self):
        """Stop driver monitoring thread."""
        self._monitoring_active = False
        if self._driver_monitor_thread:
            self._driver_monitor_thread.join(timeout=1.0)
    
    def _driver_monitor_loop(self):
        """Background loop for driver health monitoring."""
        while self._monitoring_active:
            try:
                # Check each driver
                for driver_id, driver in list(self.drivers.items()):
                    if not driver.is_alive:
                        # Remove dead driver
                        self.unregister_driver(driver_id)
                    elif driver.needs_restart:
                        # Attempt restart
                        if not driver.restart():
                            # Failed to restart, mark dead
                            driver.mark_dead()
                
                # Cleanup
                self.cleanup_dead_drivers()
                
            except Exception as e:
                print(f"Driver monitor error: {e}")
            
            time.sleep(self.driver_health_check_interval)
    
    # -------------------------------------------------------------------------
    # Ingest Operations (D - Interface)
    # -------------------------------------------------------------------------
    
    def ingest(self, raw_data: str, driver_id: Optional[str] = None, 
               metadata: Optional[dict] = None) -> Optional[NTokenRepresentation]:
        """
        Ingest raw data from external reality using n-token algorithm.
        
        Enterprise-to-AI Metadata Bridge:
        1. Process raw data → NTokenRepresentation (in-memory)
        2. Store HLLSets → Content-addressed fingerprints
        3. Commit LUTs → Persistent metadata (DuckDB)
        
        The LUT persistence enables:
        - AI → Metadata: Query tokens by (reg, zeros)
        - Metadata → ED: Trace back to source data
        - Explainability: Link decisions to original records
        
        Args:
            raw_data: Raw string data to ingest
            driver_id: Optional specific driver ID (default: "ingest_default")
            metadata: Optional metadata (source, timestamp, user, etc.)
        
        Returns:
            NTokenRepresentation with multiple HLLSets and LUTs, or None on error
        """
        if driver_id is None:
            driver_id = "ingest_default"
        
        driver = self.get_driver(driver_id)
        if not driver or not isinstance(driver, IngestDriver):
            return None
        
        try:
            # STEP 1: Process in-memory (fast)
            representation = driver.process(raw_data, self.kernel)
            
            # STEP 2: Store HLLSets (content-addressed)
            for n, hllset in representation.hllsets.items():
                self.store.store_hllset(hllset)
            
            # STEP 3: Commit LUTs to persistent metadata store
            if self.lut_store:
                for n, lut in representation.luts.items():
                    hllset_hash = representation.hllsets[n].name
                    
                    # Add metadata (source tracking)
                    commit_metadata = metadata or {}
                    commit_metadata.update({
                        'n': n,
                        'driver_id': driver_id,
                        'original_length': len(representation.original_tokens),
                        'ingested_at': time.time()
                    })
                    
                    # Commit to DuckDB
                    commit_id = self.lut_store.commit_lut(
                        n=n,
                        lut=lut,
                        hllset_hash=hllset_hash,
                        metadata=commit_metadata
                    )
                    print(f"  ✓ LUT committed: n={n}, hash={hllset_hash[:16]}..., id={commit_id}")
            
            return representation
        except RuntimeError as e:
            print(f"Ingest failed: {e}")
            return None
    
    def ingest_batch(self, raw_data_list: List[str], 
                    driver_id: Optional[str] = None) -> List[NTokenRepresentation]:
        """
        Ingest batch of raw data using n-token algorithm.
        
        Args:
            raw_data_list: List of raw string data
            driver_id: Optional specific driver ID
        
        Returns:
            List of NTokenRepresentations (each with multiple HLLSets and LUTs)
        """
        if driver_id is None:
            driver_id = "ingest_default"
        
        driver = self.get_driver(driver_id)
        if not driver or not isinstance(driver, IngestDriver):
            return []
        
        try:
            representations = driver.batch_process(raw_data_list, self.kernel)
            
            # Store all HLLSets in persistent storage
            for representation in representations:
                for n, hllset in representation.hllsets.items():
                    self.store.store_hllset(hllset)
            
            return representations
        except Exception as e:
            print(f"Batch ingest failed: {e}")
            return []
    
    # -------------------------------------------------------------------------
    # Metadata Query Operations (ED ↔ AI Bridge)
    # -------------------------------------------------------------------------
    
    def query_tokens_from_metadata(self, n: int, reg: int, zeros: int,
                                   hllset_hash: Optional[str] = None) -> List[Tuple[str, ...]]:
        """
        Query tokens from persistent metadata store (AI → ED grounding).
        
        This is the critical bridge operation that links AI operations
        back to source enterprise data.
        
        Args:
            n: N-token group size (1, 2, 3, ...)
            reg: HLL register index
            zeros: Leading zeros count
            hllset_hash: Optional - filter by specific HLLSet
        
        Returns:
            List of token tuples that map to this (reg, zeros)
        
        Example:
            # AI has (reg=42, zeros=3) from HLLSet fingerprint
            # Ground back to source tokens:
            tokens = os.query_tokens_from_metadata(n=1, reg=42, zeros=3)
            # Returns: [('customer',), ('revenue',), ...]
        """
        if not self.lut_store:
            print("⚠ LUT store not available")
            return []
        
        return self.lut_store.get_tokens(n, reg, zeros, hllset_hash)
    
    def query_by_token(self, n: int, token_tuple: Tuple[str, ...]) -> List[Tuple[int, int]]:
        """
        Reverse lookup: Find (reg, zeros) keys for a token (ED → AI).
        
        Args:
            n: N-token group size
            token_tuple: Token tuple to search for
        
        Returns:
            List of (reg, zeros) keys
        
        Example:
            # Enterprise has token "customer"
            # Find HLLSet coordinates:
            keys = os.query_by_token(n=1, token_tuple=('customer',))
            # Returns: [(42, 3), (107, 1), ...]
        """
        if not self.lut_store:
            print("⚠ LUT store not available")
            return []
        
        return self.lut_store.query_by_token(n, token_tuple)
    
    def get_ingestion_metadata(self, hllset_hash: str) -> Optional[dict]:
        """
        Get metadata for an ingested HLLSet.
        
        Returns metadata like:
        - Source information
        - Ingestion timestamp
        - Driver ID
        - Original token count
        - Custom fields
        
        Args:
            hllset_hash: Content-addressed hash of HLLSet
        
        Returns:
            Metadata dict or None
        """
        if not self.lut_store:
            return None
        
        return self.lut_store.get_metadata(hllset_hash)
    
    def get_lut_stats(self) -> Optional[dict]:
        """
        Get statistics about persistent LUT storage.
        
        Returns:
            Dict with stats: total records, unique HLLSets, n-groups, etc.
        """
        if not self.lut_store:
            return None
        
        return self.lut_store.get_stats()

    
    # -------------------------------------------------------------------------
    # Perceptron Management
    # -------------------------------------------------------------------------
    
    def add_perceptron(self, perceptron_id: str, source_name: str,
                      metadata: Optional[Dict[str, Any]] = None) -> Perceptron:
        """Add a perceptron to the system."""
        p = Perceptron(
            perceptron_id=perceptron_id,
            source_name=source_name,
            metadata=metadata or {}
        )
        self.perceptrons[perceptron_id] = p
        return p
    
    def get_perceptron(self, perceptron_id: str) -> Optional[Perceptron]:
        return self.perceptrons.get(perceptron_id)
    
    def list_perceptrons(self) -> List[str]:
        return list(self.perceptrons.keys())
    
    # -------------------------------------------------------------------------
    # Pipeline Management
    # -------------------------------------------------------------------------
    
    def add_pipeline_stage(self, stage_id: str, operation: str,
                          config: Optional[Dict[str, Any]] = None) -> PipelineStage:
        """Add a stage to the processing pipeline."""
        stage = PipelineStage(
            stage_id=stage_id,
            operation=operation,
            config=config or {}
        )
        self.pipeline.append(stage)
        return stage
    
    def clear_pipeline(self):
        """Clear all pipeline stages."""
        self.pipeline.clear()
    
    # -------------------------------------------------------------------------
    # Processing Cycle
    # -------------------------------------------------------------------------
    
    def process_cycle(self, perceptron_data: Dict[str, Set[str]]) -> OSState:
        """
        Execute one processing cycle.
        
        Args:
            perceptron_data: Dict of perceptron_id -> data to absorb
        
        Returns:
            New OS state
        """
        self.processing_cycles += 1
        
        # 1. Ingest data from perceptrons
        ingested = []
        for perceptron_id, data in perceptron_data.items():
            perceptron = self.perceptrons.get(perceptron_id)
            if perceptron and data:
                hllset = perceptron.absorb(data, self.kernel)
                ingested.append(hllset)
                # Store in content store
                self.store.store_hllset(hllset)
        
        # 2. Process through pipeline
        processed = ingested.copy()
        for stage in self.pipeline:
            if processed:
                result = stage.process(processed, self.kernel)
                if result:
                    # Store intermediate result
                    self.store.store_hllset(result)
                    processed = [result]
        
        # 3. Merge with current state (root HLLSet)
        if self.current_state:
            current_root = self.store.get_hllset(self.current_state.root_hllset_hash)
            if current_root and processed:
                new_root = self.kernel.union(current_root, processed[0])
            elif processed:
                new_root = processed[0]
            elif current_root:
                new_root = current_root
            else:
                new_root = self.kernel.absorb(set())
        else:
            new_root = processed[0] if processed else self.kernel.absorb(set())
        
        # Store new root
        self.store.store_hllset(new_root)
        
        # 4. Create new state
        # Note: HRT creation happens via separate evolution system if needed
        new_state = OSState(
            state_hash="",
            root_hllset_hash=new_root.name,
            hrt_hash=self.current_hrt.name if self.current_hrt else "",
            perceptron_states={
                p.perceptron_id: {
                    'absorption_count': p.absorption_count,
                    'last_absorption': p.last_absorption
                }
                for p in self.perceptrons.values()
            },
            pipeline_config={'stages': [s.stage_id for s in self.pipeline]},
            parent_state=self.current_state.state_hash if self.current_state else None
        )
        
        self.current_state = new_state
        return new_state
    
    # -------------------------------------------------------------------------
    # Commit / Persistence
    # -------------------------------------------------------------------------
    
    def commit(self, message: str = "") -> str:
        """
        Commit current state to persistent storage.
        
        This is the "final commit" that saves OS state before processing.
        Also saves current HRT and state HLLSets.
        """
        if not self.current_state:
            raise RuntimeError("No state to commit")
        
        # Store current state HLLSet before clearing CAS
        state_hll = self.get_root()
        if state_hll:
            self.store.store_hllset(state_hll)
        
        # Merge kernel CAS into persistent storage
        root = self.store.merge_kernel_cas(self.kernel)
        self.current_state.root_hllset_hash = root.name
        
        # Ensure HRT is stored
        if self.current_hrt:
            hrt_hash = self.store.store_hrt(self.current_hrt)
            self.current_state.hrt_hash = hrt_hash
        
        # Commit state
        state_hash = self.store.commit(self.current_state)
        
        # Store the root HLLSet again (in case merge changed it)
        merged_root = self.store.get_hllset(root.name)
        if merged_root:
            self.store.store_hllset(merged_root)
        
        # Clear kernel history (undo/redo no longer needed)
        self.kernel.clear_history()
        
        return state_hash
    
    def checkout(self, state_hash: str) -> Optional[OSState]:
        """Checkout previous state."""
        state = self.store.checkout(state_hash)
        if state:
            self.current_state = state
        return state
    
    def get_history(self) -> List[OSState]:
        """Get full state history."""
        return self.store.get_history()
    
    # -------------------------------------------------------------------------
    # Evolution
    # -------------------------------------------------------------------------
    
    def run_evolution(self, max_iterations: int = 100) -> OSState:
        """Run evolution loop."""
        self.evolution.config.max_iterations = max_iterations
        
        final_state = self.evolution.run(
            list(self.perceptrons.values()),
            self.pipeline,
            self.kernel,
            self.store,
            self.current_state
        )
        
        self.current_state = final_state
        return final_state
    
    # -------------------------------------------------------------------------
    # Queries
    # -------------------------------------------------------------------------
    
    def query_similar(self, hllset: HLLSet, threshold: float = 0.8) -> List[Tuple[str, float]]:
        """Query HLLSets similar to given one."""
        return self.kernel.find_similar(hllset, threshold)
    
    # -------------------------------------------------------------------------
    # HRT Queries
    # -------------------------------------------------------------------------
    
    def get_hrt(self) -> Optional[HRT]:
        """Get current HRT."""
        return self.current_hrt
    
    def get_hrt_from_state(self, state_hash: Optional[str] = None) -> Optional[HRT]:
        """Get HRT from specific state."""
        if state_hash is None:
            state = self.current_state
        else:
            state = self.store.states.get(state_hash)
        
        if state:
            return self.store.get_hrt(state.hrt_hash)
        return None
    
    def project_future(self, col_indices: List[int]) -> Optional[np.ndarray]:
        """Project columns to rows using HRT (future projection)."""
        if self.current_hrt:
            return self.current_hrt.project_future(col_indices)
        return None
    
    def project_past(self, row_indices: List[int]) -> Optional[np.ndarray]:
        """Project rows to columns using HRT (past reconstruction)."""
        if self.current_hrt:
            return self.current_hrt.project_past(row_indices)
        return None
    
    def compute_cover(self, hllset: HLLSet) -> Optional[Cover]:
        """Compute optimal cover for HLLSet using current HRT lattice."""
        if self.current_hrt and self.current_hrt.get_lattice():
            return self.current_hrt.get_lattice().compute_cover(hllset, self.kernel)
        return None
    
    def get_basic_hllsets(self) -> List[BasicHLLSet]:
        """Get all basic HLLSets from current HRT."""
        if self.current_hrt and self.current_hrt.get_lattice():
            lattice = self.current_hrt.get_lattice()
            return lattice.row_basic + lattice.col_basic
        return []
    
    def get_root(self) -> Optional[HLLSet]:
        """Get current root HLLSet."""
        if self.current_state:
            return self.store.get_hllset(self.current_state.root_hllset_hash)
        return None
    
    # -------------------------------------------------------------------------
    # Entanglement Operations
    # -------------------------------------------------------------------------
    
    def compute_entanglement(self, hrt1_hash: Optional[str] = None, hrt2_hash: Optional[str] = None) -> Optional[EntanglementMorphism]:
        """
        Compute entanglement morphism between two HRTs.
        
        If hashes not provided, uses current HRT and previous HRT from store.
        """
        # Get HRTs
        if hrt1_hash and hrt2_hash:
            hrt1 = self.store.get_hrt(hrt1_hash)
            hrt2 = self.store.get_hrt(hrt2_hash)
        elif self.current_hrt and self.current_state and self.current_state.parent_state:
            # Use current and parent HRT
            hrt2 = self.current_hrt
            parent_state = self.store.states.get(self.current_state.parent_state)
            if parent_state:
                hrt1 = self.store.get_hrt(parent_state.hrt_hash)
            else:
                return None
        else:
            return None
        
        if not hrt1 or not hrt2:
            return None
        
        # Compute entanglement
        morphism = compute_hrt_entanglement(hrt1, hrt2, self.kernel)
        
        # Store morphism
        self.entanglement_morphisms[morphism.name] = morphism
        
        return morphism
    
    def get_entanglement(self, morphism_name: str) -> Optional[EntanglementMorphism]:
        """Get cached entanglement morphism by name."""
        return self.entanglement_morphisms.get(morphism_name)
    
    def list_entanglements(self) -> List[str]:
        """List all entanglement morphism names."""
        return list(self.entanglement_morphisms.keys())
    
    def compute_entanglement_network(self, hrt_hashes: Optional[List[str]] = None) -> Dict[Tuple[str, str], EntanglementMorphism]:
        """
        Compute entanglement network among multiple HRTs.
        
        If hashes not provided, uses all stored HRTs.
        """
        if hrt_hashes:
            hrts = [self.store.get_hrt(h) for h in hrt_hashes]
            hrts = [h for h in hrts if h]
        else:
            hrts = list(self.store.hrts.values())
        
        if len(hrts) < 2:
            return {}
        
        network = compute_entanglement_network(hrts, self.kernel)
        
        # Store all morphisms
        for morphism in network.values():
            self.entanglement_morphisms[morphism.name] = morphism
        
        return network
    
    def find_entangled_pairs(self, hrt_hash: Optional[str] = None, threshold: float = 0.5) -> List[Tuple[int, int, float]]:
        """
        Find strongly entangled pairs in an HRT.
        
        Returns list of (source_idx, target_idx, strength) tuples.
        """
        if hrt_hash:
            hrt = self.store.get_hrt(hrt_hash)
        else:
            hrt = self.current_hrt
        
        if not hrt:
            return []
        
        # Compute self-entanglement (endomorphism W → W)
        morphism = self.entanglement_engine.compute_self_entanglement(
            hrt.get_lattice(), hrt.name
        )
        
        # Find pairs above threshold
        pairs = []
        for src_idx, matches in morphism.mapping.items():
            for tgt_idx, measurement in matches:
                if measurement.strength >= threshold:
                    pairs.append((src_idx, tgt_idx, measurement.strength))
        
        return sorted(pairs, key=lambda x: x[2], reverse=True)
    
    # -------------------------------------------------------------------------
    # State-to-State Mapping
    # -------------------------------------------------------------------------
    
    def map_to_previous_state(self) -> Optional[StateMapping]:
        """
        Map current state to previous state using entanglement.
        
        Returns complete mapping including:
        - Decompositions of both states
        - Entanglement morphism
        - Trajectory triple (D, R, N)
        """
        return map_mos_states(self)
    
    def map_states(self, 
                   source_state_hash: Optional[str] = None,
                   target_state_hash: Optional[str] = None) -> Optional[StateMapping]:
        """
        Map any two states.
        
        If hashes not provided, uses current and parent state.
        """
        return map_mos_states(self, source_state_hash, target_state_hash)
    
    def get_trajectory(self) -> Optional[TrajectoryTriple]:
        """Get trajectory triple (D, R, N) from current to parent state."""
        mapping = self.map_to_previous_state()
        if mapping:
            return mapping.trajectory
        return None
    
    def decompose_current_state(self):
        """
        Decompose current state HLLSet into optimal cover.
        
        Returns basic HLLSet indices that cover the state.
        """
        if not self.current_hrt or not self.current_state:
            return None
        
        from .state_mapping import StateMapper
        mapper = StateMapper(self.kernel)
        
        state_hll = self.store.get_hllset(self.current_state.root_hllset_hash)
        if not state_hll:
            return None
        
        decomp = mapper.decompose_state(
            state_hll, 
            self.current_hrt, 
            self.current_state.root_hllset_hash
        )
        
        return decomp
    
    # -------------------------------------------------------------------------
    # Statistics
    # -------------------------------------------------------------------------
    
    def stats(self) -> Dict[str, Any]:
        """Get OS statistics."""
        hrt_info = {}
        if self.current_hrt:
            hrt_info = {
                'hrt_name': self.current_hrt.name[:16],
                'hrt_dimension': self.current_hrt.dimension,
                'hrt_am_nonzero': len(self.current_hrt.get_am().get_nonzero_entries()) if self.current_hrt.get_am() else 0,
                'hrt_covers_cached': len(self.current_hrt._covers),
            }
        
        entanglement_info = {
            'entanglement_morphisms': len(self.entanglement_morphisms),
        }
        
        return {
            'runtime': time.time() - self.start_time,
            'processing_cycles': self.processing_cycles,
            'perceptrons': len(self.perceptrons),
            'pipeline_stages': len(self.pipeline),
            'kernel_stats': self.kernel.stats(),
            'persistent_states': len(self.store.states),
            'persistent_hrts': len(self.store.hrts),
            'current_state': self.current_state.state_hash[:16] if self.current_state else None,
            **hrt_info,
            **entanglement_info
        }


# =============================================================================
# SECTION 6: Example Usage
# =============================================================================

def main():
    """Example ManifoldOS usage."""
    print("="*70)
    print("MANIFOLD OS: Evolution Loop and Processing Pipeline")
    print("="*70)
    
    # Create OS
    os = ManifoldOS()
    
    # 1. Add perceptrons
    print("\n1. Adding Perceptrons")
    print("-" * 40)
    
    os.add_perceptron("visual", "camera_01", {"type": "visual", "resolution": "1080p"})
    os.add_perceptron("audio", "microphone_01", {"type": "audio", "sample_rate": 44100})
    os.add_perceptron("tactile", "sensor_01", {"type": "tactile", "sensitivity": "high"})
    
    print(f"Added {len(os.perceptrons)} perceptrons")
    
    # 2. Setup pipeline
    print("\n2. Setup Processing Pipeline")
    print("-" * 40)
    
    os.add_pipeline_stage("merge_sensory", "union")
    os.add_pipeline_stage("refine", "intersection")
    
    print(f"Added {len(os.pipeline)} pipeline stages")
    
    # 3. Process cycle
    print("\n3. Processing Cycle")
    print("-" * 40)
    
    perceptron_data = {
        "visual": {"red", "green", "blue", "yellow"},
        "audio": {"low", "mid", "high", "bass"},
        "tactile": {"soft", "hard", "rough", "smooth"},
    }
    
    state = os.process_cycle(perceptron_data)
    print(f"Processed cycle, new state: {state.state_hash[:16]}...")
    
    root = os.get_root()
    print(f"Root HLLSet: {root}")
    
    # 4. Commit
    print("\n4. Commit to Persistent Storage")
    print("-" * 40)
    
    commit_hash = os.commit("Initial sensory processing")
    print(f"Committed: {commit_hash[:16]}...")
    
    # 5. Stats
    print("\n5. OS Statistics")
    print("-" * 40)
    
    for key, value in os.stats().items():
        print(f"  {key}: {value}")
    
    print("\n" + "="*70)
    print("ManifoldOS ready")
    print("="*70)
    
    return os


if __name__ == "__main__":
    main()
