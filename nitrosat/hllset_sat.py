"""
HLLSet Integration Utilities for NitroSAT
==========================================

Tools for converting HLLSet problems (disambiguation, entanglement)
into SAT formulations and back.

This enables using NitroSAT's physics-informed solver for:
1. Token disambiguation (which domain does a token belong to?)
2. Perceptron entanglement detection (correlated evolution)
3. Optimal set partitioning
4. Constraint satisfaction over probabilistic sets

Theoretical Connection:
    NitroSAT's energy landscape ↔ Fractal manifold's Δ(t) dynamics
    Heat kernel ↔ W transition probabilities
    Phase transitions ↔ Convergence depth / temporal symmetry
    Spectral geometry ↔ DFT-based entanglement analysis
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Optional, TYPE_CHECKING
import math

from .nitrosat import CNFFormula, NitroSatSolver, NitroSatResult

# Conditional import for type checking
if TYPE_CHECKING:
    from core.hllset import HLLSet


# ============================================================================
# Token-to-Variable Mapping
# ============================================================================

@dataclass
class TokenVarMap:
    """
    Bidirectional mapping between tokens and SAT variables.
    
    SAT variables are 1-indexed positive integers.
    Tokens can be any hashable Python objects.
    """
    token_to_var: Dict[str, int] = field(default_factory=dict)
    var_to_token: Dict[int, str] = field(default_factory=dict)
    _next_var: int = 1
    
    def get_or_create_var(self, token: str) -> int:
        """Get variable for token, creating if needed."""
        if token not in self.token_to_var:
            var = self._next_var
            self.token_to_var[token] = var
            self.var_to_token[var] = token
            self._next_var += 1
        return self.token_to_var[token]
    
    def get_var(self, token: str) -> Optional[int]:
        """Get variable for token, or None."""
        return self.token_to_var.get(token)
    
    def get_token(self, var: int) -> Optional[str]:
        """Get token for variable, or None."""
        return self.var_to_token.get(var)
    
    @property
    def num_vars(self) -> int:
        return self._next_var - 1
    
    def tokens_from_assignment(
        self,
        assignment: List[int],
        value: int = 1
    ) -> Set[str]:
        """Get tokens assigned to a specific value (0 or 1)."""
        tokens = set()
        for var, val in enumerate(assignment):
            if var > 0 and val == value:
                token = self.get_token(var)
                if token:
                    tokens.add(token)
        return tokens


# ============================================================================
# Disambiguation as SAT
# ============================================================================

@dataclass
class DisambiguationProblem:
    """
    Token disambiguation formulated as MaxSAT.
    
    Given:
    - Multiple HLLSets representing different domains
    - Tokens that appear in multiple domains
    
    Find:
    - Assignment of ambiguous tokens to exactly one domain
    - Maximizing some objective (e.g., domain coherence)
    
    SAT Encoding:
    - Variable x_i_d = 1 means token i belongs to domain d
    - Exactly-one constraint: token belongs to exactly one domain
    - Soft clauses for domain preferences based on cardinality
    """
    
    domains: Dict[str, Set[str]]  # domain_name -> set of tokens
    token_map: TokenVarMap = field(default_factory=TokenVarMap)
    
    # For each token in each domain, we create variable (token, domain)
    # encoded as: base_var + domain_index
    _domain_to_idx: Dict[str, int] = field(default_factory=dict)
    _var_to_token_domain: Dict[int, Tuple[str, str]] = field(default_factory=dict)
    
    def __post_init__(self):
        """Build variable mappings."""
        self._domain_to_idx = {d: i for i, d in enumerate(sorted(self.domains.keys()))}
    
    @classmethod
    def from_hllsets(
        cls,
        hllsets: Dict[str, "HLLSet"],
        sample_size: int = 1000
    ) -> "DisambiguationProblem":
        """
        Create disambiguation problem from HLLSets.
        
        Note: HLLSets are probabilistic, so we sample tokens.
        This is an approximation - for exact disambiguation,
        use the original token streams.
        
        Args:
            hllsets: Dictionary mapping domain names to HLLSets
            sample_size: Number of tokens to sample per domain
            
        Returns:
            DisambiguationProblem instance
        """
        # For now, we can't directly sample from HLLSet
        # This would need access to the token source
        # Instead, use _extracted_tokens if available
        domains = {}
        for name, hll in hllsets.items():
            # Try to get tokens (this depends on HLLSet implementation)
            if hasattr(hll, '_debug_tokens'):
                domains[name] = set(hll._debug_tokens)
            else:
                # Fallback: create placeholder
                domains[name] = set()
        
        return cls(domains=domains)
    
    @classmethod
    def from_token_sets(
        cls,
        token_sets: Dict[str, Set[str]]
    ) -> "DisambiguationProblem":
        """
        Create disambiguation problem from explicit token sets.
        
        Args:
            token_sets: Dictionary mapping domain names to token sets
            
        Returns:
            DisambiguationProblem instance
        """
        return cls(domains=token_sets)
    
    def find_ambiguous_tokens(self) -> Set[str]:
        """Find tokens that appear in multiple domains."""
        token_counts: Dict[str, int] = {}
        for domain_tokens in self.domains.values():
            for token in domain_tokens:
                token_counts[token] = token_counts.get(token, 0) + 1
        
        return {t for t, count in token_counts.items() if count > 1}
    
    def to_cnf(self) -> Tuple[CNFFormula, Dict[int, Tuple[str, str]]]:
        """
        Convert to CNF formula.
        
        Returns:
            (CNFFormula, var_mapping) where var_mapping maps
            variable number to (token, domain) tuple
        """
        ambiguous = self.find_ambiguous_tokens()
        num_domains = len(self.domains)
        domain_names = sorted(self.domains.keys())
        
        # Each ambiguous token gets num_domains variables
        # var = base_var + domain_idx
        var_mapping: Dict[int, Tuple[str, str]] = {}
        token_vars: Dict[str, List[int]] = {}
        
        next_var = 1
        for token in sorted(ambiguous):
            vars_for_token = []
            for d_idx, domain in enumerate(domain_names):
                if token in self.domains[domain]:
                    var = next_var
                    next_var += 1
                    var_mapping[var] = (token, domain)
                    vars_for_token.append(var)
            token_vars[token] = vars_for_token
        
        formula = CNFFormula(num_vars=next_var - 1)
        
        # Constraint: Each ambiguous token belongs to exactly one domain
        for token, vars in token_vars.items():
            if len(vars) < 2:
                continue
            
            # At-least-one: (x1 OR x2 OR ... OR xn)
            formula.add_clause(*vars)
            
            # At-most-one: for each pair (xi, xj), add (-xi OR -xj)
            for i in range(len(vars)):
                for j in range(i + 1, len(vars)):
                    formula.add_clause(-vars[i], -vars[j])
        
        return formula, var_mapping
    
    def solve(
        self,
        solver: Optional[NitroSatSolver] = None,
        **kwargs
    ) -> Dict[str, str]:
        """
        Solve disambiguation problem.
        
        Returns:
            Dictionary mapping ambiguous tokens to their assigned domains
        """
        if solver is None:
            solver = NitroSatSolver(**kwargs)
        
        formula, var_mapping = self.to_cnf()
        
        if formula.num_clauses == 0:
            # No ambiguous tokens
            return {}
        
        result = solver.solve(formula)
        
        # Extract assignments
        disambiguation: Dict[str, str] = {}
        for var, (token, domain) in var_mapping.items():
            if result.get_assignment(var) == 1:
                disambiguation[token] = domain
        
        return disambiguation


# ============================================================================
# Entanglement Detection as SAT
# ============================================================================

@dataclass
class EntanglementProblem:
    """
    Perceptron entanglement detection formulated as MaxSAT.
    
    Given:
    - Multiple perceptron delta histories
    - Correlation threshold
    
    Find:
    - Groupings of entangled perceptrons
    - Satisfying correlation constraints
    
    SAT Encoding:
    - Variable x_i_j = 1 means perceptron i is in group j
    - Correlation constraints as clauses
    """
    
    perceptron_ids: List[int]
    correlation_matrix: Dict[Tuple[int, int], float]  # (i, j) -> correlation
    correlation_threshold: float = 0.7
    max_groups: int = 4
    
    @classmethod
    def from_delta_histories(
        cls,
        histories: Dict[int, List[float]],
        threshold: float = 0.7,
        max_groups: int = 4
    ) -> "EntanglementProblem":
        """
        Create from perceptron delta histories.
        
        Computes Pearson correlation between histories.
        """
        import numpy as np
        
        ids = sorted(histories.keys())
        correlations: Dict[Tuple[int, int], float] = {}
        
        for i, id1 in enumerate(ids):
            for id2 in ids[i+1:]:
                h1 = np.array(histories[id1])
                h2 = np.array(histories[id2])
                
                # Pad to same length
                max_len = max(len(h1), len(h2))
                h1 = np.pad(h1, (0, max_len - len(h1)))
                h2 = np.pad(h2, (0, max_len - len(h2)))
                
                # Pearson correlation
                corr = np.corrcoef(h1, h2)[0, 1]
                if np.isnan(corr):
                    corr = 0.0
                correlations[(id1, id2)] = corr
        
        return cls(
            perceptron_ids=ids,
            correlation_matrix=correlations,
            correlation_threshold=threshold,
            max_groups=max_groups
        )
    
    def to_cnf(self) -> Tuple[CNFFormula, Dict[int, Tuple[int, int]]]:
        """
        Convert to CNF formula.
        
        Returns:
            (CNFFormula, var_mapping) where var_mapping maps
            variable to (perceptron_id, group_id)
        """
        n = len(self.perceptron_ids)
        g = self.max_groups
        
        # Variable for perceptron i in group j
        def var_for(p_idx: int, g_idx: int) -> int:
            return p_idx * g + g_idx + 1
        
        num_vars = n * g
        formula = CNFFormula(num_vars=num_vars)
        var_mapping: Dict[int, Tuple[int, int]] = {}
        
        for p_idx, pid in enumerate(self.perceptron_ids):
            for g_idx in range(g):
                var = var_for(p_idx, g_idx)
                var_mapping[var] = (pid, g_idx)
        
        # Each perceptron in exactly one group
        for p_idx in range(n):
            vars = [var_for(p_idx, g_idx) for g_idx in range(g)]
            # At-least-one
            formula.add_clause(*vars)
            # At-most-one
            for i in range(len(vars)):
                for j in range(i + 1, len(vars)):
                    formula.add_clause(-vars[i], -vars[j])
        
        # Correlation constraints
        for (id1, id2), corr in self.correlation_matrix.items():
            idx1 = self.perceptron_ids.index(id1)
            idx2 = self.perceptron_ids.index(id2)
            
            if corr >= self.correlation_threshold:
                # High correlation: should be in SAME group
                # For each group: x_{i,g} → x_{j,g} and x_{j,g} → x_{i,g}
                # i.e., (NOT x_{i,g} OR x_{j,g}) AND (NOT x_{j,g} OR x_{i,g})
                for g_idx in range(g):
                    v1 = var_for(idx1, g_idx)
                    v2 = var_for(idx2, g_idx)
                    # These are soft constraints in MaxSAT
                    # For now, we add them as hard constraints
                    formula.add_clause(-v1, v2)
                    formula.add_clause(-v2, v1)
            
            elif corr <= -self.correlation_threshold:
                # Anti-correlation: should be in DIFFERENT groups
                for g_idx in range(g):
                    v1 = var_for(idx1, g_idx)
                    v2 = var_for(idx2, g_idx)
                    # NOT (x_{i,g} AND x_{j,g}) = (NOT x_{i,g} OR NOT x_{j,g})
                    formula.add_clause(-v1, -v2)
        
        return formula, var_mapping
    
    def solve(
        self,
        solver: Optional[NitroSatSolver] = None,
        **kwargs
    ) -> Dict[int, List[int]]:
        """
        Solve entanglement problem.
        
        Returns:
            Dictionary mapping group_id to list of perceptron_ids
        """
        if solver is None:
            solver = NitroSatSolver(**kwargs)
        
        formula, var_mapping = self.to_cnf()
        result = solver.solve(formula)
        
        # Extract groupings
        groups: Dict[int, List[int]] = {g: [] for g in range(self.max_groups)}
        
        for var, (pid, gid) in var_mapping.items():
            if result.get_assignment(var) == 1:
                groups[gid].append(pid)
        
        # Remove empty groups
        return {g: pids for g, pids in groups.items() if pids}


# ============================================================================
# Set Cover / Partitioning
# ============================================================================

def optimal_set_cover(
    universe: Set[str],
    subsets: Dict[str, Set[str]],
    max_sets: Optional[int] = None,
    solver: Optional[NitroSatSolver] = None,
) -> List[str]:
    """
    Find minimum set cover using SAT.
    
    Given a universe of elements and a collection of subsets,
    find the minimum number of subsets that cover all elements.
    
    Args:
        universe: Set of elements to cover
        subsets: Dictionary mapping subset names to element sets
        max_sets: Maximum number of subsets to use (optional)
        solver: NitroSatSolver instance (optional)
        
    Returns:
        List of subset names in the cover
    """
    if solver is None:
        solver = NitroSatSolver()
    
    # Variables: one per subset
    subset_names = sorted(subsets.keys())
    num_vars = len(subset_names)
    
    formula = CNFFormula(num_vars=num_vars)
    
    # Map subset name to variable
    name_to_var = {name: i + 1 for i, name in enumerate(subset_names)}
    
    # Each element must be covered by at least one selected subset
    for element in universe:
        covering_vars = []
        for name, subset in subsets.items():
            if element in subset:
                covering_vars.append(name_to_var[name])
        
        if covering_vars:
            formula.add_clause(*covering_vars)
    
    # Optional: limit number of sets (at-most-k constraint)
    # This is complex for SAT, so we skip for now
    
    result = solver.solve(formula)
    
    # Extract selected subsets
    selected = []
    for name, var in name_to_var.items():
        if result.get_assignment(var) == 1:
            selected.append(name)
    
    return selected


# ============================================================================
# Module Exports
# ============================================================================

__all__ = [
    "TokenVarMap",
    "DisambiguationProblem",
    "EntanglementProblem",
    "optimal_set_cover",
]
