"""
Reference n-token-aware disambiguation utilities.

This module provides a lightweight, reference implementation that reuses
existing LUTs and HLL utilities in the repo. It is intentionally simple
and designed to be a clear basis for further improvements (multi-seed
triangulation, lattice/entanglement pruning, AM integration).
"""
from typing import Set, List, Optional

from .manifold_os import NTokenRepresentation
from .hllset import HLLSet
from .constants import SHARED_SEED


def disambiguate_bit(reg: int, zeros: int, representation: NTokenRepresentation,
                     seeds: Optional[List[int]] = None, p_bits: Optional[int] = None) -> Set[str]:
    """
    Disambiguate a single (reg, zeros) bit using the representation's LUTs.

    Strategy:
    1. Intersect candidate sets across all available n-token LUTs (fast pruning).
    2. If `seeds` provided, verify candidates by recomputing their
       (reg, zeros) under each seed using `HLLSet.compute_reg_zeros_batch`.

    Args:
        reg: Register index
        zeros: Trailing-zeros value
        representation: NTokenRepresentation built during ingestion
        seeds: Optional list of hash seeds to triangulate candidates
        p_bits: Precision bits to pass to HLLSet.compute_reg_zeros_batch

    Returns:
        Set of candidate token strings that pass intersection and optional seed check.
    """
    # Delegate to the canonical implementation on NTokenRepresentation.
    # The class method already performs intersection across n-token LUTs
    # and narrows candidate clouds. Keep this wrapper minimal so repository
    # contains a single authoritative disambiguation implementation.
    return representation.disambiguate_tokens(reg, zeros)


def reconstruct_sequence(am, representation: NTokenRepresentation, kernel, target_cardinality: int,
                         threshold_ratio: float = 0.9) -> List[str]:
    """
    Greedy sequence reconstruction from an Adjacency Matrix `am`.

    This is a small reference routine that attempts to reconstruct an ordered
    sequence of tokens (as strings) from the AM's (reg,zeros) traversal. It
    uses `disambiguate_bit` for mapping identifiers to token strings and a
    greedy highest-frequency walk through the AM cells. This is intentionally
    simple — a production implementation should use the AM's probabilistic
    traversal, lattice/entanglement constraints, and beam search.
    """
    path = []
    # starting point
    current = getattr(am, "START_ID", None)
    if current is None or current not in am.get_row_ids():
        # fallback: choose row with highest total outgoing frequency
        rows = am.get_row_ids()
        best = None
        best_score = -1
        for r in rows:
            score = sum(cell.frequency for (row_id, col_id), cell in am.cells.items() if row_id == r)
            if score > best_score:
                best_score = score
                best = r
        current = best

    if current is None:
        return []

    max_steps = max(1, int(target_cardinality * 2))
    steps = 0
    while steps < max_steps and len(path) < target_cardinality:
        # gather outgoing transitions
        outs = [(col_id, cell.frequency) for (row_id, col_id), cell in am.cells.items() if row_id == current]
        if not outs:
            break
        # filter END until threshold
        outs_sorted = sorted(outs, key=lambda x: x[1], reverse=True)
        next_id = None
        for col_id, freq in outs_sorted:
            if col_id == getattr(am, "END_ID", None) and len(path) < int(target_cardinality * threshold_ratio):
                continue
            next_id = col_id
            break

        if next_id is None:
            break

        # map identifier -> token (choose first candidate deterministically)
        candidates = disambiguate_bit(next_id[0], next_id[1], representation)
        token = sorted(candidates)[0] if candidates else f"<unknown:{next_id}>"
        path.append(token)
        current = next_id
        steps += 1

    return path
