#!/usr/bin/env python3
"""
NitroSAT Test Examples
======================

Test suite demonstrating NitroSAT integration with fractal_manifold.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from nitrosat import (
    NitroSatSolver,
    CNFFormula,
    DisambiguationProblem,
    EntanglementProblem,
    solve,
)


def test_basic_sat():
    """Test basic SAT solving."""
    print("\n" + "="*60)
    print("TEST 1: Basic SAT Solving")
    print("="*60)
    
    # Simple satisfiable formula
    # (x1 OR x2) AND (NOT x1 OR x3) AND (x2 OR NOT x3)
    formula = CNFFormula(num_vars=3)
    formula.add_clause(1, 2)      # x1 OR x2
    formula.add_clause(-1, 3)     # NOT x1 OR x3
    formula.add_clause(2, -3)     # x2 OR NOT x3
    
    print(f"\nFormula (DIMACS):\n{formula.to_dimacs()}")
    
    solver = NitroSatSolver(verbose=False)
    print(f"\nNitroSAT version: {solver.version}")
    
    result = solver.solve(formula)
    
    print(f"\nResult:")
    print(f"  Solved: {result.solved}")
    print(f"  Satisfied: {result.satisfied}/{result.num_clauses}")
    print(f"  Solve time: {result.solve_time:.4f}s")
    print(f"  Topology:")
    print(f"    β₀: {result.topology.initial_beta0} → {result.topology.final_beta0}")
    print(f"    β₁: {result.topology.initial_beta1} → {result.topology.final_beta1}")
    
    print(f"\nAssignments:")
    for v in range(1, formula.num_vars + 1):
        print(f"  x{v} = {result.get_assignment(v)}")
    
    # Verify solution
    is_valid = True
    for clause in formula.clauses:
        satisfied = False
        for lit in clause:
            var = abs(lit)
            val = result.get_assignment(var)
            if (lit > 0 and val == 1) or (lit < 0 and val == 0):
                satisfied = True
                break
        if not satisfied:
            is_valid = False
            print(f"  ✗ Clause {clause} not satisfied!")
    
    if is_valid:
        print("\n✓ Solution verified!")
    
    return result.solved


def test_harder_sat():
    """Test a harder SAT problem."""
    print("\n" + "="*60)
    print("TEST 2: Harder SAT Problem (20 vars, 60 clauses)")
    print("="*60)
    
    import random
    random.seed(42)
    
    num_vars = 20
    num_clauses = 60
    
    formula = CNFFormula(num_vars=num_vars)
    
    # Generate random 3-SAT clauses
    for _ in range(num_clauses):
        vars = random.sample(range(1, num_vars + 1), 3)
        clause = [v if random.random() > 0.5 else -v for v in vars]
        formula.add_clause(*clause)
    
    print(f"Generated random 3-SAT: {num_vars} vars, {num_clauses} clauses")
    
    solver = NitroSatSolver(max_steps=2000, verbose=False)
    result = solver.solve(formula)
    
    print(f"\nResult:")
    print(f"  Solved: {result.solved}")
    print(f"  Satisfied: {result.satisfied}/{result.num_clauses}")
    print(f"  Satisfaction ratio: {result.satisfaction_ratio:.1%}")
    print(f"  Solve time: {result.solve_time:.4f}s")
    print(f"  Topology events: {result.topology.persistence_events}")
    
    return result.solved


def test_disambiguation():
    """Test token disambiguation."""
    print("\n" + "="*60)
    print("TEST 3: Token Disambiguation")
    print("="*60)
    
    # Domains with overlapping tokens
    domains = {
        "tech": {"python", "java", "algorithm", "data", "code", "function"},
        "coffee": {"java", "espresso", "beans", "roast", "brew", "aroma"},
        "math": {"algorithm", "theorem", "proof", "data", "equation", "set"},
        "music": {"set", "note", "key", "scale", "chord", "beat"},
    }
    
    print("\nDomains:")
    for name, tokens in domains.items():
        print(f"  {name}: {tokens}")
    
    problem = DisambiguationProblem.from_token_sets(domains)
    
    ambiguous = problem.find_ambiguous_tokens()
    print(f"\nAmbiguous tokens: {ambiguous}")
    
    result = problem.solve(verbose=False)
    
    print(f"\nDisambiguation result:")
    for token, domain in sorted(result.items()):
        print(f"  '{token}' → {domain}")
    
    return len(result) > 0


def test_entanglement():
    """Test perceptron entanglement detection."""
    print("\n" + "="*60)
    print("TEST 4: Perceptron Entanglement Detection")
    print("="*60)
    
    # Simulated delta histories
    # Group 1: correlated (similar patterns)
    # Group 2: different pattern
    histories = {
        0: [37.0, 12, 44, 15, 30, 28, 15, 38, 7, 24],
        1: [35.0, 15, 42, 18, 28, 25, 18, 35, 10, 22],  # Similar to 0
        2: [38.0, 10, 46, 12, 32, 30, 12, 40, 5, 26],   # Similar to 0
        3: [10.0, 5, 8, 3, 4, 2, 5, 6, 2, 3],           # Different
        4: [12.0, 4, 10, 2, 5, 3, 4, 8, 1, 4],          # Similar to 3
    }
    
    print("\nDelta histories:")
    for pid, hist in histories.items():
        print(f"  P{pid}: {hist[:5]}...")
    
    problem = EntanglementProblem.from_delta_histories(
        histories,
        threshold=0.7,
        max_groups=3
    )
    
    print(f"\nCorrelations (threshold=0.7):")
    for (i, j), corr in sorted(problem.correlation_matrix.items()):
        marker = "✓" if abs(corr) >= 0.7 else " "
        print(f"  {marker} P{i}-P{j}: {corr:.3f}")
    
    groups = problem.solve(verbose=False)
    
    print(f"\nEntanglement groups:")
    for gid, pids in sorted(groups.items()):
        print(f"  Group {gid}: {pids}")
    
    return len(groups) > 0


def test_dimacs_roundtrip():
    """Test DIMACS format read/write."""
    print("\n" + "="*60)
    print("TEST 5: DIMACS Roundtrip")
    print("="*60)
    
    original = CNFFormula(num_vars=5)
    original.add_clauses([
        [1, -2, 3],
        [-1, 2, -3],
        [2, 3, 4],
        [-3, 4, 5],
        [1, -4, 5],
    ])
    
    dimacs = original.to_dimacs()
    print(f"\nOriginal DIMACS:\n{dimacs}")
    
    parsed = CNFFormula.from_dimacs(dimacs)
    
    print(f"\nParsed: {parsed.num_vars} vars, {parsed.num_clauses} clauses")
    
    # Verify
    assert parsed.num_vars == original.num_vars
    assert parsed.num_clauses == original.num_clauses
    for i, clause in enumerate(parsed.clauses):
        assert clause == original.clauses[i], f"Clause {i} mismatch"
    
    print("✓ Roundtrip successful!")
    return True


def main():
    """Run all tests."""
    print("\n" + "#"*60)
    print("# NitroSAT Python Wrapper Tests")
    print("# Based on NitroSAT by Sethu Iyer")
    print("#"*60)
    
    results = {
        "Basic SAT": test_basic_sat(),
        "Harder SAT": test_harder_sat(),
        "Disambiguation": test_disambiguation(),
        "Entanglement": test_entanglement(),
        "DIMACS Roundtrip": test_dimacs_roundtrip(),
    }
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for test, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {test}")
    
    all_passed = all(results.values())
    print(f"\n{'All tests passed!' if all_passed else 'Some tests failed.'}")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
