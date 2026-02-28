# NitroSAT Python Wrapper

A high-level Python interface to the NitroSAT physics-informed MaxSAT solver.

## Overview

NitroSAT is a cutting-edge MaxSAT solver that uses physics-inspired techniques:

- **Spectral Geometry**: XOR-Laplacian eigenvector initialization
- **Heat Kernel Smoothing**: Gradient flow with exponential decay
- **Persistent Homology**: Betti number tracking (β₀, β₁)
- **BAHA**: Branch-Aware Holonomy Annealing for phase transitions
- **Dynamic Clause Weighting**: Adaptive clause importance

This wrapper enables integration with fractal_manifold's HLLSet operations for:
- Token disambiguation
- Perceptron entanglement detection
- Optimal set partitioning

## Installation

1. Build the C library:

```bash
cd nitrosat
python build.py
```

2. Use from Python:

```python
from nitrosat import NitroSatSolver, CNFFormula
```

## Quick Start

### Solving a CNF Formula

```python
from nitrosat import CNFFormula, solve

# Create formula: (x1 OR x2) AND (NOT x1 OR x3) AND (x2 OR NOT x3)
formula = CNFFormula(num_vars=3)
formula.add_clause(1, 2)      # x1 OR x2
formula.add_clause(-1, 3)     # NOT x1 OR x3
formula.add_clause(2, -3)     # x2 OR NOT x3

# Solve
result = solve(formula)

print(f"Solved: {result.solved}")
print(f"Satisfaction ratio: {result.satisfaction_ratio:.1%}")
print(f"Solve time: {result.solve_time:.3f}s")

# Get assignments
for var in range(1, formula.num_vars + 1):
    print(f"x{var} = {result.get_assignment(var)}")
```

### Solving from DIMACS File

```python
from nitrosat import solve_file

result = solve_file("problem.cnf", verbose=True)
print(result.to_dict())
```

### Token Disambiguation

```python
from nitrosat import DisambiguationProblem

# Define domains with overlapping tokens
domains = {
    "tech": {"python", "java", "algorithm", "data"},
    "coffee": {"java", "espresso", "beans", "roast"},
    "math": {"algorithm", "theorem", "proof", "data"},
}

# Create and solve
problem = DisambiguationProblem.from_token_sets(domains)
result = problem.solve(verbose=True)

print("Disambiguation:")
for token, domain in result.items():
    print(f"  '{token}' → {domain}")
```

### Perceptron Entanglement Detection

```python
from nitrosat import EntanglementProblem

# Delta histories from perceptron swarm
histories = {
    0: [37.0, 12, 44, 15, 30, 28, ...],
    1: [35.0, 15, 42, 18, 28, 25, ...],  # Similar to 0
    2: [10.0, 5, 8, 3, 4, 2, ...],       # Different pattern
}

# Find entangled groups
problem = EntanglementProblem.from_delta_histories(
    histories,
    threshold=0.7
)
groups = problem.solve()

print("Entanglement groups:")
for group_id, perceptron_ids in groups.items():
    print(f"  Group {group_id}: {perceptron_ids}")
```

## API Reference

### Core Classes

- `NitroSatSolver`: Main solver class with configurable options
- `CNFFormula`: CNF formula builder and DIMACS I/O
- `NitroSatResult`: Solution result with topology info
- `TopologyInfo`: Persistent homology data (Betti numbers)

### HLLSet Integration

- `DisambiguationProblem`: Token disambiguation as MaxSAT
- `EntanglementProblem`: Correlation-based grouping as SAT
- `optimal_set_cover`: Set cover using SAT

## Theoretical Connection to fractal_manifold

| NitroSAT | Fractal Manifold |
|----------|------------------|
| Energy E(t) | Δ(t) = \|N\| - \|D\| |
| Energy minimum | Noether equilibrium (Δ → 0) |
| Phase transition | Convergence depth |
| Heat kernel | W transition probabilities |
| Spectral geometry | DFT entanglement analysis |
| β₀, β₁ (Betti) | Connected components, cycles |

## Citation

```bibtex
@software{sethurathienam_iyer_2026_18753235,
    author = {Sethurathienam Iyer},
    title = {NitroSAT: A Physics-Informed MaxSAT Solver},
    year = 2026,
    publisher = {Zenodo},
    doi = {10.5281/zenodo.18753235}
}
```

## License

The wrapper code is part of fractal_manifold (Apache 2.0).
NitroSAT core is by Sethu Iyer under Apache 2.0 license.
