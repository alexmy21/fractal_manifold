# Fractal Manifold

A practical system with a rigorous mathematical foundation that **predicts** rather than just describes. This framework provides tools to analyze new phenomena through categorical theory, symmetry analysis, entanglement detection, conservation laws, spectral sequences, and renormalization group flows.

## Overview

When you encounter a new phenomenon in your system, the Fractal Manifold framework allows you to ask:

- **What does this correspond to in the categorical framework?**
- **Does it reveal a new symmetry, a new entanglement, a new conservation law?**
- **How does it fit into the spectral sequence or the renormalization group flow?**

## Features

### Core Components

1. **Categorical Framework** (`categorical.py`)
   - Define and work with mathematical categories
   - Create functors between categories
   - Natural transformations
   - Map phenomena to categorical structures

2. **Symmetry Analysis** (`symmetry.py`)
   - Detect discrete symmetries (reflections, rotations, permutations)
   - Analyze continuous symmetries (Lie groups)
   - Identify gauge symmetries
   - Classify symmetry breaking patterns
   - Find conserved charges via Noether's theorem

3. **Entanglement Analysis** (`entanglement.py`)
   - Compute von Neumann entropy
   - Detect bipartite and multipartite entanglement
   - Calculate mutual information
   - Compute negativity measures
   - Analyze entanglement spectrum

4. **Conservation Laws** (`conservation.py`)
   - Identify conservation laws from symmetries (Noether's theorem)
   - Check energy, momentum, and charge conservation
   - Detect emergent conservation laws
   - Approximate conservation analysis

5. **Spectral Sequences** (`spectral.py`)
   - Compute spectral sequence pages
   - Initialize from filtered complexes
   - Compute to convergence
   - Extract limit terms and total homology

6. **Renormalization Group** (`renormalization.py`)
   - Define beta functions
   - Compute RG flows
   - Find fixed points
   - Calculate scaling dimensions and critical exponents
   - Analyze universality classes

7. **Unified Predictor** (`predictor.py`)
   - Integrate all analysis tools
   - Answer key questions about phenomena
   - Generate predictions from mathematical structure

## Installation

```bash
pip install -e .
```

Or with development dependencies:

```bash
pip install -e ".[dev]"
```

## Quick Start

```python
import numpy as np
from fractal_manifold import PhenomenonPredictor

# Initialize the predictor
predictor = PhenomenonPredictor()

# Analyze a quantum Bell state
bell_state = np.array([1, 0, 0, 1]) / np.sqrt(2)

analysis = predictor.analyze_phenomenon(
    phenomenon_name="bell_state",
    data=bell_state,
    subsystems=["qubit_A", "qubit_B"],
    analyze_entanglement=True
)

# Ask questions about the phenomenon
print(predictor.answer_categorical_question("bell_state"))
print(predictor.answer_symmetry_question("bell_state"))

# Get complete summary
summary = predictor.get_complete_summary("bell_state")
print(summary)
```

## Examples

See the `examples/` directory for detailed examples:

- `quantum_analysis.py` - Analyzing quantum systems with entanglement
- Complete demonstrations of symmetry and conservation law detection
- RG flow analysis examples

## Usage

### Analyzing a Phenomenon

```python
from fractal_manifold import PhenomenonPredictor

predictor = PhenomenonPredictor()

# Analyze with symmetry checking
analysis = predictor.analyze_phenomenon(
    phenomenon_name="my_system",
    data=my_data,
    check_discrete=True,
    discrete_transformations=[transform1, transform2],
    check_continuous=True,
    lagrangian=my_lagrangian,
    parameter_name="time"
)
```

### Working with Categories

```python
from fractal_manifold import Category, Morphism, Functor

# Create a category
cat = Category(name="my_category")
cat.add_object("A")
cat.add_object("B")

# Add morphisms
f = Morphism(source="A", target="B", name="f")
cat.add_morphism(f)

# Create functors between categories
functor = Functor(name="F", source_category=cat1, target_category=cat2)
functor.add_object_mapping("A", "A'")
```

### Symmetry Analysis

```python
from fractal_manifold import SymmetryAnalyzer

analyzer = SymmetryAnalyzer()

# Detect discrete symmetry
sym = analyzer.analyze_discrete_symmetry(data, transformation_func)

# Analyze continuous symmetry
sym = analyzer.analyze_continuous_symmetry(lagrangian, "time")

# Detect gauge symmetry
gauge_syms = analyzer.detect_gauge_symmetry(field_config)
```

### RG Flow Analysis

```python
from fractal_manifold import RenormalizationGroup

rg = RenormalizationGroup()

# Define beta function
def beta_g(couplings):
    g = couplings["g"]
    return -g**2  # Example: asymptotic freedom

rg.add_beta_function("g", beta_g)

# Compute flow
flow = rg.compute_flow(initial_couplings={"g": 1.0}, n_steps=100)

# Find fixed points
fixed_points = rg.find_fixed_points({"g": (0.0, 2.0)})
```

## Testing

Run the test suite:

```bash
pytest tests/ -v
```

## Mathematical Background

This framework integrates several areas of mathematics and physics:

- **Category Theory**: Provides the abstract framework for relating different phenomena
- **Group Theory & Lie Algebras**: Used for symmetry analysis
- **Quantum Information**: Entanglement measures and quantum correlations
- **Noether's Theorem**: Links symmetries to conservation laws
- **Algebraic Topology**: Spectral sequences for computing invariants
- **Renormalization Group**: Understanding scale-dependent behavior and universality

## License

MIT License - see LICENSE file for details

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues.

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{fractal_manifold,
  title = {Fractal Manifold: A Categorical Framework for Predictive Analysis},
  author = {Fractal Manifold Contributors},
  year = {2026},
  url = {https://github.com/alexmy21/fractal_manifold}
}
```
