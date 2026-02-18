"""
Example: Analyzing a Quantum System

This example demonstrates how to use the Fractal Manifold framework
to analyze a quantum system and answer the key questions.
"""

import numpy as np
from fractal_manifold import PhenomenonPredictor


def main():
    # Initialize the predictor
    predictor = PhenomenonPredictor()
    
    print("=" * 60)
    print("Fractal Manifold - Categorical Framework Demo")
    print("=" * 60)
    print()
    
    # Example 1: Analyze a simple quantum system
    print("Example 1: Bell State Analysis")
    print("-" * 60)
    
    # Create a Bell state (maximally entangled)
    bell_state = np.array([1, 0, 0, 1]) / np.sqrt(2)
    
    # Analyze the phenomenon
    analysis = predictor.analyze_phenomenon(
        phenomenon_name="bell_state",
        data=bell_state,
        subsystems=["qubit_A", "qubit_B"],
        analyze_entanglement=True
    )
    
    print("\n1. What does this correspond to in the categorical framework?")
    print(predictor.answer_categorical_question("bell_state"))
    
    print("\n2. Does it reveal entanglement?")
    if analysis.entanglement_structure:
        print(f"   Entanglement structure: {analysis.entanglement_structure.get('structure')}")
        measures = analysis.entanglement_structure.get('measures', [])
        if measures:
            print(f"   Entanglement entropy: {measures[0].get('value', 0):.4f}")
    
    print("\n" + "=" * 60)
    print()
    
    # Example 2: Symmetry and conservation laws
    print("Example 2: Symmetry and Conservation Analysis")
    print("-" * 60)
    
    # Define a simple transformation (identity for demonstration)
    def time_translation(x):
        return x
    
    # Analyze with symmetry checking
    analysis2 = predictor.analyze_phenomenon(
        phenomenon_name="symmetric_system",
        data=np.array([1, 2, 3, 4]),
        check_discrete=True,
        discrete_transformations=[time_translation],
        check_continuous=True,
        lagrangian=lambda x: np.sum(x**2),
        parameter_name="time"
    )
    
    print("\n1. Does it reveal new symmetries?")
    print(predictor.answer_symmetry_question("symmetric_system"))
    
    print("\n2. Does it reveal new conservation laws?")
    print(predictor.answer_conservation_question("symmetric_system"))
    
    print("\n" + "=" * 60)
    print()
    
    # Example 3: RG Flow Analysis
    print("Example 3: Renormalization Group Flow")
    print("-" * 60)
    
    # Define simple beta functions
    def beta_g(couplings):
        g = couplings.get("g", 0)
        return -g**2  # Simple example: g flows to 0
    
    # Analyze RG flow
    analysis3 = predictor.analyze_phenomenon(
        phenomenon_name="flowing_system",
        data=None,
        analyze_rg_flow=True,
        beta_functions={"g": beta_g},
        initial_couplings={"g": 1.0}
    )
    
    print("\n1. How does it fit into the RG flow?")
    if analysis3.rg_flow_data:
        print(f"   Flow computed: {analysis3.rg_flow_data.get('flow_computed')}")
        fixed_points = analysis3.rg_flow_data.get('fixed_points', [])
        if fixed_points:
            fp = fixed_points[0]
            print(f"   Fixed point found: {fp['name']}")
            print(f"   Couplings at FP: {fp['couplings']}")
            print(f"   Stability: {fp['stability']}")
    
    print("\n" + "=" * 60)
    print()
    
    # Get complete summary
    print("Complete Summary for 'bell_state':")
    print("-" * 60)
    summary = predictor.get_complete_summary("bell_state")
    
    print(f"\nPhenomenon: {summary['phenomenon']}")
    print(f"Symmetries detected: {len(summary['symmetries'])}")
    print(f"Conservation laws: {len(summary['conservation_laws'])}")
    
    if summary['predictions']:
        print("\nPredictions:")
        for key, value in summary['predictions'].items():
            if value:
                print(f"  {key}: {value}")
    
    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
