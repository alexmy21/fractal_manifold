"""
Renormalization Group Module

Implements renormalization group flow analysis and fixed point identification.
"""

from typing import Any, Dict, List, Optional, Callable, Tuple
from dataclasses import dataclass, field
import numpy as np


@dataclass
class FixedPoint:
    """Represents a fixed point in RG flow."""
    name: str
    couplings: Dict[str, float]
    stability: str = "unknown"  # stable, unstable, saddle
    scaling_dimensions: Dict[str, float] = field(default_factory=dict)


@dataclass
class RGFlow:
    """Represents a renormalization group flow trajectory."""
    start_couplings: Dict[str, float]
    trajectory: List[Dict[str, float]] = field(default_factory=list)
    fixed_point: Optional[FixedPoint] = None


class RenormalizationGroup:
    """
    Implements renormalization group analysis.
    
    Computes RG flow, identifies fixed points, and determines scaling dimensions
    and critical exponents.
    """
    
    def __init__(self):
        self.beta_functions: Dict[str, Callable] = {}
        self.fixed_points: List[FixedPoint] = []
        self.flows: List[RGFlow] = []
        self.scaling_dimension_operators: Dict[str, float] = {}
    
    def add_beta_function(self, coupling_name: str, beta_func: Callable) -> None:
        """
        Add a beta function for a coupling constant.
        
        Args:
            coupling_name: Name of the coupling constant
            beta_func: Beta function β(g) = dg/d(log μ)
        """
        self.beta_functions[coupling_name] = beta_func
    
    def compute_flow(self, 
                    initial_couplings: Dict[str, float],
                    n_steps: int = 100,
                    step_size: float = 0.01) -> RGFlow:
        """
        Compute RG flow from initial couplings.
        
        Args:
            initial_couplings: Initial values of coupling constants
            n_steps: Number of RG flow steps
            step_size: Step size in log(scale)
            
        Returns:
            RGFlow object containing the trajectory
        """
        flow = RGFlow(start_couplings=initial_couplings.copy())
        couplings = initial_couplings.copy()
        
        for _ in range(n_steps):
            flow.trajectory.append(couplings.copy())
            
            # Update couplings using beta functions
            new_couplings = {}
            for name, value in couplings.items():
                if name in self.beta_functions:
                    beta = self.beta_functions[name](couplings)
                    new_couplings[name] = value + step_size * beta
                else:
                    new_couplings[name] = value
            
            couplings = new_couplings
            
            # Check if reached fixed point
            if self._is_at_fixed_point(couplings):
                fp = self._identify_fixed_point(couplings)
                flow.fixed_point = fp
                break
        
        self.flows.append(flow)
        return flow
    
    def find_fixed_points(self, 
                         coupling_ranges: Dict[str, Tuple[float, float]],
                         n_samples: int = 10) -> List[FixedPoint]:
        """
        Find fixed points by solving β(g*) = 0.
        
        Args:
            coupling_ranges: Ranges for each coupling to search
            n_samples: Number of initial points to try
            
        Returns:
            List of identified fixed points
        """
        found_fixed_points = []
        
        # Simple grid search (can be improved with optimization)
        for _ in range(n_samples):
            # Sample random initial point
            initial = {
                name: np.random.uniform(low, high)
                for name, (low, high) in coupling_ranges.items()
            }
            
            # Run flow to see if it converges to fixed point
            flow = self.compute_flow(initial, n_steps=50)
            
            if flow.fixed_point and flow.fixed_point not in found_fixed_points:
                found_fixed_points.append(flow.fixed_point)
                self.fixed_points.append(flow.fixed_point)
        
        return found_fixed_points
    
    def compute_scaling_dimensions(self, fixed_point: FixedPoint) -> Dict[str, float]:
        """
        Compute scaling dimensions at a fixed point.
        
        Scaling dimensions are eigenvalues of the stability matrix.
        
        Args:
            fixed_point: The fixed point to analyze
            
        Returns:
            Dictionary of operator names to scaling dimensions
        """
        # Compute Jacobian of beta functions at fixed point
        epsilon = 1e-6
        coupling_names = list(fixed_point.couplings.keys())
        n = len(coupling_names)
        
        jacobian = np.zeros((n, n))
        
        for i, name_i in enumerate(coupling_names):
            for j, name_j in enumerate(coupling_names):
                # Numerical derivative
                couplings_plus = fixed_point.couplings.copy()
                couplings_plus[name_j] += epsilon
                
                beta_plus = self.beta_functions[name_i](couplings_plus)
                beta_0 = self.beta_functions[name_i](fixed_point.couplings)
                
                jacobian[i, j] = (beta_plus - beta_0) / epsilon
        
        # Eigenvalues give scaling dimensions
        eigenvalues = np.linalg.eigvals(jacobian)
        
        for i, name in enumerate(coupling_names):
            dimension = float(eigenvalues[i].real)
            fixed_point.scaling_dimensions[name] = dimension
            self.scaling_dimension_operators[f"{name}_at_{fixed_point.name}"] = dimension
        
        return fixed_point.scaling_dimensions
    
    def classify_fixed_point_stability(self, fixed_point: FixedPoint) -> str:
        """
        Classify fixed point as UV or IR based on stability.
        
        Args:
            fixed_point: The fixed point to classify
            
        Returns:
            Classification string
        """
        if not fixed_point.scaling_dimensions:
            self.compute_scaling_dimensions(fixed_point)
        
        # Count relevant (negative eigenvalue) vs irrelevant (positive) operators
        relevant = sum(1 for d in fixed_point.scaling_dimensions.values() if d < 0)
        irrelevant = sum(1 for d in fixed_point.scaling_dimensions.values() if d > 0)
        
        if relevant > irrelevant:
            fixed_point.stability = "UV_attractive"
        elif irrelevant > relevant:
            fixed_point.stability = "IR_attractive"
        else:
            fixed_point.stability = "saddle"
        
        return fixed_point.stability
    
    def compute_critical_exponents(self, fixed_point: FixedPoint) -> Dict[str, float]:
        """
        Compute critical exponents from scaling dimensions.
        
        Args:
            fixed_point: The critical fixed point
            
        Returns:
            Dictionary of critical exponents
        """
        if not fixed_point.scaling_dimensions:
            self.compute_scaling_dimensions(fixed_point)
        
        critical_exponents = {}
        
        # Standard relationships between exponents and dimensions
        # ν = 1/y_t where y_t is thermal eigenvalue
        # η = 2 - γ where γ is anomalous dimension
        
        for op_name, dimension in fixed_point.scaling_dimensions.items():
            if "thermal" in op_name or "temperature" in op_name:
                critical_exponents["nu"] = 1.0 / abs(dimension) if dimension != 0 else 0
            elif "field" in op_name:
                critical_exponents["eta"] = 2.0 - dimension
        
        return critical_exponents
    
    def analyze_universality_class(self, fixed_point: FixedPoint) -> Dict[str, Any]:
        """
        Determine universality class from fixed point properties.
        
        Args:
            fixed_point: The fixed point to analyze
            
        Returns:
            Dictionary describing the universality class
        """
        exponents = self.compute_critical_exponents(fixed_point)
        
        return {
            "fixed_point": fixed_point.name,
            "critical_exponents": exponents,
            "scaling_dimensions": fixed_point.scaling_dimensions,
            "stability": fixed_point.stability,
            "dimension": len(fixed_point.couplings)
        }
    
    def compute_anomalous_dimension(self, operator_name: str,
                                   fixed_point: FixedPoint) -> float:
        """
        Compute anomalous dimension of an operator at a fixed point.
        
        Args:
            operator_name: Name of the operator
            fixed_point: The fixed point
            
        Returns:
            Anomalous dimension γ
        """
        # Anomalous dimension is deviation from canonical dimension
        # γ = Δ - Δ_0 where Δ is scaling dimension, Δ_0 is canonical
        
        if operator_name in fixed_point.scaling_dimensions:
            # Simplified: assume canonical dimension is 1
            canonical_dim = 1.0
            actual_dim = fixed_point.scaling_dimensions[operator_name]
            return actual_dim - canonical_dim
        
        return 0.0
    
    def _is_at_fixed_point(self, couplings: Dict[str, float], 
                          tolerance: float = 1e-4) -> bool:
        """Check if current couplings are at a fixed point."""
        for name, value in couplings.items():
            if name in self.beta_functions:
                beta = self.beta_functions[name](couplings)
                if abs(beta) > tolerance:
                    return False
        return True
    
    def _identify_fixed_point(self, couplings: Dict[str, float]) -> FixedPoint:
        """Create a FixedPoint object from couplings."""
        # Check if this fixed point already exists
        for fp in self.fixed_points:
            if self._couplings_close(fp.couplings, couplings):
                return fp
        
        # Create new fixed point
        fp = FixedPoint(
            name=f"FP_{len(self.fixed_points)}",
            couplings=couplings.copy()
        )
        self.classify_fixed_point_stability(fp)
        return fp
    
    def _couplings_close(self, c1: Dict[str, float], c2: Dict[str, float],
                        tolerance: float = 1e-3) -> bool:
        """Check if two coupling configurations are close."""
        if set(c1.keys()) != set(c2.keys()):
            return False
        
        for name in c1.keys():
            if abs(c1[name] - c2[name]) > tolerance:
                return False
        return True
    
    def get_rg_summary(self) -> Dict[str, Any]:
        """Get summary of RG analysis."""
        return {
            "beta_functions": list(self.beta_functions.keys()),
            "fixed_points": [
                {
                    "name": fp.name,
                    "couplings": fp.couplings,
                    "stability": fp.stability,
                    "scaling_dimensions": fp.scaling_dimensions
                }
                for fp in self.fixed_points
            ],
            "flows_computed": len(self.flows),
            "operators": list(self.scaling_dimension_operators.keys())
        }
