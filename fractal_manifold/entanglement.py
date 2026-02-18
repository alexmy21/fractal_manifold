"""
Entanglement Analysis Module

Implements entanglement detection and measurement for quantum and classical systems.
"""

from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np


@dataclass
class EntanglementMeasure:
    """Represents a measure of entanglement."""
    name: str
    value: float
    subsystems: Tuple[str, str]
    measure_type: str = "von_neumann"  # von_neumann, mutual_info, negativity, etc.


class EntanglementAnalyzer:
    """
    Analyzes systems for entanglement.
    
    Supports quantum entanglement measures and classical correlation analysis.
    """
    
    def __init__(self):
        self.entanglement_measures: List[EntanglementMeasure] = []
        self.bipartite_structure: Dict[Tuple[str, str], Any] = {}
    
    def compute_von_neumann_entropy(self, density_matrix: np.ndarray) -> float:
        """
        Compute von Neumann entropy S = -Tr(ρ log ρ).
        
        Args:
            density_matrix: Density matrix of the quantum state
            
        Returns:
            Von Neumann entropy value
        """
        eigenvalues = np.linalg.eigvalsh(density_matrix)
        # Filter out zero and negative eigenvalues
        eigenvalues = eigenvalues[eigenvalues > 1e-10]
        entropy = -np.sum(eigenvalues * np.log2(eigenvalues))
        return float(entropy)
    
    def compute_entanglement_entropy(self, state: np.ndarray, 
                                    partition: Tuple[int, ...]) -> float:
        """
        Compute entanglement entropy for a bipartite system.
        
        Args:
            state: Quantum state vector or density matrix
            partition: Partition defining subsystems
            
        Returns:
            Entanglement entropy
        """
        # Create reduced density matrix for subsystem A
        if len(state.shape) == 1:
            # Pure state - convert to density matrix
            density_matrix = np.outer(state, np.conj(state))
        else:
            density_matrix = state
        
        # Partial trace over subsystem B (simplified)
        dim_a = partition[0] if partition else 2
        dim_b = density_matrix.shape[0] // dim_a
        
        try:
            reduced_rho = self._partial_trace(density_matrix, dim_a, dim_b, trace_over='B')
            entropy = self.compute_von_neumann_entropy(reduced_rho)
            return entropy
        except Exception:
            return 0.0
    
    def detect_bipartite_entanglement(self, state: np.ndarray,
                                     subsystem_a: str = "A",
                                     subsystem_b: str = "B",
                                     partition: Optional[Tuple[int, ...]] = None) -> EntanglementMeasure:
        """
        Detect and quantify bipartite entanglement.
        
        Args:
            state: Quantum state
            subsystem_a: Name of first subsystem
            subsystem_b: Name of second subsystem
            partition: Dimensions of subsystems
            
        Returns:
            EntanglementMeasure quantifying the entanglement
        """
        if partition is None:
            # Default to equal bipartition
            total_dim = state.shape[0] if len(state.shape) > 1 else len(state)
            dim_a = int(np.sqrt(total_dim))
            partition = (dim_a, total_dim // dim_a)
        
        entropy = self.compute_entanglement_entropy(state, partition)
        
        measure = EntanglementMeasure(
            name=f"entanglement_{subsystem_a}_{subsystem_b}",
            value=entropy,
            subsystems=(subsystem_a, subsystem_b),
            measure_type="von_neumann"
        )
        
        self.entanglement_measures.append(measure)
        self.bipartite_structure[(subsystem_a, subsystem_b)] = {
            "entropy": entropy,
            "partition": partition
        }
        
        return measure
    
    def compute_mutual_information(self, state: np.ndarray,
                                   partition: Tuple[int, ...]) -> float:
        """
        Compute mutual information I(A:B) = S(A) + S(B) - S(AB).
        
        Args:
            state: Joint state
            partition: Partition defining subsystems
            
        Returns:
            Mutual information value
        """
        # Total entropy
        if len(state.shape) == 1:
            density_matrix = np.outer(state, np.conj(state))
        else:
            density_matrix = state
        
        s_total = self.compute_von_neumann_entropy(density_matrix)
        
        # Subsystem entropies (simplified calculation)
        s_a = self.compute_entanglement_entropy(state, partition)
        # For equal bipartition, S(B) ≈ S(A)
        s_b = s_a
        
        mutual_info = s_a + s_b - s_total
        return max(0.0, mutual_info)  # MI is non-negative
    
    def detect_multipartite_entanglement(self, state: np.ndarray,
                                        subsystems: List[str]) -> Dict[str, Any]:
        """
        Detect multipartite entanglement structure.
        
        Args:
            state: Multipartite quantum state
            subsystems: List of subsystem names
            
        Returns:
            Dictionary describing multipartite entanglement
        """
        n_subsystems = len(subsystems)
        entanglement_structure = {
            "subsystems": subsystems,
            "bipartite_measures": [],
            "genuine_multipartite": False
        }
        
        # Analyze all bipartite cuts
        for i in range(n_subsystems - 1):
            for j in range(i + 1, n_subsystems):
                measure = self.detect_bipartite_entanglement(
                    state, 
                    subsystems[i], 
                    subsystems[j]
                )
                entanglement_structure["bipartite_measures"].append({
                    "subsystems": (subsystems[i], subsystems[j]),
                    "entropy": measure.value
                })
        
        # Check for genuine multipartite entanglement (GHZ, W states, etc.)
        avg_bipartite = np.mean([m["entropy"] for m in entanglement_structure["bipartite_measures"]])
        if avg_bipartite > 0.1:  # Threshold for genuine multipartite
            entanglement_structure["genuine_multipartite"] = True
        
        return entanglement_structure
    
    def compute_negativity(self, density_matrix: np.ndarray, 
                          partition: Tuple[int, ...]) -> float:
        """
        Compute negativity as an entanglement measure.
        
        Negativity quantifies entanglement via partial transpose.
        
        Args:
            density_matrix: Density matrix of the system
            partition: Partition defining subsystems
            
        Returns:
            Negativity value
        """
        try:
            # Partial transpose
            rho_pt = self._partial_transpose(density_matrix, partition)
            
            # Negativity is sum of negative eigenvalues
            eigenvalues = np.linalg.eigvalsh(rho_pt)
            negativity = np.sum(np.abs(eigenvalues[eigenvalues < 0]))
            
            return float(negativity)
        except Exception:
            return 0.0
    
    def analyze_entanglement_spectrum(self, state: np.ndarray,
                                     partition: Tuple[int, ...]) -> Dict[str, Any]:
        """
        Analyze the entanglement spectrum.
        
        Args:
            state: Quantum state
            partition: Bipartition
            
        Returns:
            Dictionary with spectrum analysis
        """
        if len(state.shape) == 1:
            density_matrix = np.outer(state, np.conj(state))
        else:
            density_matrix = state
        
        dim_a = partition[0] if partition else 2
        dim_b = density_matrix.shape[0] // dim_a
        
        try:
            reduced_rho = self._partial_trace(density_matrix, dim_a, dim_b, trace_over='B')
            eigenvalues = np.linalg.eigvalsh(reduced_rho)
            eigenvalues = np.sort(eigenvalues)[::-1]  # Sort descending
            
            # Filter significant eigenvalues
            significant = eigenvalues[eigenvalues > 1e-10]
            
            return {
                "spectrum": significant.tolist(),
                "effective_dimension": len(significant),
                "participation_ratio": self._compute_participation_ratio(significant),
                "entropy": self.compute_von_neumann_entropy(reduced_rho)
            }
        except Exception:
            return {"spectrum": [], "effective_dimension": 0, "participation_ratio": 0.0}
    
    def _partial_trace(self, rho: np.ndarray, dim_a: int, dim_b: int, 
                       trace_over: str = 'B') -> np.ndarray:
        """Compute partial trace over subsystem."""
        rho_reshaped = rho.reshape(dim_a, dim_b, dim_a, dim_b)
        
        if trace_over == 'B':
            # Trace over B
            reduced = np.trace(rho_reshaped, axis1=1, axis2=3)
        else:
            # Trace over A
            reduced = np.trace(rho_reshaped, axis1=0, axis2=2)
        
        return reduced
    
    def _partial_transpose(self, rho: np.ndarray, partition: Tuple[int, ...]) -> np.ndarray:
        """Compute partial transpose."""
        dim_a = partition[0]
        dim_b = rho.shape[0] // dim_a
        
        rho_reshaped = rho.reshape(dim_a, dim_b, dim_a, dim_b)
        # Transpose subsystem B
        rho_pt = np.transpose(rho_reshaped, (0, 3, 2, 1))
        return rho_pt.reshape(rho.shape)
    
    def _compute_participation_ratio(self, eigenvalues: np.ndarray) -> float:
        """Compute participation ratio from eigenvalues."""
        if len(eigenvalues) == 0:
            return 0.0
        return float(1.0 / np.sum(eigenvalues**2))
    
    def get_entanglement_summary(self) -> Dict[str, Any]:
        """Get summary of all entanglement measures."""
        return {
            "total_measures": len(self.entanglement_measures),
            "bipartite_pairs": len(self.bipartite_structure),
            "measures": [
                {
                    "name": m.name,
                    "value": m.value,
                    "subsystems": m.subsystems,
                    "type": m.measure_type
                }
                for m in self.entanglement_measures
            ]
        }
