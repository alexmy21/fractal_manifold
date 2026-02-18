"""
Spectral Sequence Module

Implements spectral sequence computations for algebraic topology
and homological algebra applications.
"""

from typing import Any, Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
import numpy as np


@dataclass
class SpectralSequencePage:
    """Represents a page in a spectral sequence."""
    page_number: int
    bidegree_terms: Dict[Tuple[int, int], Any] = field(default_factory=dict)
    differentials: Dict[Tuple[int, int], Any] = field(default_factory=dict)


class SpectralSequence:
    """
    Implements spectral sequence computations.
    
    Spectral sequences are tools for computing homology/cohomology groups
    through successive approximations.
    """
    
    def __init__(self, name: str = "spectral_sequence"):
        self.name = name
        self.pages: Dict[int, SpectralSequencePage] = {}
        self.current_page = 0
        self.converged = False
        self.limit_page: Optional[SpectralSequencePage] = None
    
    def initialize_e0_page(self, terms: Dict[Tuple[int, int], Any]) -> None:
        """
        Initialize the E^0 page of the spectral sequence.
        
        Args:
            terms: Dictionary mapping (p, q) bidegrees to terms
        """
        page = SpectralSequencePage(page_number=0, bidegree_terms=terms)
        self.pages[0] = page
        self.current_page = 0
    
    def initialize_from_filtration(self, filtered_complex: List[Any]) -> None:
        """
        Initialize spectral sequence from a filtered complex.
        
        Args:
            filtered_complex: Filtered chain/cochain complex
        """
        # E^0 page from associated graded complex
        e0_terms = {}
        for p, filtration_level in enumerate(filtered_complex):
            # Simplified: each filtration level contributes to E^0
            for q in range(len(filtration_level) if hasattr(filtration_level, '__len__') else 1):
                e0_terms[(p, q)] = filtration_level
        
        self.initialize_e0_page(e0_terms)
    
    def compute_differential(self, page_num: int, 
                            bidegree: Tuple[int, int]) -> Optional[Any]:
        """
        Compute the differential d_r at a given bidegree.
        
        For page E^r, differential d_r: E^r_{p,q} → E^r_{p+r, q-r+1}
        
        Args:
            page_num: Page number r
            bidegree: (p, q) bidegree
            
        Returns:
            Differential map
        """
        if page_num not in self.pages:
            return None
        
        p, q = bidegree
        # Differential maps (p,q) to (p+r, q-r+1) on page r
        target_bidegree = (p + page_num, q - page_num + 1)
        
        page = self.pages[page_num]
        if bidegree in page.bidegree_terms:
            # Store differential
            page.differentials[bidegree] = target_bidegree
            return target_bidegree
        
        return None
    
    def compute_next_page(self) -> SpectralSequencePage:
        """
        Compute the next page E^{r+1} from E^r.
        
        E^{r+1}_{p,q} = ker(d_r) / im(d_r)
        
        Returns:
            Next page of the spectral sequence
        """
        if self.current_page not in self.pages:
            raise ValueError(f"Current page {self.current_page} not initialized")
        
        current = self.pages[self.current_page]
        next_page_num = self.current_page + 1
        next_page = SpectralSequencePage(page_number=next_page_num)
        
        # Compute homology at each bidegree
        for bidegree, term in current.bidegree_terms.items():
            # Compute ker(d_r) / im(d_r)
            # Simplified: just propagate non-zero terms
            if term is not None:
                next_page.bidegree_terms[bidegree] = term
        
        self.pages[next_page_num] = next_page
        self.current_page = next_page_num
        
        return next_page
    
    def compute_to_convergence(self, max_pages: int = 10) -> None:
        """
        Compute spectral sequence until convergence.
        
        Args:
            max_pages: Maximum number of pages to compute
        """
        for _ in range(max_pages):
            if self._check_convergence():
                self.converged = True
                self.limit_page = self.pages[self.current_page]
                break
            
            self.compute_next_page()
    
    def _check_convergence(self) -> bool:
        """Check if spectral sequence has converged."""
        if self.current_page < 2:
            return False
        
        current = self.pages.get(self.current_page)
        previous = self.pages.get(self.current_page - 1)
        
        if not current or not previous:
            return False
        
        # Converged if terms stabilize
        current_bidegrees = set(current.bidegree_terms.keys())
        previous_bidegrees = set(previous.bidegree_terms.keys())
        
        return current_bidegrees == previous_bidegrees
    
    def get_limit_term(self, bidegree: Tuple[int, int]) -> Optional[Any]:
        """
        Get the limit term E^∞_{p,q}.
        
        Args:
            bidegree: (p, q) bidegree
            
        Returns:
            Limit term if converged
        """
        if not self.converged or not self.limit_page:
            return None
        
        return self.limit_page.bidegree_terms.get(bidegree)
    
    def compute_edge_homomorphism(self, degree: int) -> List[Any]:
        """
        Compute edge homomorphism from spectral sequence.
        
        Args:
            degree: Total degree n = p + q
            
        Returns:
            Edge homomorphism data
        """
        edge_terms = []
        
        if self.limit_page:
            for (p, q), term in self.limit_page.bidegree_terms.items():
                if p + q == degree:
                    edge_terms.append(term)
        
        return edge_terms
    
    def get_total_homology(self, degree: int) -> Dict[str, Any]:
        """
        Compute total homology at a given degree from limit page.
        
        Args:
            degree: Total degree n
            
        Returns:
            Dictionary with total homology information
        """
        if not self.converged:
            self.compute_to_convergence()
        
        terms_at_degree = []
        
        if self.limit_page:
            for (p, q), term in self.limit_page.bidegree_terms.items():
                if p + q == degree:
                    terms_at_degree.append({
                        "bidegree": (p, q),
                        "term": term
                    })
        
        return {
            "degree": degree,
            "terms": terms_at_degree,
            "rank": len(terms_at_degree)
        }
    
    def get_spectral_sequence_summary(self) -> Dict[str, Any]:
        """Get summary of spectral sequence computation."""
        return {
            "name": self.name,
            "current_page": self.current_page,
            "total_pages": len(self.pages),
            "converged": self.converged,
            "pages_computed": list(self.pages.keys()),
            "limit_bidegrees": list(self.limit_page.bidegree_terms.keys()) if self.limit_page else []
        }
    
    def visualize_page(self, page_num: int) -> str:
        """
        Create a text visualization of a spectral sequence page.
        
        Args:
            page_num: Page number to visualize
            
        Returns:
            String representation of the page
        """
        if page_num not in self.pages:
            return f"Page E^{page_num} not computed"
        
        page = self.pages[page_num]
        lines = [f"Spectral Sequence Page E^{page_num}:"]
        lines.append("-" * 40)
        
        if not page.bidegree_terms:
            lines.append("(empty)")
        else:
            # Get range of bidegrees
            p_values = [p for p, q in page.bidegree_terms.keys()]
            q_values = [q for p, q in page.bidegree_terms.keys()]
            
            if p_values and q_values:
                max_p = max(p_values)
                max_q = max(q_values)
                
                for (p, q), term in sorted(page.bidegree_terms.items()):
                    lines.append(f"  E^{page_num}_{{{p},{q}}} = {term}")
        
        return "\n".join(lines)
