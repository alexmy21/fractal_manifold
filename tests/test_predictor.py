"""
Tests for the Phenomenon Predictor
"""

import pytest
import numpy as np

from fractal_manifold import PhenomenonPredictor


class TestPhenomenonPredictor:
    def test_create_predictor(self):
        predictor = PhenomenonPredictor()
        assert predictor.category.name == "phenomena_category"
        assert len(predictor.phenomena_analyzed) == 0
    
    def test_analyze_simple_phenomenon(self):
        predictor = PhenomenonPredictor()
        data = np.array([1, 0, 0, 1]) / np.sqrt(2)
        
        analysis = predictor.analyze_phenomenon(
            phenomenon_name="test_phenomenon",
            data=data
        )
        
        assert analysis.phenomenon_name == "test_phenomenon"
        assert analysis.categorical_correspondence is not None
        assert "test_phenomenon" in predictor.phenomena_analyzed
    
    def test_analyze_with_entanglement(self):
        predictor = PhenomenonPredictor()
        bell_state = np.array([1, 0, 0, 1]) / np.sqrt(2)
        
        analysis = predictor.analyze_phenomenon(
            phenomenon_name="bell_state",
            data=bell_state,
            subsystems=["A", "B"],
            analyze_entanglement=True
        )
        
        assert analysis.entanglement_structure is not None
        assert len(analysis.entanglement_structure.get("measures", [])) > 0
    
    def test_analyze_with_symmetry(self):
        predictor = PhenomenonPredictor()
        
        def identity(x):
            return x
        
        analysis = predictor.analyze_phenomenon(
            phenomenon_name="symmetric",
            data=np.array([1, 2, 3]),
            check_discrete=True,
            discrete_transformations=[identity]
        )
        
        assert len(analysis.symmetries) > 0
    
    def test_categorical_question(self):
        predictor = PhenomenonPredictor()
        data = np.array([1, 2, 3])
        
        predictor.analyze_phenomenon("test", data)
        
        answer = predictor.answer_categorical_question("test")
        assert "test" in answer
        assert "category" in answer.lower()
    
    def test_symmetry_question(self):
        predictor = PhenomenonPredictor()
        
        def identity(x):
            return x
        
        predictor.analyze_phenomenon(
            "test",
            np.array([1, 2, 3]),
            check_discrete=True,
            discrete_transformations=[identity]
        )
        
        answer = predictor.answer_symmetry_question("test")
        assert "test" in answer
    
    def test_conservation_question(self):
        predictor = PhenomenonPredictor()
        
        def identity(x):
            return x
        
        predictor.analyze_phenomenon(
            "test",
            np.array([1, 2, 3]),
            check_discrete=True,
            discrete_transformations=[identity],
            check_continuous=True,
            lagrangian=lambda x: np.sum(x**2),
            parameter_name="time"
        )
        
        answer = predictor.answer_conservation_question("test")
        assert "test" in answer
    
    def test_get_complete_summary(self):
        predictor = PhenomenonPredictor()
        data = np.array([1, 0, 0, 1]) / np.sqrt(2)
        
        predictor.analyze_phenomenon(
            "test",
            data,
            subsystems=["A", "B"]
        )
        
        summary = predictor.get_complete_summary("test")
        assert summary["phenomenon"] == "test"
        assert "categorical" in summary
        assert "predictions" in summary
    
    def test_analyze_rg_flow(self):
        predictor = PhenomenonPredictor()
        
        def beta_g(couplings):
            return -couplings.get("g", 0)**2
        
        analysis = predictor.analyze_phenomenon(
            "rg_test",
            None,
            analyze_rg_flow=True,
            beta_functions={"g": beta_g},
            initial_couplings={"g": 1.0}
        )
        
        assert analysis.rg_flow_data is not None
        assert analysis.rg_flow_data.get("flow_computed") == True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
