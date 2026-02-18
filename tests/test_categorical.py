"""
Tests for the Categorical Framework
"""

import pytest
import numpy as np

from fractal_manifold.categorical import Category, Morphism, Functor, NaturalTransformation


class TestCategory:
    def test_create_category(self):
        cat = Category(name="test_category")
        assert cat.name == "test_category"
        assert len(cat.objects) == 0
        assert len(cat.morphisms) == 0
    
    def test_add_object(self):
        cat = Category(name="test")
        cat.add_object("A")
        assert "A" in cat.objects
        # Should have identity morphism
        id_morphisms = [m for m in cat.morphisms if m.source == "A" and m.target == "A"]
        assert len(id_morphisms) == 1
    
    def test_add_morphism(self):
        cat = Category(name="test")
        morph = Morphism(source="A", target="B", name="f")
        cat.add_morphism(morph)
        
        assert "A" in cat.objects
        assert "B" in cat.objects
        assert morph in cat.morphisms
    
    def test_compose_morphisms(self):
        cat = Category(name="test")
        f = Morphism(source="A", target="B", name="f")
        g = Morphism(source="B", target="C", name="g")
        
        cat.add_morphism(f)
        cat.add_morphism(g)
        
        composed = cat.compose(f, g)
        assert composed is not None
        assert composed.source == "A"
        assert composed.target == "C"
    
    def test_compose_invalid(self):
        cat = Category(name="test")
        f = Morphism(source="A", target="B", name="f")
        g = Morphism(source="C", target="D", name="g")
        
        composed = cat.compose(f, g)
        assert composed is None


class TestFunctor:
    def test_create_functor(self):
        source = Category(name="C")
        target = Category(name="D")
        functor = Functor(name="F", source_category=source, target_category=target)
        
        assert functor.name == "F"
        assert functor.source_category == source
        assert functor.target_category == target
    
    def test_object_mapping(self):
        source = Category(name="C")
        target = Category(name="D")
        source.add_object("A")
        target.add_object("A'")
        
        functor = Functor(name="F", source_category=source, target_category=target)
        functor.add_object_mapping("A", "A'")
        
        assert functor.map_object("A") == "A'"
    
    def test_morphism_mapping(self):
        source = Category(name="C")
        target = Category(name="D")
        
        f = Morphism(source="A", target="B", name="f")
        source.add_morphism(f)
        
        target.add_object("A'")
        target.add_object("B'")
        
        functor = Functor(name="F", source_category=source, target_category=target)
        functor.add_object_mapping("A", "A'")
        functor.add_object_mapping("B", "B'")
        functor.add_morphism_mapping("f", "f'")
        
        mapped = functor.map_morphism(f)
        assert mapped is not None
        assert mapped.source == "A'"
        assert mapped.target == "B'"


class TestNaturalTransformation:
    def test_create_natural_transformation(self):
        source_cat = Category(name="C")
        target_cat = Category(name="D")
        
        F = Functor(name="F", source_category=source_cat, target_category=target_cat)
        G = Functor(name="G", source_category=source_cat, target_category=target_cat)
        
        eta = NaturalTransformation(name="eta", source_functor=F, target_functor=G)
        assert eta.name == "eta"
    
    def test_invalid_natural_transformation(self):
        cat1 = Category(name="C1")
        cat2 = Category(name="C2")
        cat3 = Category(name="D")
        
        F = Functor(name="F", source_category=cat1, target_category=cat3)
        G = Functor(name="G", source_category=cat2, target_category=cat3)
        
        with pytest.raises(ValueError):
            NaturalTransformation(name="eta", source_functor=F, target_functor=G)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
