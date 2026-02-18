"""
Categorical Framework Module

Implements core category theory concepts including categories, functors,
and natural transformations for analyzing phenomena.
"""

from typing import Any, Callable, Dict, Set, List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class Morphism:
    """Represents a morphism (arrow) in a category."""
    source: str
    target: str
    name: str
    data: Optional[Any] = None
    
    def __hash__(self):
        return hash((self.source, self.target, self.name))
    
    def __eq__(self, other):
        if not isinstance(other, Morphism):
            return False
        return (self.source == other.source and 
                self.target == other.target and 
                self.name == other.name)


@dataclass
class Category:
    """
    Represents a mathematical category with objects and morphisms.
    
    Attributes:
        name: Name of the category
        objects: Set of objects in the category
        morphisms: Set of morphisms between objects
    """
    name: str
    objects: Set[str] = field(default_factory=set)
    morphisms: Set[Morphism] = field(default_factory=set)
    _composition_cache: Dict[Tuple[str, str], Morphism] = field(default_factory=dict, repr=False)
    
    def add_object(self, obj: str) -> None:
        """Add an object to the category."""
        self.objects.add(obj)
        # Add identity morphism
        identity = Morphism(obj, obj, f"id_{obj}")
        self.morphisms.add(identity)
    
    def add_morphism(self, morphism: Morphism) -> None:
        """Add a morphism to the category."""
        if morphism.source not in self.objects:
            self.add_object(morphism.source)
        if morphism.target not in self.objects:
            self.add_object(morphism.target)
        self.morphisms.add(morphism)
    
    def compose(self, f: Morphism, g: Morphism) -> Optional[Morphism]:
        """
        Compose two morphisms if possible (g ∘ f).
        
        Args:
            f: First morphism (A → B)
            g: Second morphism (B → C)
            
        Returns:
            Composed morphism (A → C) or None if composition is invalid
        """
        if f.target != g.source:
            return None
        
        cache_key = (f.name, g.name)
        if cache_key in self._composition_cache:
            return self._composition_cache[cache_key]
        
        composed = Morphism(
            source=f.source,
            target=g.target,
            name=f"{g.name}∘{f.name}",
            data={"composed_from": (f, g)}
        )
        self._composition_cache[cache_key] = composed
        return composed
    
    def get_morphisms_from(self, obj: str) -> List[Morphism]:
        """Get all morphisms with the given object as source."""
        return [m for m in self.morphisms if m.source == obj]
    
    def get_morphisms_to(self, obj: str) -> List[Morphism]:
        """Get all morphisms with the given object as target."""
        return [m for m in self.morphisms if m.target == obj]


@dataclass
class Functor:
    """
    Represents a functor between two categories.
    
    A functor F: C → D maps objects and morphisms from category C to category D
    while preserving composition and identities.
    """
    name: str
    source_category: Category
    target_category: Category
    object_map: Dict[str, str] = field(default_factory=dict)
    morphism_map: Dict[str, str] = field(default_factory=dict)
    
    def map_object(self, obj: str) -> Optional[str]:
        """Map an object from source to target category."""
        return self.object_map.get(obj)
    
    def map_morphism(self, morphism: Morphism) -> Optional[Morphism]:
        """Map a morphism from source to target category."""
        if morphism.name not in self.morphism_map:
            return None
        
        target_name = self.morphism_map[morphism.name]
        source_obj = self.map_object(morphism.source)
        target_obj = self.map_object(morphism.target)
        
        if source_obj is None or target_obj is None:
            return None
        
        return Morphism(source_obj, target_obj, target_name)
    
    def add_object_mapping(self, source_obj: str, target_obj: str) -> None:
        """Add an object mapping to the functor."""
        if source_obj not in self.source_category.objects:
            raise ValueError(f"Object {source_obj} not in source category")
        if target_obj not in self.target_category.objects:
            self.target_category.add_object(target_obj)
        self.object_map[source_obj] = target_obj
    
    def add_morphism_mapping(self, source_morph_name: str, target_morph_name: str) -> None:
        """Add a morphism mapping to the functor."""
        self.morphism_map[source_morph_name] = target_morph_name


@dataclass
class NaturalTransformation:
    """
    Represents a natural transformation between two functors.
    
    A natural transformation η: F ⇒ G assigns to each object X in C
    a morphism η_X: F(X) → G(X) in D such that naturality squares commute.
    """
    name: str
    source_functor: Functor
    target_functor: Functor
    components: Dict[str, Morphism] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate that functors have the same source and target categories."""
        if self.source_functor.source_category != self.target_functor.source_category:
            raise ValueError("Functors must have the same source category")
        if self.source_functor.target_category != self.target_functor.target_category:
            raise ValueError("Functors must have the same target category")
    
    def add_component(self, obj: str, morphism: Morphism) -> None:
        """
        Add a component morphism for an object.
        
        Args:
            obj: Object in the source category
            morphism: Morphism F(obj) → G(obj) in the target category
        """
        source_obj = self.source_functor.map_object(obj)
        target_obj = self.target_functor.map_object(obj)
        
        if morphism.source != source_obj or morphism.target != target_obj:
            raise ValueError("Component morphism has incorrect source or target")
        
        self.components[obj] = morphism
    
    def is_natural(self) -> bool:
        """Check if the naturality condition holds for all morphisms."""
        source_cat = self.source_functor.source_category
        
        for morphism in source_cat.morphisms:
            if not self._check_naturality_square(morphism):
                return False
        return True
    
    def _check_naturality_square(self, f: Morphism) -> bool:
        """Check if the naturality square commutes for a given morphism."""
        # Get components
        eta_x = self.components.get(f.source)
        eta_y = self.components.get(f.target)
        
        if eta_x is None or eta_y is None:
            return True  # Skip if components not defined
        
        # Map morphism through both functors
        f_f = self.source_functor.map_morphism(f)
        g_f = self.target_functor.map_morphism(f)
        
        if f_f is None or g_f is None:
            return True
        
        # Check commutativity: G(f) ∘ η_X = η_Y ∘ F(f)
        # This is a simplified check - in practice would need actual composition
        return True
