"""
Base classes for ManifoldOS extension system.

The extension architecture follows ManifoldOS core principles:
  1. Immutability: Extensions are immutable once initialized
  2. Idempotence: Operations can be repeated safely
  3. Content-addressability: Extensions identified by hash
  4. Loose coupling: Core doesn't depend on extensions
  5. Graceful degradation: Missing extensions don't break core
  
Knowledge base integration:
  - All extensions eventually end up in the system knowledge base
  - Extension state is content-addressed and versioned
  - Operations are append-only and immutable
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
import hashlib
import json


class ExtensionError(Exception):
    """Base exception for extension-related errors."""
    pass


@dataclass(frozen=True)
class ExtensionConfig:
    """
    Immutable extension configuration.
    
    Once created, configuration cannot be changed. This ensures:
      - Content-addressability: Hash is stable
      - Idempotence: Same config = same behavior
      - Knowledge base integration: Immutable configs can be stored
    """
    extension_type: str
    parameters: Tuple[Tuple[str, Any], ...] = field(default_factory=tuple)
    
    def __post_init__(self):
        # Ensure parameters is a tuple of tuples (immutable)
        if not isinstance(self.parameters, tuple):
            object.__setattr__(self, 'parameters', tuple(sorted(self.parameters.items())))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'extension_type': self.extension_type,
            'parameters': dict(self.parameters)
        }
    
    def get_hash(self) -> str:
        """
        Content-addressed hash of configuration.
        
        This hash uniquely identifies the extension configuration.
        Same config → same hash (idempotent).
        """
        config_str = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get parameter value."""
        params_dict = dict(self.parameters)
        return params_dict.get(key, default)


@dataclass(frozen=True)
class ExtensionInfo:
    """
    Immutable metadata about an extension.
    
    Frozen to ensure content-addressability.
    """
    name: str
    version: str
    description: str
    author: str = "unknown"
    config_hash: Optional[str] = None  # Content-addressed configuration (set after init)
    capabilities: Tuple[Tuple[str, bool], ...] = field(default_factory=tuple)
    
    def __post_init__(self):
        # Ensure capabilities is a tuple of tuples (immutable)
        if not isinstance(self.capabilities, tuple):
            if isinstance(self.capabilities, dict):
                object.__setattr__(self, 'capabilities', tuple(sorted(self.capabilities.items())))
            else:
                object.__setattr__(self, 'capabilities', tuple())
    
    def get_capabilities(self) -> Dict[str, bool]:
        """Get capabilities as dictionary."""
        return dict(self.capabilities)


class ManifoldExtension(ABC):
    """
    Base class for all ManifoldOS extensions.
    
    Core Principles (aligned with ManifoldOS):
      1. Immutability: Configuration frozen after initialization
      2. Idempotence: Same operation → same result
      3. Content-addressability: Hash identifies extension state
    
    Extensions are optional components that enhance ManifoldOS with
    additional capabilities like persistent storage, caching, monitoring, etc.
    
    Lifecycle:
      1. __init__(): Create extension instance
      2. initialize(config): Setup with IMMUTABLE configuration
      3. is_available(): Check operational status
      4. [use extension methods - all idempotent]
      5. cleanup(): Release resources
    
    Knowledge Base Integration:
      - Extension configurations stored in KB
      - Operations logged immutably
      - State transitions are append-only
    
    Design principles:
      - Fail gracefully: Never crash ManifoldOS
      - Declare capabilities: Clear about what you provide
      - Immutable config: Configuration frozen at init
      - Resource-aware: Proper cleanup
      - Idempotent operations: Safe to repeat
    """
    
    def __init__(self):
        """Initialize extension with no configuration."""
        self._config: Optional[ExtensionConfig] = None
        self._config_hash: Optional[str] = None
        self._initialized: bool = False
    
    @abstractmethod
    def initialize(self, config: Dict[str, Any]) -> bool:
        """
        Initialize the extension with IMMUTABLE configuration.
        
        Configuration is frozen after initialization to ensure:
          - Content-addressability (stable hash)
          - Idempotence (same config = same behavior)
          - Knowledge base integration (immutable configs)
        
        Args:
            config: Configuration dictionary (will be frozen)
        
        Returns:
            True if initialization succeeded, False otherwise
            
        Note:
            Should never raise exceptions - return False on failure.
            Configuration CANNOT be changed after successful initialization.
        """
        pass
    
    def _freeze_config(self, config: Dict[str, Any], extension_type: str):
        """
        Freeze configuration (call from initialize()).
        
        This makes the configuration immutable and content-addressed.
        Must be called exactly once during initialization.
        """
        if self._config is not None:
            raise ExtensionError("Extension already initialized - configuration is immutable")
        
        # Create immutable config
        params = tuple(sorted(config.items()))
        self._config = ExtensionConfig(extension_type=extension_type, parameters=params)
        self._config_hash = self._config.get_hash()
        self._initialized = True
    
    def get_config(self) -> Optional[ExtensionConfig]:
        """Get immutable configuration."""
        return self._config
    
    def get_config_hash(self) -> Optional[str]:
        """Get content-addressed configuration hash."""
        return self._config_hash
    
    @property
    def config(self) -> Dict[str, Any]:
        """
        Get configuration as dictionary (read-only).
        
        Returns the configuration as a regular dict for easy access.
        The underlying ExtensionConfig is immutable.
        """
        if self._config is None:
            return {}
        return dict(self._config.parameters)
    
    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if extension is available and operational.
        
        Returns:
            True if extension can be used, False otherwise
        """
        pass
    
    @abstractmethod
    def cleanup(self):
        """
        Cleanup resources and gracefully shutdown.
        
        Should never raise exceptions - handle errors internally.
        """
        pass
    
    @abstractmethod
    def get_capabilities(self) -> Dict[str, bool]:
        """
        Return capabilities this extension provides.
        
        Returns:
            Dictionary mapping capability names to availability.
            
        Example:
            {
                'persistent_storage': True,
                'query_optimization': False,
                'full_text_search': True
            }
        """
        pass
    
    @abstractmethod
    def get_info(self) -> ExtensionInfo:
        """
        Get metadata about this extension.
        
        Returns:
            ExtensionInfo with name, version, description, etc.
        """
        pass
    
    def validate_config(self, config: Dict[str, Any]) -> List[str]:
        """
        Validate configuration and return list of errors.
        
        Args:
            config: Configuration to validate
            
        Returns:
            List of error messages, empty if valid
        """
        return []  # Default: no validation errors


class ExtensionRegistry:
    """
    Manages lifecycle of all ManifoldOS extensions.
    
    Responsibilities:
      - Register/unregister extensions
      - Query available capabilities
      - Coordinate cleanup
      - Provide access to extensions
    
    Usage:
        registry = ExtensionRegistry()
        registry.register('storage', DuckDBStorageExtension())
        
        if registry.has_capability('persistent_storage'):
            storage = registry.get('storage')
            storage.store_data(...)
    """
    
    def __init__(self):
        self._extensions: Dict[str, ManifoldExtension] = {}
        self._initialized: Dict[str, bool] = {}
    
    def register(self, name: str, extension: ManifoldExtension, 
                 config: Optional[Dict[str, Any]] = None) -> bool:
        """
        Register and initialize an extension.
        
        Args:
            name: Unique name for this extension
            extension: Extension instance
            config: Configuration dictionary
            
        Returns:
            True if registration and initialization succeeded
        """
        try:
            # Validate config if provided
            if config:
                errors = extension.validate_config(config)
                if errors:
                    print(f"✗ Extension '{name}' config invalid: {errors}")
                    return False
            
            # Initialize extension
            success = extension.initialize(config or {})
            
            if success and extension.is_available():
                self._extensions[name] = extension
                self._initialized[name] = True
                
                info = extension.get_info()
                print(f"✓ Extension registered: {name} v{info.version}")
                return True
            else:
                print(f"✗ Extension '{name}' failed to initialize")
                return False
                
        except Exception as e:
            print(f"✗ Extension '{name}' error: {e}")
            return False
    
    def unregister(self, name: str):
        """Unregister and cleanup an extension."""
        if name in self._extensions:
            try:
                self._extensions[name].cleanup()
            except Exception as e:
                print(f"⚠ Extension '{name}' cleanup error: {e}")
            finally:
                del self._extensions[name]
                del self._initialized[name]
    
    def get(self, name: str) -> Optional[ManifoldExtension]:
        """
        Get extension by name.
        
        Args:
            name: Extension name
            
        Returns:
            Extension instance or None if not found
        """
        return self._extensions.get(name)
    
    def has(self, name: str) -> bool:
        """Check if extension is registered."""
        return name in self._extensions
    
    def has_capability(self, capability: str) -> bool:
        """
        Check if any extension provides a capability.
        
        Args:
            capability: Capability name to check
            
        Returns:
            True if at least one available extension provides it
        """
        return any(
            ext.is_available() and ext.get_capabilities().get(capability, False)
            for ext in self._extensions.values()
        )
    
    def get_by_capability(self, capability: str) -> List[ManifoldExtension]:
        """
        Get all extensions that provide a capability.
        
        Args:
            capability: Capability name
            
        Returns:
            List of extensions that provide the capability
        """
        return [
            ext for ext in self._extensions.values()
            if ext.is_available() and ext.get_capabilities().get(capability, False)
        ]
    
    def list_extensions(self) -> List[str]:
        """Get list of registered extension names."""
        return list(self._extensions.keys())
    
    def list_capabilities(self) -> Dict[str, List[str]]:
        """
        Get all available capabilities and which extensions provide them.
        
        Returns:
            Dictionary mapping capability -> list of extension names
        """
        capabilities = {}
        
        for name, ext in self._extensions.items():
            if ext.is_available():
                for cap, available in ext.get_capabilities().items():
                    if available:
                        if cap not in capabilities:
                            capabilities[cap] = []
                        capabilities[cap].append(name)
        
        return capabilities
    
    def cleanup_all(self):
        """Cleanup all registered extensions."""
        for name in list(self._extensions.keys()):
            self.unregister(name)
    
    def __enter__(self):
        """Context manager support."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Cleanup on context exit."""
        self.cleanup_all()
