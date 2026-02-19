"""
ManifoldOS Extension System

Provides pluggable architecture for external resources and integrations.
Extensions are optional and ManifoldOS degrades gracefully when unavailable.

Architecture:
  - ManifoldExtension: Base class for all extensions
  - ExtensionRegistry: Manages extension lifecycle
  - Capability-based: Extensions declare what they provide
  - Progressive enhancement: Core works without extensions

Example extensions:
  - Storage (DuckDB, PostgreSQL, Redis)
  - Caching (Redis, Memcached)
  - Monitoring (Prometheus, StatsD)
  - Vector stores (Pinecone, Weaviate)
"""

from core.extensions.base import (
    ManifoldExtension,
    ExtensionRegistry,
    ExtensionError
)

from core.extensions.storage import (
    StorageExtension,
    DuckDBStorageExtension
)

from core.extensions.stateless_validator import (
    StatelessnessValidator,
    StateViolation,
    validate_extension_statelessness
)

__all__ = [
    'ManifoldExtension',
    'ExtensionRegistry',
    'ExtensionError',
    'StorageExtension',
    'DuckDBStorageExtension',
    'StatelessnessValidator',
    'StateViolation',
    'validate_extension_statelessness',
]
