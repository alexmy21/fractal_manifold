#!/usr/bin/env python3
"""
Extension Statelessness Validator

Ensures all ManifoldOS extensions are truly stateless:
  1. No mutable state accumulated across operations
  2. Same inputs → same outputs (deterministic)
  3. No hidden side effects
  4. Idempotent operations

Statelessness is critical for:
  - Knowledge base integration
  - Distributed systems
  - Reproducibility
  - Testing
"""

import inspect
from typing import Any, Dict, List, Set, Tuple, Optional, Type
from dataclasses import dataclass
import ast


@dataclass
class StateViolation:
    """A detected statelessness violation."""
    extension_name: str
    violation_type: str
    location: str
    description: str
    severity: str  # 'error', 'warning', 'info'
    
    def __str__(self):
        symbol = {'error': '✗', 'warning': '⚠', 'info': 'ℹ'}[self.severity]
        return f"{symbol} [{self.violation_type}] {self.extension_name}.{self.location}: {self.description}"


class StatelessnessValidator:
    """
    Validates that extensions are stateless.
    
    Checks for common stateful patterns:
      - Mutable class variables (lists, dicts, sets)
      - Instance variables modified after init
      - Global state mutations
      - Non-idempotent operations
      - Hidden side effects
    """
    
    # Mutable types that indicate potential state
    MUTABLE_TYPES = (list, dict, set, bytearray)
    
    # Allowed instance variables (core infrastructure)
    ALLOWED_INSTANCE_VARS = {
        '_config', '_config_hash', '_initialized', '_available',
        'lut_store', 'conn'  # External resources (stateless interfaces)
    }
    
    def __init__(self):
        self.violations: List[StateViolation] = []
    
    def validate_extension(self, extension_class: Type, 
                          extension_name: Optional[str] = None) -> List[StateViolation]:
        """
        Validate an extension class for statelessness.
        
        Args:
            extension_class: The extension class to validate
            extension_name: Optional name for reporting
            
        Returns:
            List of violations found
        """
        self.violations = []
        name = extension_name or extension_class.__name__
        
        # Check class-level state
        self._check_class_variables(extension_class, name)
        
        # Check instance variables
        self._check_instance_variables(extension_class, name)
        
        # Check method signatures for state accumulation
        self._check_methods(extension_class, name)
        
        # Check source code for stateful patterns
        self._check_source_code(extension_class, name)
        
        return self.violations
    
    def _check_class_variables(self, cls: Type, name: str):
        """Check for mutable class variables."""
        for attr_name, attr_value in cls.__dict__.items():
            # Skip magic attributes and methods
            if attr_name.startswith('_') and attr_name.endswith('_'):
                continue
            if callable(attr_value):
                continue
            
            # Check for mutable types
            if isinstance(attr_value, self.MUTABLE_TYPES):
                self.violations.append(StateViolation(
                    extension_name=name,
                    violation_type="MUTABLE_CLASS_VAR",
                    location=attr_name,
                    description=f"Mutable class variable of type {type(attr_value).__name__}. "
                               "Use immutable types or instance variables.",
                    severity='error'
                ))
    
    def _check_instance_variables(self, cls: Type, name: str):
        """Check instance variables for potential state accumulation."""
        # Get __init__ method
        if not hasattr(cls, '__init__'):
            return
        
        init_method = cls.__init__
        try:
            source = inspect.getsource(init_method)
            tree = ast.parse(source)
            
            # Find all self.* assignments
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Attribute):
                            if isinstance(target.value, ast.Name) and target.value.id == 'self':
                                var_name = target.attr
                                
                                # Skip allowed vars
                                if var_name in self.ALLOWED_INSTANCE_VARS:
                                    continue
                                
                                # Check if it's a mutable collection
                                if isinstance(node.value, (ast.List, ast.Dict, ast.Set)):
                                    self.violations.append(StateViolation(
                                        extension_name=name,
                                        violation_type="MUTABLE_INSTANCE_VAR",
                                        location=f"__init__.{var_name}",
                                        description=f"Mutable instance variable initialized. "
                                                   "May accumulate state across operations.",
                                        severity='warning'
                                    ))
        except Exception as e:
            # Can't parse - just warn
            self.violations.append(StateViolation(
                extension_name=name,
                violation_type="PARSE_ERROR",
                location="__init__",
                description=f"Could not parse __init__ method: {e}",
                severity='info'
            ))
    
    def _check_methods(self, cls: Type, name: str):
        """Check methods for state-accumulating patterns."""
        for method_name, method in inspect.getmembers(cls, predicate=inspect.isfunction):
            # Skip private methods and magic methods
            if method_name.startswith('_'):
                continue
            
            try:
                source = inspect.getsource(method)
                tree = ast.parse(source)
                
                # Look for self.var modifications (state accumulation)
                for node in ast.walk(tree):
                    # Check for self.var = ... or self.var.append(...), etc.
                    if isinstance(node, ast.Assign):
                        for target in node.targets:
                            if isinstance(target, ast.Attribute):
                                if isinstance(target.value, ast.Name) and target.value.id == 'self':
                                    var_name = target.attr
                                    
                                    # Skip allowed vars
                                    if var_name in self.ALLOWED_INSTANCE_VARS:
                                        continue
                                    
                                    self.violations.append(StateViolation(
                                        extension_name=name,
                                        violation_type="STATE_MUTATION",
                                        location=f"{method_name}.{var_name}",
                                        description=f"Method modifies instance variable. "
                                                   "Extensions should be stateless.",
                                        severity='error'
                                    ))
                    
                    # Check for in-place mutations (.append, .extend, etc.)
                    if isinstance(node, ast.Call):
                        if isinstance(node.func, ast.Attribute):
                            if isinstance(node.func.value, ast.Attribute):
                                if (isinstance(node.func.value.value, ast.Name) and 
                                    node.func.value.value.id == 'self'):
                                    method_called = node.func.attr
                                    if method_called in ['append', 'extend', 'update', 'add', 'pop', 'remove']:
                                        var_name = node.func.value.attr
                                        if var_name not in self.ALLOWED_INSTANCE_VARS:
                                            self.violations.append(StateViolation(
                                                extension_name=name,
                                                violation_type="IN_PLACE_MUTATION",
                                                location=f"{method_name}.{var_name}.{method_called}",
                                                description=f"In-place mutation detected. "
                                                           "Extensions should not accumulate state.",
                                                severity='error'
                                            ))
                                        
            except Exception:
                # Can't parse this method - skip
                pass
    
    def _check_source_code(self, cls: Type, name: str):
        """Check source code for global state and other patterns."""
        try:
            source = inspect.getsource(cls)
            
            # Check for global keyword (modifying global state)
            if 'global ' in source:
                self.violations.append(StateViolation(
                    extension_name=name,
                    violation_type="GLOBAL_STATE",
                    location="class",
                    description="Uses 'global' keyword - modifying global state. "
                               "Extensions should be self-contained.",
                    severity='error'
                ))
            
            # Check for common stateful patterns
            stateful_patterns = [
                ('cache', 'CACHE_USAGE', 'warning', 'Uses caching - ensure cache is external/stateless'),
                ('singleton', 'SINGLETON_PATTERN', 'error', 'Singleton pattern detected - should be stateless'),
            ]
            
            for pattern, vtype, severity, desc in stateful_patterns:
                if pattern.lower() in source.lower():
                    self.violations.append(StateViolation(
                        extension_name=name,
                        violation_type=vtype,
                        location="class",
                        description=desc,
                        severity=severity
                    ))
                    
        except Exception:
            # Can't get source - skip
            pass
    
    def validate_idempotence(self, extension_instance: Any, 
                            operation_name: str,
                            test_input: Any,
                            num_runs: int = 3) -> Tuple[bool, Optional[str]]:
        """
        Test if an operation is idempotent.
        
        Runs the operation multiple times with same input and checks
        if results are identical.
        
        Args:
            extension_instance: Extension instance
            operation_name: Name of method to test
            test_input: Input to pass to operation
            num_runs: Number of times to run (default 3)
            
        Returns:
            (is_idempotent, error_message)
        """
        if not hasattr(extension_instance, operation_name):
            return False, f"Method '{operation_name}' not found"
        
        method = getattr(extension_instance, operation_name)
        
        try:
            results = []
            for i in range(num_runs):
                result = method(test_input)
                results.append(result)
            
            # Check if all results are identical
            first_result = results[0]
            for i, result in enumerate(results[1:], 1):
                if result != first_result:
                    return False, f"Run {i+1} produced different result: {result} != {first_result}"
            
            return True, None
            
        except Exception as e:
            return False, f"Operation failed: {e}"
    
    def generate_report(self, violations: List[StateViolation]) -> str:
        """Generate a human-readable report."""
        if not violations:
            return "✓ No statelessness violations detected"
        
        # Group by severity
        errors = [v for v in violations if v.severity == 'error']
        warnings = [v for v in violations if v.severity == 'warning']
        infos = [v for v in violations if v.severity == 'info']
        
        lines = []
        lines.append("=" * 70)
        lines.append("Statelessness Validation Report")
        lines.append("=" * 70)
        lines.append(f"Total violations: {len(violations)}")
        lines.append(f"  Errors:   {len(errors)}")
        lines.append(f"  Warnings: {len(warnings)}")
        lines.append(f"  Info:     {len(infos)}")
        lines.append("")
        
        if errors:
            lines.append("ERRORS (must fix):")
            lines.append("-" * 70)
            for v in errors:
                lines.append(str(v))
            lines.append("")
        
        if warnings:
            lines.append("WARNINGS (should review):")
            lines.append("-" * 70)
            for v in warnings:
                lines.append(str(v))
            lines.append("")
        
        if infos:
            lines.append("INFO (for awareness):")
            lines.append("-" * 70)
            for v in infos:
                lines.append(str(v))
            lines.append("")
        
        lines.append("=" * 70)
        
        return "\n".join(lines)


def validate_extension_statelessness(extension_class: Type,
                                     extension_name: Optional[str] = None) -> Tuple[bool, List[StateViolation]]:
    """
    Convenience function to validate an extension.
    
    Returns:
        (is_stateless, violations)
    """
    validator = StatelessnessValidator()
    violations = validator.validate_extension(extension_class, extension_name)
    
    # Only errors prevent certification
    errors = [v for v in violations if v.severity == 'error']
    is_stateless = len(errors) == 0
    
    return is_stateless, violations
