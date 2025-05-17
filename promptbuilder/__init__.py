"""
PromptBuilder: Query Handling Training Data Generator for Code Assisting LLMs

This package provides a comprehensive suite of tools for generating
training examples for fine-tuning LLMs as coding assistants. The system
supports multiple reasoning frameworks, query types, and difficulty levels.

Usage:
    import promptbuilder
    promptbuilder.version()  # Check version
    
    # Generate examples
    from promptbuilder.core.example_generator import ExampleGenerator
    generator = ExampleGenerator()
    examples = generator.generate_examples()

The package is structured as follows:
    - core: Core components for example generation and validation
    - domains: Domain-specific knowledge and templates
    - utils: Utility functions for logging, parsing, etc.
    - config: Configuration management

For CLI usage, see promptbuilder.main module.

Author: Made in Jurgistan
Version: 2.0.0
License: MIT
"""

from pathlib import Path
import sys
from importlib.metadata import version as get_pkg_version, PackageNotFoundError

# Add parent directory to path for local development
sys.path.insert(0, str(Path(__file__).parent.parent))

__version__ = "2.0.0"
__author__ = "Made in Jurgistan"
__license__ = "MIT"

def version():
    """Return the package version."""
    return __version__

def license():
    """Return the package license."""
    return __license__

# Core components
from .config import Config, validate_config, load_config
from .core.query_components import (
    QueryType,
    DifficultyLevel,
    FrameworkType,
    Section,
    QueryComponents,
    QueryInfo,
    Example
)
from .core.framework_registry import FrameworkRegistry
from .core.example_generator import ExampleGenerator
from .core.example_validator import ExampleValidator

# Domain data and utilities
from .domains import get_domain_mapping, get_available_domains
from .utils import setup_logging, get_logger, log_dict

# Package metadata with fallback for development environments
try:
    __version__ = get_pkg_version("promptbuilder")
except PackageNotFoundError:
    __version__ = "2.0.0"  # Default fallback version

__author__ = "Made in Jurgistan"
__license__ = "MIT"

__all__ = [
    # Configuration
    'Config', 'validate_config', 'load_config',
    
    # Core enumerations
    'QueryType', 'DifficultyLevel', 'FrameworkType',
    
    # Core data structures
    'Section', 'QueryComponents', 'QueryInfo', 'Example',
    
    # Framework components
    'FrameworkRegistry',
    
    # Generation engine
    'ExampleGenerator', 'ExampleValidator',
    
    # Domain utilities
    'get_domain_mapping', 'get_available_domains',
    
    # Logging utilities
    'setup_logging', 'get_logger', 'log_dict',
]

def get_version():
    """Return the package version as a string."""
    return __version__

def get_copyright():
    """Return copyright information."""
    return f"Copyright © 2025 {__author__}. MIT License."

def get_framework_types():
    """Return available framework types.
    
    Returns:
        list: List of available reasoning framework types
    """
    return [framework.value for framework in FrameworkType]

def get_query_types():
    """Return available query types.
    
    Returns:
        list: List of available query types
    """
    return [query_type.value for query_type in QueryType]

def get_difficulty_levels():
    """Return available difficulty levels.
    
    Returns:
        list: List of available difficulty levels
    """
    return [level.value for level in DifficultyLevel]