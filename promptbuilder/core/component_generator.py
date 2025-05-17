"""
Component Generator Module for PromptBuilder.

This module provides specialized generators for creating query components
based on query type, difficulty level, and domain. It abstracts the component
generation logic to reduce complexity in the main generator.

Author: Made in Jurgistan
Version: 2.0.0
License: MIT
"""

import logging
import random
from typing import Dict, Any, Optional, List, Union

from promptbuilder.core.query_components import QueryComponents, QueryType, DifficultyLevel


class ComponentGenerator:
    """Generates query components for different query types and difficulty levels."""

    def __init__(self, technology_mapping: Dict[str, Any] = None):
        """Initialize the component generator.

        Args:
            technology_mapping: Mapping of programming domains to technologies (optional)
        """
        self.technology_mapping = technology_mapping or {
            "web_development": {
                "languages": ["JavaScript", "TypeScript", "CSS", "HTML"],
                "frameworks": ["React", "Vue", "Angular", "Express", "Next.js"],
                "technologies": []
            },
            "data_science": {
                "languages": ["Python", "R"],
                "frameworks": [],
                "technologies": ["Pandas", "NumPy", "Matplotlib", "TensorFlow", "PyTorch", "Scikit-learn"]
            }
        }
        self.logger = logging.getLogger(__name__)
        self.logger.info("ComponentGenerator initialized")

    def generate_components(
        self,
        query_type: QueryType,
        difficulty: DifficultyLevel,
        domain: str
    ) -> QueryComponents:
        """Generate query components for the given parameters."""
        self.logger.debug("Generating components for %s, difficulty %s, domain %s",
                         query_type.value, difficulty.value, domain)
        components = QueryComponents()

        # Set domain-specific components
        components.domain = domain

        # Get technology data for the domain
        domain_data = self.technology_mapping.get(domain, {})
        if not domain_data:
            self.logger.warning("No domain data found for %s, using defaults", domain)

        # Set basic components
        components.language = "Python"
        components.technology = "Example Technology"
        components.concept = "sample concept"
        
        # Add specific components based on query type
        if query_type == QueryType.DEBUGGING:
            self._add_debugging_components(components, difficulty, domain_data)
        elif query_type == QueryType.OPTIMIZATION:
            self._add_optimization_components(components, difficulty, domain_data)
        else:
            # Default components for other query types
            components.error_message = "sample error"
            components.technology_context = "example context"
            
        return components
    
    def _add_debugging_components(
        self,
        components: QueryComponents,
        difficulty: DifficultyLevel,
        domain_data: Dict[str, Any]
    ) -> None:
        """Add debugging-specific components."""
        components.error_message = "TypeError"
        components.stack_trace = "Error at line 42"
        components.debug_code_snippet = "def sample():\n    return 1/0  # Error"
    
    def _add_optimization_components(
        self,
        components: QueryComponents,
        difficulty: DifficultyLevel,
        domain_data: Dict[str, Any]
    ) -> None:
        """Add optimization-specific components."""
        components.optimization_target = "performance"
        components.performance_issue = "slow execution"
        components.code_snippet = "def slow_function():\n    result = []\n    for i in range(1000):\n        for j in range(1000):\n            result.append(i*j)" 