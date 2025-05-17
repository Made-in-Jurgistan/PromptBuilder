"""
Framework Registry Module for PromptBuilder.

This module provides a registry for reasoning frameworks used in the example
generation process. It maps query types to appropriate reasoning frameworks
to ensure each query is processed using the most effective approach.

Usage:
    from promptbuilder.core.framework_registry import FrameworkRegistry
    registry = FrameworkRegistry()
    framework = registry.get_appropriate_framework(query_type)

Author: Made in Jurgistan
Version: 2.1.0
License: MIT
"""

import logging
from typing import Dict, Any, Optional

from promptbuilder.core.query_components import QueryType, FrameworkType


class FrameworkRegistry:
    """Registry for reasoning frameworks.

    This class provides a mapping between query types and reasoning frameworks,
    ensuring that each type of query is addressed using the most appropriate
    reasoning approach.

    Attributes:
        framework_mapping: Mapping from query types to framework types
        default_framework: Default framework to use if no mapping exists
        logger: Logger instance
    """

    def __init__(self, default_framework: FrameworkType = FrameworkType.DEVELOPER_CLARIFICATION):
        """Initialize the framework registry with default mappings.

        Args:
            default_framework: Default framework to use if no mapping exists (optional)
        """
        self.logger = logging.getLogger(__name__)
        self.framework_mapping = {
            QueryType.CONCEPT_EXPLANATION: FrameworkType.DEVELOPER_CLARIFICATION,
            QueryType.PROBLEM_SOLVING: FrameworkType.CHAIN_OF_THOUGHT,
            QueryType.DEBUGGING: FrameworkType.CHAIN_OF_THOUGHT,
            QueryType.OPTIMIZATION: FrameworkType.CHAIN_OF_THOUGHT,
            QueryType.IMPLEMENTATION_REQUESTS: FrameworkType.CHAIN_OF_THOUGHT,
            QueryType.STRATEGIC_DECISIONS: FrameworkType.HYBRID_DECISION_MAKING,
            QueryType.VAGUE_QUERIES: FrameworkType.HYBRID_DECISION_MAKING,
            QueryType.RECOMMENDATION: FrameworkType.HYBRID_DECISION_MAKING,
            QueryType.BEST_PRACTICES: FrameworkType.DEVELOPER_CLARIFICATION,
            QueryType.CODE_REVIEW: FrameworkType.HYBRID_DECISION_MAKING
        }
        self.default_framework = default_framework
        self.logger.info("FrameworkRegistry initialized with default framework: %s", default_framework.value)

    def get_appropriate_framework(self, query_type: QueryType) -> FrameworkType:
        """Get the appropriate framework for a query type.

        Args:
            query_type: Type of query to get framework for

        Returns:
            FrameworkType: Appropriate framework for the query type
        """
        framework = self.framework_mapping.get(query_type, self.default_framework)
        self.logger.debug("Selected framework %s for query type %s", framework.value, query_type.value)
        return framework

    def select_framework(self, query_type: QueryType) -> FrameworkType:
        """Select the appropriate framework for a query type.
        
        This is an alias for get_appropriate_framework for better API compatibility.

        Args:
            query_type: Type of query to get framework for

        Returns:
            FrameworkType: Appropriate framework for the query type
        """
        return self.get_appropriate_framework(query_type)

    def register_framework(self, query_type: QueryType, framework_type: FrameworkType) -> None:
        """Register a framework for a query type.

        Args:
            query_type: Type of query to register framework for
            framework_type: Framework to use for the query type

        Raises:
            ValueError: If query_type or framework_type is invalid
        """
        if not isinstance(query_type, QueryType):
            raise ValueError(f"Invalid query_type: {query_type}, must be a QueryType enum")
        if not isinstance(framework_type, FrameworkType):
            raise ValueError(f"Invalid framework_type: {framework_type}, must be a FrameworkType enum")
        
        self.framework_mapping[query_type] = framework_type
        self.logger.info("Registered framework %s for query type %s", framework_type.value, query_type.value)

    def get_all_frameworks(self) -> Dict[str, str]:
        """Get all registered frameworks.

        Returns:
            Dict[str, str]: Mapping from query type names to framework type names
        """
        frameworks = {
            query_type.value: framework_type.value
            for query_type, framework_type in self.framework_mapping.items()
        }
        self.logger.debug("Retrieved all registered frameworks: %s", frameworks)
        return frameworks
    
    def update_framework(self, query_type: QueryType, framework_type: FrameworkType) -> None:
        """Update an existing framework mapping for a query type.
        
        Args:
            query_type: Type of query to update framework for
            framework_type: New framework to use for the query type
            
        Raises:
            ValueError: If query_type or framework_type is invalid
            KeyError: If query_type is not already registered
        """
        if not isinstance(query_type, QueryType):
            raise ValueError(f"Invalid query_type: {query_type}, must be a QueryType enum")
        if not isinstance(framework_type, FrameworkType):
            raise ValueError(f"Invalid framework_type: {framework_type}, must be a FrameworkType enum")
        
        if query_type not in self.framework_mapping:
            raise KeyError(f"Query type {query_type.value} not found in framework mapping")
        
        self.framework_mapping[query_type] = framework_type
        self.logger.info("Updated framework for query type %s to %s", query_type.value, framework_type.value)
    
    def remove_framework(self, query_type: QueryType) -> None:
        """Remove a framework mapping for a query type.
        
        Args:
            query_type: Type of query to remove framework for
            
        Raises:
            ValueError: If query_type is invalid
            KeyError: If query_type is not registered
        """
        if not isinstance(query_type, QueryType):
            raise ValueError(f"Invalid query_type: {query_type}, must be a QueryType enum")
        
        if query_type not in self.framework_mapping:
            raise KeyError(f"Query type {query_type.value} not found in framework mapping")
        
        del self.framework_mapping[query_type]
        self.logger.info("Removed framework mapping for query type %s", query_type.value)
    
    def reset_to_defaults(self) -> None:
        """Reset the framework mapping to the default configuration."""
        self.__init__(self.default_framework)
        self.logger.info("Reset framework registry to default configuration")
