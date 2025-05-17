"""
Template Manager Module for PromptBuilder.

This module handles loading, managing, and accessing templates for query generation.
It abstracts all template-related functionality for cleaner separation of concerns.

Usage:
    from promptbuilder.core.template_manager import TemplateManager
    template_manager = TemplateManager()
    template = template_manager.get_template(query_type, difficulty)

Author: Made in Jurgistan
Version: 3.0.0
License: MIT
"""

import json
import logging
import random
import re
from pathlib import Path
from typing import Dict, Optional, Any, List


from promptbuilder.core.query_components import QueryType, DifficultyLevel


class TemplateManager:
    """Manages query templates for example generation.
    
    This class handles loading, storing, and retrieving templates for
    different query types and difficulty levels. It provides a clean
    interface for accessing templates and supports both default templates
    and custom templates loaded from files.
    
    Attributes:
        templates: Dictionary mapping query types to difficulty levels to templates
        logger: Logger instance
    """
    
    def __init__(self, templates_path: Optional[str] = None):
        """Initialize the template manager.
        
        Args:
            templates_path: Optional path to a templates JSON file
        """
        self.logger = logging.getLogger(__name__)
        self.templates = self._load_templates(templates_path)
        self.logger.info("TemplateManager initialized with %d template categories", 
                        len(self.templates))
    
    def _load_templates(self, templates_path: Optional[str] = None) -> Dict[str, Dict[str, List[str]]]:
        """Load query templates from file or use defaults.
        
        Args:
            templates_path: Optional path to templates JSON file
            
        Returns:
            Dict[str, Dict[str, List[str]]]: Query templates by query type and difficulty
        """
        # Get default templates
        defaults = self._get_default_templates()
        
        # If no templates path provided, return defaults
        if not templates_path:
            self.logger.debug("No templates path provided, using defaults")
            return defaults
        
        # Attempt to load templates from file
        template_path = Path(templates_path) if templates_path else None
        if not template_path or not template_path.exists():
            self.logger.warning("Templates file not found: %s, using defaults", templates_path)
            return defaults
        
        try:
            self.logger.info("Loading templates from %s", template_path)
            with open(template_path, 'r', encoding='utf-8') as f:
                custom_templates = json.load(f)
                
            # Merge custom templates with defaults
            for query_type, difficulties in custom_templates.items():
                if query_type not in defaults:
                    defaults[query_type] = {}
                    
                for difficulty, templates in difficulties.items():
                    defaults[query_type][difficulty] = templates
                    
            return defaults
        except Exception as e:
            self.logger.error("Error loading templates: %s", e, exc_info=True)
            return defaults
    
    def _get_default_templates(self) -> Dict[str, Dict[str, List[str]]]:
        """Get default query templates optimized for programming assistance scenarios.
        
        Returns:
            Dict[str, Dict[str, List[str]]]: Default query templates by query type and difficulty
        """
        return {
            "CONCEPT_EXPLANATION": {
                "BASIC": [
                    "What is {concept} in {language}?",
                    "Can you explain {concept} in {language} for beginners?",
                    "How does {concept} work in {language}?",
                    "I'm new to {language}. What is {concept}?",
                    "Could you explain what {concept} means in {language} programming?"
                ],
                "INTERMEDIATE": [
                    "How does {concept} compare to {related_concept} in {language}?",
                    "What are the practical applications of {concept} in {language}?",
                    "Can you explain the implementation details of {concept} in {language}?",
                    "What are the best practices for using {concept} in {language}?",
                    "I understand the basics of {concept} in {language}, but how does it work under the hood?"
                ],
                "ADVANCED": [
                    "What are the advanced techniques for optimizing {concept} in {language}?",
                    "Can you explain the internal implementation of {concept} in {language} and its performance implications?",
                    "What are the edge cases and limitations when working with {concept} in {language}?",
                    "How has the implementation of {concept} evolved in recent versions of {language}?",
                    "What are the design considerations when implementing {concept} in a large-scale {language} application?"
                ]
            },
            "DEBUGGING": {
                "BASIC": [
                    "I'm getting this error: '{error_message}'. What does it mean?",
                    "My {language} code shows '{error_message}'. How do I fix it?",
                    "What does this error mean: '{error_message}'?",
                    "I'm a beginner with {language} and getting '{error_message}'. Help?",
                    "How do I resolve '{error_message}' in my {language} code?"
                ],
                "INTERMEDIATE": [
                    "I'm experiencing '{error_message}' when {error_occurrence_pattern}. What could be causing this?",
                    "My {language} application throws '{error_message}' in production but not locally. How can I debug this?",
                    "What are the common causes of '{error_message}' in {language} when working with {technology}?",
                    "How can I systematically debug this error: '{error_message}' in my {language} codebase?",
                    "I need help troubleshooting '{error_message}' that occurs {error_occurrence_pattern} in my {language} app."
                ],
                "ADVANCED": [
                    "We have an intermittent '{error_message}' in our high-load {language} service that only occurs {error_occurrence_pattern}. How can we identify the root cause?",
                    "Our {language} microservices architecture throws '{error_message}' during peak traffic. What advanced debugging techniques should we employ?",
                    "What could cause '{error_message}' in a multi-threaded {language} application under {error_occurrence_pattern} conditions?",
                    "We need to debug '{error_message}' in our distributed {language} system. What monitoring and tracing approaches would you recommend?",
                    "Our {language} application with {technology} shows '{error_message}' in the production environment. How can we implement advanced logging and diagnostics?"
                ]
            },
            "IMPLEMENTATION": {
                "BASIC": [
                    "How do I implement {feature_to_implement} in {language}?",
                    "What's the easiest way to add {feature_to_implement} to my {language} project?",
                    "Can you help me write code for {feature_to_implement} in {language}?",
                    "I need a simple implementation of {feature_to_implement} for my {language} application.",
                    "How would I create {feature_to_implement} using {language}?"
                ],
                "INTERMEDIATE": [
                    "What's an efficient way to implement {feature_to_implement} in {language} using {technology}?",
                    "How can I create a robust implementation of {feature_to_implement} in my {language} application?",
                    "What design patterns should I consider when implementing {feature_to_implement} in {language}?",
                    "How do I implement {feature_to_implement} in {language} that's both maintainable and efficient?",
                    "Can you provide a scalable approach to implementing {feature_to_implement} with {language}?"
                ],
                "ADVANCED": [
                    "What's the most performance-optimized way to implement {feature_to_implement} in {language} for a high-throughput environment?",
                    "How can I implement {feature_to_implement} in {language} that's thread-safe and handles concurrency correctly?",
                    "What architectural considerations should I address when implementing {feature_to_implement} in a distributed {language} system?",
                    "How would you implement {feature_to_implement} in {language} to ensure it's extensible and adaptable to future requirements?",
                    "What would be a highly optimized implementation of {feature_to_implement} in {language} that balances memory usage and processing speed?"
                ]
            },
            "STRATEGIC_DECISIONS": {
                "BASIC": [
                    "Should I use {strategic_option_a} or {strategic_option_b} for my {language} project?",
                    "What are the pros and cons of {strategic_option_a} versus {strategic_option_b} for {language}?",
                    "I'm deciding between {strategic_option_a} and {strategic_option_b} for my app. Which is better for {language}?",
                    "Can you compare {strategic_option_a} and {strategic_option_b} for a simple {language} application?",
                    "As a beginner in {language}, should I go with {strategic_option_a} or {strategic_option_b}?"
                ],
                "INTERMEDIATE": [
                    "What factors should I consider when choosing between {strategic_option_a} and {strategic_option_b} for a {language} project with {constraints}?",
                    "How do {strategic_option_a} and {strategic_option_b} compare in terms of developer productivity and performance for {language} applications?",
                    "For a growing {language} application, what are the long-term implications of choosing {strategic_option_a} over {strategic_option_b}?",
                    "Our team is considering switching from {strategic_option_a} to {strategic_option_b} for our {language} codebase. What are the migration challenges?",
                    "How do {strategic_option_a} and {strategic_option_b} compare for {language} projects requiring {constraints}?"
                ],
                "ADVANCED": [
                    "What are the architectural implications of choosing {strategic_option_a} versus {strategic_option_b} for a large-scale, distributed {language} system?",
                    "How do {strategic_option_a} and {strategic_option_b} compare in terms of performance, scalability, and maintainability for enterprise {language} applications?",
                    "What metrics should we use to evaluate {strategic_option_a} against {strategic_option_b} for our high-throughput {language} platform with {constraints}?",
                    "Our organization needs to standardize on either {strategic_option_a} or {strategic_option_b} for all {language} development. What factors should influence this strategic decision?",
                    "How would {strategic_option_a} and {strategic_option_b} handle the extreme scaling requirements of our {language} infrastructure while maintaining {long_term_goal}?"
                ]
            }
            # Additional query types can be added here
        }
    
    def get_template(self, query_type: QueryType, difficulty: DifficultyLevel) -> str:
        """Get a template for the specified query type and difficulty level.
        
        Args:
            query_type: Type of query to get template for
            difficulty: Difficulty level
            
        Returns:
            str: Template string
            
        Raises:
            ValueError: If no template exists for the query type and difficulty
        """
        qt_str = query_type.value
        diff_str = difficulty.value
        
        # Try to get templates for the query type and difficulty
        if qt_str in self.templates and diff_str in self.templates[qt_str]:
            templates = self.templates[qt_str][diff_str]
            if templates:
                return random.choice(templates)
        
        # If no specific template is found, try to get a template from another difficulty
        if qt_str in self.templates:
            all_templates = []
            for d, t in self.templates[qt_str].items():
                all_templates.extend(t)
            
            if all_templates:
                self.logger.warning("No template found for %s at %s difficulty. Using template from another difficulty.",
                                  qt_str, diff_str)
                return random.choice(all_templates)
        
        # If no template is found at all, return a generic template
        self.logger.warning("No template found for %s. Using generic template.", qt_str)
        return f"How can I work with {query_type.display_name.lower()} in {{language}}?"
    
    def format_template(self, template: str, components: Dict[str, Any]) -> str:
        """Format a template by replacing placeholders with component values.
        
        Args:
            template: Template string with placeholders
            components: Component values to substitute
            
        Returns:
            str: Formatted template string
        """
        # Make a copy of the components to avoid modifying the original
        formatted_components = components.copy()
        
        # Replace placeholders with component values
        try:
            # Find all placeholders in the template
            placeholders = re.findall(r'\{([^}]+)\}', template)
            
            # Replace missing components with default values
            for placeholder in placeholders:
                if placeholder not in formatted_components or not formatted_components[placeholder]:
                    formatted_components[placeholder] = self._get_default_value_for_placeholder(
                        placeholder, components.get('query_type', 'UNKNOWN'))
            
            # Format the template
            return template.format(**formatted_components)
        except KeyError as e:
            self.logger.error("Missing component for template formatting: %s", e, exc_info=True)
            # Try to format with what we have, ignoring missing keys
            for placeholder in components:
                template = template.replace("{" + placeholder + "}", str(components[placeholder]))
            return template
        except Exception as e:
            self.logger.error("Error formatting template: %s", e, exc_info=True)
            return template
    
    def _get_default_value_for_placeholder(self, placeholder: str, query_type: str) -> str:
        """Get a default value for a placeholder based on the query type.
        
        Args:
            placeholder: Placeholder name
            query_type: Query type
            
        Returns:
            str: Default value for the placeholder
        """
        # Common defaults for frequently used placeholders
        common_defaults = {
            "language": "Python",
            "technology": "a popular framework",
            "concept": "an important programming concept",
            "related_concept": "a related programming concept",
            "feature_to_implement": "a useful feature",
            "error_message": "an error message",
            "error_occurrence_pattern": "under certain conditions",
            "problem_area": "a challenging problem",
            "optimization_target": "performance",
            "strategic_option_a": "Option A",
            "strategic_option_b": "Option B",
            "constraints": "specific requirements",
            "long_term_goal": "maintainability and scalability"
        }
        
        # Query type specific defaults
        query_type_defaults = {
            "CONCEPT_EXPLANATION": {
                "concept": "object-oriented programming",
                "related_concept": "functional programming"
            },
            "DEBUGGING": {
                "error_message": "ImportError: No module named 'module_name'",
                "error_occurrence_pattern": "when importing modules"
            },
            "IMPLEMENTATION": {
                "feature_to_implement": "user authentication"
            },
            "STRATEGIC_DECISIONS": {
                "strategic_option_a": "REST API",
                "strategic_option_b": "GraphQL"
            },
            "OPTIMIZATION": {
                "optimization_target": "database query performance"
            },
            "PROBLEM_SOLVING": {
                "problem_area": "data processing efficiency"
            }
        }
        
        # First check if there's a query-type specific default
        if query_type in query_type_defaults and placeholder in query_type_defaults[query_type]:
            return query_type_defaults[query_type][placeholder]
        
        # Otherwise, use the common default if available
        if placeholder in common_defaults:
            return common_defaults[placeholder]
        
        # Finally, use a generic placeholder value
        return f"{placeholder.replace('_', ' ')}"