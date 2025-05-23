"""
Query Component Module for PromptBuilder.

This module defines the core data structures and enumerations used throughout the
query generation system. It provides a comprehensive, type-safe foundation for 
creating, manipulating, and serializing training examples that demonstrate advanced
reasoning patterns for coding assistant LLMs.

Key features:
  - Type-safe enumerations for frameworks, query types, and difficulty levels
  - Comprehensive dataclasses for representing query components and metadata
  - JSON serialization and deserialization methods
  - Validation and creation helpers
  - Immutable example representation for training data

Classes:
    FrameworkType: Enumeration of reasoning framework types
    QueryType: Enumeration of supported query types
    DifficultyLevel: Enumeration of difficulty levels
    OutputFormat: Enumeration of output formats
    Section: Dataclass for framework sections
    QueryComponents: Dataclass for query component attributes
    QueryInfo: Dataclass for complete query information
    Example: Dataclass for complete training examples

Author: Made in Jurgistan
Version: 2.1.0
License: MIT
"""

from __future__ import annotations

import json
import uuid
import logging
from dataclasses import asdict, dataclass, field, fields as dataclass_fields
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union, cast, TypeVar, ClassVar

# Type variable for generic enum conversion methods
T = TypeVar('T', bound=Enum)


class FrameworkType(str, Enum):
    """Enumeration of primary reasoning framework types.
    
    These frameworks define different approaches to handling queries and generating responses,
    each optimized for specific query categories. Using string values enables easy serialization.
    
    Attributes:
        DEVELOPER_CLARIFICATION: Framework for explaining concepts and teaching
        CHAIN_OF_THOUGHT: Framework for systematic problem solving and implementation
        HYBRID_DECISION_MAKING: Framework for evaluating options and making recommendations
    """
    DEVELOPER_CLARIFICATION = "Developer Clarification Model"
    CHAIN_OF_THOUGHT = "Chain of Thought (CoT) Framework"
    HYBRID_DECISION_MAKING = "Advanced Hybrid Decision-Making Framework"

    def __str__(self) -> str:
        """String representation of the framework type."""
        return self.value
    
    @classmethod
    def from_str(cls, value: str) -> 'FrameworkType':
        """Create a FrameworkType from a string.
        
        Args:
            value: String representation of the framework type
            
        Returns:
            FrameworkType: Corresponding enum value
            
        Raises:
            ValueError: If the string doesn't match any framework type
        """
        value_lower = value.lower()
        for framework in cls:
            if framework.value.lower() == value_lower:
                return framework
        
        # Provide more helpful error message with valid options
        valid_values = [f.value for f in cls]
        raise ValueError(f"Unknown framework type: '{value}'. Valid values are: {valid_values}")

    @staticmethod
    def safe_from_str(value: str, default: FrameworkType) -> FrameworkType:
        """Safely create a FrameworkType from a string, returning default on error.
        
        Args:
            value: String representation of the framework type
            default: Default value to return if conversion fails
            
        Returns:
            FrameworkType: Corresponding enum value or default
        """
        try:
            return FrameworkType.from_str(value)
        except ValueError:
            logging.warning(f"Invalid framework type '{value}', using default: {default.value}")
            return default


class QueryType(str, Enum):
    """Enumeration of query types, covering diverse coding assistance scenarios.
    
    Each type represents a distinct category of user queries that the system can handle,
    enabling specialized response generation tailored to query characteristics.
    
    Attributes:
        CONCEPT_EXPLANATION: Explaining programming concepts
        PROBLEM_SOLVING: Solving computational problems
        DEBUGGING: Finding and fixing bugs
        OPTIMIZATION: Improving performance or efficiency
        IMPLEMENTATION_REQUESTS: Creating new code components
        STRATEGIC_DECISIONS: Making technology choices
        VAGUE_QUERIES: Handling ambiguous questions
        RECOMMENDATION: Recommending tools or approaches
        BEST_PRACTICES: Providing coding standards
        CODE_REVIEW: Analyzing and improving code
    """
    CONCEPT_EXPLANATION = "conceptExplanation"
    PROBLEM_SOLVING = "problemSolving"
    STRATEGIC_DECISIONS = "strategicDecisions"
    VAGUE_QUERIES = "vagueQueries"
    IMPLEMENTATION_REQUESTS = "implementationRequests"
    DEBUGGING = "debugging"
    OPTIMIZATION = "optimization"
    RECOMMENDATION = "recommendation"
    BEST_PRACTICES = "bestPractices"
    CODE_REVIEW = "codeReview"

    def __str__(self) -> str:
        """String representation of the query type."""
        return self.value
    
    @classmethod
    def from_str(cls, value: str) -> 'QueryType':
        """Create a QueryType from a string.
        
        Args:
            value: String representation of the query type
            
        Returns:
            QueryType: Corresponding enum value
            
        Raises:
            ValueError: If the string doesn't match any query type
        """
        value_lower = value.lower()
        
        # Check direct match
        for query_type in cls:
            if query_type.value.lower() == value_lower:
                return query_type
                
        # Check aliases (now global)
        if value_lower in QueryType_ALIASES:
            alias_value = QueryType_ALIASES[value_lower]
            return cls(alias_value)
            
        # Provide helpful error message with valid options
        valid_values = [f.value for f in cls]
        raise ValueError(f"Unknown query type: '{value}'. Valid values are: {valid_values}")
    
    @staticmethod
    def safe_from_str(value: str, default: QueryType) -> QueryType:
        """Safely create a QueryType from a string, returning default on error.
        
        Args:
            value: String representation of the query type
            default: Default value to return if conversion fails
            
        Returns:
            QueryType: Corresponding enum value or default
        """
        try:
            return QueryType.from_str(value)
        except ValueError:
            logging.warning(f"Invalid query type '{value}', using default: {default.value}")
            return default
    
    @property
    def display_name(self) -> str:
        """Get a human-readable display name for the query type.
        
        Returns:
            str: Display name with proper spacing and capitalization
        """
        # Convert camelCase to Title Case with Spaces
        result = ""
        for i, char in enumerate(self.value):
            if i > 0 and char.isupper():
                result += " "
            result += char
        return result.title()


class DifficultyLevel(str, Enum):
    """Enumeration of difficulty levels for generated content.
    
    These levels determine the complexity, depth, and sophistication of generated 
    responses, allowing fine-grained control over training example characteristics.
    
    Attributes:
        BASIC: Entry-level content for beginners
        INTERMEDIATE: Content for developers with some experience
        ADVANCED: Sophisticated content for experienced developers
    """
    BASIC = "basic"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"

    def __str__(self) -> str:
        """String representation of the difficulty level."""
        return self.value
    
    @classmethod
    def from_str(cls, value: str) -> 'DifficultyLevel':
        """Create a DifficultyLevel from a string.
        
        Args:
            value: String representation of the difficulty level
            
        Returns:
            DifficultyLevel: Corresponding enum value
            
        Raises:
            ValueError: If the string doesn't match any difficulty level
        """
        value_lower = value.lower()
        for level in cls:
            if level.value.lower() == value_lower:
                return level
                
        # Attempt to match by prefix (e.g., "int" -> "intermediate")
        for level in cls:
            if level.value.lower().startswith(value_lower):
                return level
                
        # Provide helpful error message with valid options
        valid_values = [f.value for f in cls]
        raise ValueError(f"Unknown difficulty level: '{value}'. Valid values are: {valid_values}")
    
    @staticmethod
    def safe_from_str(value: str, default: DifficultyLevel) -> DifficultyLevel:
        """Safely create a DifficultyLevel from a string, returning default on error.
        
        Args:
            value: String representation of the difficulty level
            default: Default value to return if conversion fails
            
        Returns:
            DifficultyLevel: Corresponding enum value or default
        """
        try:
            return DifficultyLevel.from_str(value)
        except ValueError:
            logging.warning(f"Invalid difficulty level '{value}', using default: {default.value}")
            return default
    
    @property
    def display_name(self) -> str:
        """Get a human-readable display name for the difficulty level.
        
        Returns:
            str: Display name with proper capitalization
        """
        return self.value.title()
    
    @property
    def complexity_factor(self) -> float:
        """Get a numeric factor representing the complexity of this level.
        
        Returns:
            float: Complexity factor (1.0 for basic, 1.5 for intermediate, 2.0 for advanced)
        """
        if self == DifficultyLevel.BASIC:
            return 1.0
        elif self == DifficultyLevel.INTERMEDIATE:
            return 1.5
        else:  # ADVANCED
            return 2.0


class OutputFormat(str, Enum):
    """Enumeration of supported output formats.
    
    Defines the available formats for saving generated training data,
    allowing flexibility in how examples are stored and consumed.
    
    Attributes:
        JSONL: JSON Lines format (one JSON object per line)
        CSV: Comma-separated values format
        MARKDOWN: Markdown format with rich formatting
        JSON: Standard JSON format (array of objects)
        YAML: YAML format for human-readable configuration
    """
    JSONL = "jsonl"
    CSV = "csv"
    MARKDOWN = "md"
    JSON = "json"
    YAML = "yaml"

    def __str__(self) -> str:
        """String representation of the output format."""
        return self.value
    
    @classmethod
    def from_str(cls, value: str) -> 'OutputFormat':
        """Create an OutputFormat from a string.
        
        Args:
            value: String representation of the output format
            
        Returns:
            OutputFormat: Corresponding enum value
            
        Raises:
            ValueError: If the string doesn't match any output format
        """
        value_lower = value.lower()
        
        # Special case for extensions with dot
        if value_lower.startswith('.'):
            value_lower = value_lower[1:]
            
        for output_format in cls:
            if output_format.value.lower() == value_lower:
                return output_format
                
        # Handle common aliases
        if value_lower in ["yml", "yaml"]:
            return cls.YAML
        elif value_lower in ["markdown"]:
            return cls.MARKDOWN
            
        # Provide helpful error message with valid options
        valid_values = [f.value for f in cls]
        raise ValueError(f"Unknown output format: '{value}'. Valid values are: {valid_values}")
    
    @staticmethod
    def safe_from_str(value: str, default: OutputFormat) -> OutputFormat:
        """Safely create an OutputFormat from a string, returning default on error.
        
        Args:
            value: String representation of the output format
            default: Default value to return if conversion fails
            
        Returns:
            OutputFormat: Corresponding enum value or default
        """
        try:
            return OutputFormat.from_str(value)
        except ValueError:
            logging.warning(f"Invalid output format '{value}', using default: {default.value}")
            return default
    
    @property
    def file_extension(self) -> str:
        """Get the file extension for this output format.
        
        Returns:
            str: File extension including the dot
        """
        return f".{self.value}"
    
    @property
    def content_type(self) -> str:
        """Get the MIME content type for this output format.
        
        Returns:
            str: MIME content type
        """
        return OutputFormat_CONTENT_TYPES.get(self.value, "text/plain")


@dataclass
class Section:
    """Represents a structured section within a reasoning framework.
    
    Sections organize reasoning steps into logical groups, promoting
    comprehensive and systematic response generation.
    
    Attributes:
        title: The section's title
        items: List of items or points in the section
    """
    title: str
    items: List[str]

    def __post_init__(self) -> None:
        """Validate the section after initialization."""
        if not self.title:
            raise ValueError("Section title cannot be empty")
        
        # Ensure items is a list
        if not isinstance(self.items, list):
            if isinstance(self.items, str):
                self.items = [self.items]
            else:
                try:
                    self.items = list(self.items)
                except (TypeError, ValueError):
                    raise ValueError(f"Section items must be a list, got {type(self.items)}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert the section to a dictionary.
        
        Returns:
            Dict[str, Any]: Dictionary representation of the section
        """
        return {
            "title": self.title,
            "items": self.items
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Section:
        """Create a section from a dictionary.
        
        Args:
            data: Dictionary representation of a section
            
        Returns:
            Section: New section object
            
        Raises:
            ValueError: If required fields are missing
        """
        if not isinstance(data, dict):
            raise ValueError(f"Section data must be a dictionary, got {type(data)}")
            
        if "title" not in data:
            raise ValueError("Section data must contain 'title' field")
        
        items = data.get("items", [])
        
        return cls(
            title=data["title"],
            items=items
        )
    
    def add_item(self, item: str) -> None:
        """Add an item to the section.
        
        Args:
            item: Item to add
        """
        if not item:
            return
        self.items.append(item)
    
    def merge(self, other: Section) -> Section:
        """Merge another section into this one.
        
        Args:
            other: Section to merge
            
        Returns:
            Section: New merged section
        """
        if self.title != other.title:
            raise ValueError(f"Cannot merge sections with different titles: '{self.title}' and '{other.title}'")
            
        return Section(
            title=self.title,
            items=self.items + [item for item in other.items if item not in self.items]
        )


@dataclass
class QueryComponents:
    """Contains components for a generated query with comprehensive attributes.
    
    This class maintains all the necessary components for generating
    realistic and context-aware queries and responses. It supports
    a wide range of query types and domains with specialized attributes.
    
    Attributes are organized by category to improve readability and maintainability.
    """
    # Core components
    concept: str = ""
    concept_definition: str = ""
    domain_area: str = ""
    related_concept: str = ""
    related_concept_definition: str = ""
    
    # Technology context
    technology_context: str = ""
    technology_version: str = ""
    language: str = ""
    framework: str = ""
    library: str = ""
    
    # User information
    expertise_level: str = ""
    
    # Problem details
    problem_area: str = ""
    problem_description: str = ""
    problem_details: str = ""
    problem_cause: str = ""
    problem_solution: str = ""
    problem_context: str = ""
    
    # System components
    system_component: str = ""
    trigger_condition: str = ""
    error_message: str = ""
    stack_trace: str = ""
    
    # User interactions
    action: str = ""
    user_reported_issue: str = ""
    user_action: str = ""
    
    # Performance and quality
    system_aspect: str = ""
    performance_issue: str = ""
    performance_characteristic: str = ""
    quality_attribute: str = ""
    quality_attributes: str = ""
    
    # Project context
    project_context: str = ""
    project_type: str = ""
    scale: str = ""
    business_context: str = ""
    
    # Options for decisions
    option_a: str = ""
    option_b: str = ""
    multiple_options: str = ""
    competing_concern_a: str = ""
    competing_concern_b: str = ""
    
    # Strategy elements
    strategic_option_a: str = ""
    strategic_option_b: str = ""
    long_term_goal: str = ""
    business_case: str = ""
    
    # Technical elements
    team_skills: str = ""
    technology: str = ""
    technology_stack: str = ""
    stack_components: List[str] = field(default_factory=list)
    system_type: str = ""
    
    # Architecture
    architectural_pattern_a: str = ""
    architectural_pattern_b: str = ""
    architectural_principles: str = ""
    
    # Constraints and requirements
    constraints: str = ""
    important_constraint: str = ""
    technical_specification: str = ""
    feature_requirement: str = ""
    compliance_requirement: str = ""
    compliance_details: str = ""
    compliance_implementation: str = ""
    
    # Vague queries
    vague_term: str = ""
    vague_term_interpretations: List[str] = field(default_factory=list)
    
    # Edge cases
    edge_cases: str = ""
    critical_issue: str = ""
    
    # Environment
    production_environment: str = ""
    environmental_context: str = ""
    
    # Domain-specific
    industry_context: str = ""
    domain: str = ""
    technical_area: str = ""
    
    # Implementation
    task_description: str = ""
    algorithm_name: str = ""
    application_layer: str = ""
    component_type: str = ""
    external_system: str = ""
    application_type: str = ""
    application_requirements: str = ""
    
    # Incorrect examples
    incorrect_domain: str = ""
    incorrect_assumption: str = ""
    unrelated_context: str = ""
    
    # Business aspects
    business_rules: str = ""
    business_problem: str = ""
    feature: str = ""
    
    # Code examples
    code_snippet: str = ""
    debug_code_snippet: str = ""
    optimized_code_snippet: str = ""
    example_code: str = ""
    unit_test: str = ""
    
    # Additional for specific query types
    optimization_target: str = ""
    recommendation_criteria: str = ""
    best_practice_area: str = ""
    alternative_concept: str = ""
    emerging_technology: str = ""
    technology_category: str = ""
    legacy_system: str = ""
    
    # Enhanced components for diverse query scenarios
    approach_a: str = ""
    approach_b: str = ""
    error_occurrence_pattern: str = ""
    impact_area: str = ""
    system_requirement: str = ""
    advanced_requirement: str = ""
    scale_characteristic: str = ""
    business_consideration: str = ""
    technical_consideration: str = ""
    strategic_capability: str = ""
    environment_type: str = ""
    deployment_environment: str = ""
    secondary_criteria: str = ""
    
    # For multi-turn dialogues
    follow_up_questions: List[str] = field(default_factory=list)
    clarification_points: List[str] = field(default_factory=list)
    
    # For information gaps
    information_gaps: List[Dict[str, str]] = field(default_factory=list)
    
    # For misinterpretation analysis
    potential_misinterpretations: List[Dict[str, str]] = field(default_factory=list)
    
    def __post_init__(self) -> None:
        """Validate and normalize component values after initialization."""
        # Ensure list fields are actually lists
        for field_name, field_type in self.__annotations__.items():
            if 'List' in str(field_type) and getattr(self, field_name) is None:
                setattr(self, field_name, [])

    def to_dict(self) -> Dict[str, Any]:
        """Convert the components to a dictionary, excluding empty values.
        
        Returns:
            Dict[str, Any]: Dictionary with non-empty component values
        """
        # Filter out empty values for cleaner output
        result = {}
        for key, value in asdict(self).items():
            # Skip empty strings, empty lists, empty dicts
            if value == "" or value == [] or value == {}:
                continue
            result[key] = value
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> QueryComponents:
        """Create components from a dictionary.
        
        Args:
            data: Dictionary representation of query components
            
        Returns:
            QueryComponents: New components object
        """
        if not isinstance(data, dict):
            raise ValueError(f"Components data must be a dictionary, got {type(data)}")
            
        # Create with only keys present in the dataclass to avoid errors
        valid_fields = {f.name for f in dataclass_fields(cls)}
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}
        
        return cls(**filtered_data)
    
    def merge(self, other: QueryComponents) -> QueryComponents:
        """Merge another QueryComponents instance into this one.
        
        Non-empty values from other will override values in this instance.
        Lists will be combined without duplicates.
        
        Args:
            other: QueryComponents to merge
            
        Returns:
            QueryComponents: New merged components
        """
        result_dict = self.to_dict()
        other_dict = other.to_dict()
        
        for key, value in other_dict.items():
            if isinstance(value, list):
                # Combine lists without duplicates
                existing = result_dict.get(key, [])
                if isinstance(existing, list):
                    result_dict[key] = existing + [v for v in value if v not in existing]
                else:
                    result_dict[key] = value
            elif value:  # Only override if value is not empty
                result_dict[key] = value
                
        return QueryComponents.from_dict(result_dict)
    
    def get_key_components(self) -> Dict[str, str]:
        """Get a dictionary of key components for query generation.
        
        Returns:
            Dict[str, str]: Dictionary with key components
        """
        key_components = {}
        
        # Add core components
        for field_name in ["concept", "language", "technology", "domain", "problem_area"]:
            value = getattr(self, field_name, None)
            if value:
                key_components[field_name] = value
                
        return key_components
    
    def validate_for_query_type(self, query_type: QueryType) -> List[str]:
        """Validate components for a specific query type.
        
        Args:
            query_type: Query type to validate for
            
        Returns:
            List[str]: List of missing required components, empty if valid
        """
        missing = []
        
        # Required components for all query types
        if not (self.concept or self.problem_area or self.technology):
            missing.append("At least one of concept, problem_area, or technology")
            
        # Query type-specific requirements
        if query_type == QueryType.CONCEPT_EXPLANATION and not self.concept:
            missing.append("concept for conceptExplanation query")
            
        elif query_type == QueryType.PROBLEM_SOLVING and not self.problem_area:
            missing.append("problem_area for problemSolving query")
            
        elif query_type == QueryType.DEBUGGING and not (self.code_snippet or self.debug_code_snippet):
            missing.append("code_snippet or debug_code_snippet for debugging query")
            
        elif query_type == QueryType.IMPLEMENTATION_REQUESTS and not self.task_description:
            missing.append("task_description for implementationRequests query")
            
        return missing


@dataclass
class QueryInfo:
    """Information about a query."""
    query_type: QueryType
    difficulty: DifficultyLevel
    components: QueryComponents
    framework: Optional[FrameworkType] = None
    query: str = ''
    dialogue_turns: List[Dict[str, str]] = field(default_factory=list)
    
    def __post_init__(self):
        """Initialize the query text if not provided."""
        if not self.query:
            self.query = self.generate_query_text()
    
    def generate_query_text(self) -> str:
        """Generate a query text based on the components."""
        if not self.components:
            return "How can I improve my code?"
        
        # Generate different query texts based on query type
        if self.query_type == QueryType.CONCEPT_EXPLANATION:
            concept = getattr(self.components, 'concept', 'this concept')
            language = getattr(self.components, 'language', 'programming')
            return f"Explain {concept} in {language}"
        
        elif self.query_type == QueryType.PROBLEM_SOLVING:
            language = getattr(self.components, 'language', 'code')
            problem_area = getattr(self.components, 'problem_area', 'issue')
            technology = getattr(self.components, 'technology', 'this technology')
            return f"Solve this {problem_area} in {language} using {technology}"
        
        elif self.query_type == QueryType.STRATEGIC_DECISIONS:
            language = getattr(self.components, 'language', 'programming')
            option_a = getattr(self.components, 'strategic_option_a', 'option A')
            option_b = getattr(self.components, 'strategic_option_b', 'option B')
            return f"Compare {option_a} vs {option_b} for {language} development"
        
        elif self.query_type == QueryType.DEBUGGING:
            language = getattr(self.components, 'language', 'code')
            error_message = getattr(self.components, 'error_message', 'error')
            technology = getattr(self.components, 'technology', 'this technology')
            return f"Debug this {error_message} error in my {language} code using {technology}"
        
        elif self.query_type == QueryType.OPTIMIZATION:
            language = getattr(self.components, 'language', 'code')
            target = getattr(self.components, 'optimization_target', 'performance')
            technology = getattr(self.components, 'technology', 'this technology')
            issue = getattr(self.components, 'performance_issue', '')
            query = f"Optimize my {language} code for better {target} when using {technology}"
            if issue:
                query += f" to solve {issue} issues"
            return query
        
        elif self.query_type == QueryType.VAGUE_QUERIES:
            language = getattr(self.components, 'language', 'programming')
            return f"How do I use {language} better?"
        
        elif self.query_type == QueryType.IMPLEMENTATION_REQUESTS:
            concept = getattr(self.components, 'concept', 'a feature')
            language = getattr(self.components, 'language', 'code')
            technology = getattr(self.components, 'technology', 'this technology')
            return f"How to implement {concept} in {language} using {technology}"
        
        elif self.query_type == QueryType.RECOMMENDATION:
            language = getattr(self.components, 'language', 'programming')
            tech_context = getattr(self.components, 'technology_context', 'my project')
            return f"Recommend the best {language} libraries for {tech_context}"
        
        elif self.query_type == QueryType.BEST_PRACTICES:
            language = getattr(self.components, 'language', 'programming')
            concept = getattr(self.components, 'concept', 'coding')
            return f"What are the best practices for {concept} in {language}?"
        
        elif self.query_type == QueryType.CODE_REVIEW:
            language = getattr(self.components, 'language', 'code')
            concept = getattr(self.components, 'concept', 'implementation')
            return f"Review my {concept} {language} code"
        
        else:
            return f"How can I work with {self.query_type.value} in {getattr(self.components, 'language', 'programming')}?"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'query_type': self.query_type.value,
            'difficulty': self.difficulty.value,
            'components': asdict(self.components),
            'query': self.query,
            'dialogue_turns': self.dialogue_turns
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'QueryInfo':
        """Create a QueryInfo from a dictionary."""
        if not isinstance(data, dict):
            raise ValueError(f"QueryInfo data must be a dictionary, got {type(data)}")

        # Handle both legacy and new keys
        query_type_value = data.get("query_type") or data.get("queryType") or QueryType.CONCEPT_EXPLANATION.value
        difficulty_value = data.get("difficulty") or DifficultyLevel.BASIC.value
        framework_value = data.get("framework") or None
        components_data = data.get("components", {})
        query = data.get("query", "")
        dialogue_turns = data.get("dialogue_turns") or data.get("dialogueTurns") or []

        # Convert string values to enums
        query_type = QueryType.safe_from_str(query_type_value, QueryType.CONCEPT_EXPLANATION)
        difficulty = DifficultyLevel.safe_from_str(difficulty_value, DifficultyLevel.BASIC)
        framework = None
        if framework_value is not None:
            framework = FrameworkType.safe_from_str(framework_value, FrameworkType.DEVELOPER_CLARIFICATION)

        components = (
            components_data if isinstance(components_data, QueryComponents)
            else QueryComponents.from_dict(components_data)
        )

        return cls(
            query_type=query_type,
            difficulty=difficulty,
            components=components,
            framework=framework,
            query=query,
            dialogue_turns=dialogue_turns
        )
    
    def validate(self) -> List[str]:
        """Validate the QueryInfo for completeness and consistency.
        
        Returns:
            List[str]: List of validation issues, empty if valid
        """
        issues = []
        # Validate query type and difficulty
        if not isinstance(self.query_type, QueryType):
            issues.append("query_type must be a QueryType enum value")
        if not isinstance(self.difficulty, DifficultyLevel):
            issues.append("difficulty must be a DifficultyLevel enum value")
        # Validate components
        if not isinstance(self.components, QueryComponents):
            issues.append("components must be a QueryComponents instance")
        else:
            issues.extend(self.components.validate_for_query_type(self.query_type))
        return issues


@dataclass
class Example:
    """A complete training example with feedback support.
    
    Represents a fully-formed training example for LLM fine-tuning,
    including query information, reasoning process, and responses.
    
    Attributes:
        id: Unique identifier for the example
        query_info: Complete query information
        internal_reasoning: Internal reasoning process
        external_response: Response provided to the user
        feedback_response: Optional feedback for dialogue
        created_at: Timestamp when the example was created
    """
    id: str
    query_info: QueryInfo
    internal_reasoning: str
    external_response: str
    feedback_response: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def __post_init__(self) -> None:
        """Validate the example after initialization."""
        if not self.id:
            self.id = str(uuid.uuid4())
            
        # Ensure query_info is a QueryInfo instance
        if not isinstance(self.query_info, QueryInfo):
            if isinstance(self.query_info, dict):
                self.query_info = QueryInfo.from_dict(self.query_info)
            else:
                raise ValueError(f"Query info must be a QueryInfo instance or dict, got {type(self.query_info)}")
                
    @classmethod
    def create(
        cls, 
        query_info: QueryInfo, 
        internal_reasoning: str, 
        external_response: str,
        feedback_response: str = ""
    ) -> Example:
        """Create a new example with a generated UUID.
        
        Factory method to create a new example with proper ID generation
        and validation.
        
        Args:
            query_info: Complete query information
            internal_reasoning: Internal reasoning process
            external_response: Response provided to the user
            feedback_response: Optional feedback for dialogue
            
        Returns:
            Example: A new example with a generated ID
            
        Raises:
            ValueError: If required fields are empty
        """
        # Validate required fields
        if not query_info or not isinstance(query_info, (QueryInfo, dict)):
            raise ValueError(f"QueryInfo must be provided and valid, got {type(query_info)}")
            
        if not internal_reasoning:
            raise ValueError("Internal reasoning cannot be empty")
            
        if not external_response:
            raise ValueError("External response cannot be empty")
            
        # Convert dict to QueryInfo if needed
        if isinstance(query_info, dict):
            query_info = QueryInfo.from_dict(query_info)
            
        return cls(
            id=str(uuid.uuid4()),
            query_info=query_info,
            internal_reasoning=internal_reasoning,
            external_response=external_response,
            feedback_response=feedback_response
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the example to a dictionary.
        
        Returns:
            Dict[str, Any]: Dictionary representation of the example
        """
        return {
            "id": self.id,
            "query": self.query_info.query,
            "internal_reasoning": self.internal_reasoning,
            "external_response": self.external_response,
            "feedback_response": self.feedback_response,
            "created_at": self.created_at,
            "metadata": {
                "difficulty": self.query_info.difficulty.value,
                "queryType": self.query_info.query_type.value,
                "components": self.query_info.components.to_dict(),
                "framework": self.query_info.framework.value if self.query_info.framework else None,
                "dialogueTurns": self.query_info.dialogue_turns
            }
        }
    
    def to_json(self, indent: Optional[int] = None) -> str:
        """Convert the example to a JSON string.
        
        Args:
            indent: Number of spaces for indentation (None for compact JSON)
            
        Returns:
            str: JSON string representation of the example
        """
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Example:
        """Create an example from a dictionary.
        
        Args:
            data: Dictionary representation of an example
            
        Returns:
            Example: New example object
            
        Raises:
            ValueError: If required fields are missing
        """
        if not isinstance(data, dict):
            raise ValueError(f"Example data must be a dictionary, got {type(data)}")
            
        # Check required fields
        required_fields = ["query", "internal_reasoning", "external_response"]
        missing = [field for field in required_fields if field not in data]
        if missing:
            raise ValueError(f"Missing required fields: {', '.join(missing)}")
        
        metadata = data.get("metadata", {})
        
        # Create QueryComponents from metadata
        components = QueryComponents.from_dict(metadata.get("components", {}))
        
        # Handle different key naming conventions
        query_type_value = metadata.get("queryType", metadata.get("query_type", QueryType.CONCEPT_EXPLANATION.value))
        difficulty_value = metadata.get("difficulty", DifficultyLevel.BASIC.value)
        framework_value = metadata.get("framework", FrameworkType.DEVELOPER_CLARIFICATION.value)
        dialogue_turns = metadata.get("dialogueTurns", metadata.get("dialogue_turns", []))
        
        # Convert string values to enums
        try:
            query_type = (
                QueryType(query_type_value) if isinstance(query_type_value, str) 
                else query_type_value
            )
            
            difficulty = (
                DifficultyLevel(difficulty_value) if isinstance(difficulty_value, str) 
                else difficulty_value
            )
            
            framework = (
                FrameworkType(framework_value) if isinstance(framework_value, str) 
                else framework_value
            )
        except ValueError as e:
            raise ValueError(f"Invalid enum value: {e}")
        
        # Create QueryInfo
        query_info = QueryInfo(
            query=data["query"],
            query_type=query_type,
            difficulty=difficulty,
            components=components,
            framework=framework,
            dialogue_turns=dialogue_turns
        )
        
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            query_info=query_info,
            internal_reasoning=data["internal_reasoning"],
            external_response=data["external_response"],
            feedback_response=data.get("feedback_response", ""),
            created_at=data.get("created_at", datetime.now().isoformat())
        )
    
    @classmethod
    def from_json(cls, json_str: str) -> Example:
        """Create an example from a JSON string.
        
        Args:
            json_str: JSON string representation of an example
            
        Returns:
            Example: New example object
            
        Raises:
            ValueError: If the JSON is invalid
        """
        try:
            data = json.loads(json_str)
            return cls.from_dict(data)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON: {e}")
        except Exception as e:
            raise ValueError(f"Error creating example from JSON: {e}")
    
    def validate(self) -> List[str]:
        """Validate the example for completeness and consistency.
        
        Returns:
            List[str]: List of validation issues, empty if valid
        """
        issues = []
        # Validate query info
        if hasattr(self.query_info, 'validate'):
            query_info_issues = self.query_info.validate()
            issues.extend([f"Query info issue: {issue}" for issue in query_info_issues])
        else:
            issues.append("Query info does not have a validate() method")
        # Validate internal reasoning
        if not self.internal_reasoning:
            issues.append("Internal reasoning cannot be empty")
        elif len(self.internal_reasoning.split()) < 50:
            issues.append("Internal reasoning is too short (< 50 words)")
        # Validate external response
        if not self.external_response:
            issues.append("External response cannot be empty")
        elif len(self.external_response.split()) < 50:
            issues.append("External response is too short (< 50 words)")
        # Check for placeholder content
        if "placeholder" in self.internal_reasoning.lower():
            issues.append("Internal reasoning contains placeholder content")
        if "placeholder" in self.external_response.lower():
            issues.append("External response contains placeholder content")
        return issues


# Map of aliases for QueryType (outside enum to avoid type annotation error)
QueryType_ALIASES: Dict[str, str] = {
    "concept": "conceptExplanation",
    "explain": "conceptExplanation",
    "solve": "problemSolving",
    "problem": "problemSolving",
    "decision": "strategicDecisions",
    "strategy": "strategicDecisions",
    "vague": "vagueQueries",
    "implement": "implementationRequests",
    "create": "implementationRequests",
    "debug": "debugging",
    "fix": "debugging",
    "optimize": "optimization",
    "improve": "optimization",
    "recommend": "recommendation",
    "suggest": "recommendation",
    "best practice": "bestPractices",
    "practices": "bestPractices",
    "review": "codeReview",
    "analyze": "codeReview"
}

# Map of content types for HTTP responses
# (moved out of enum body to avoid type annotation error)
OutputFormat_CONTENT_TYPES: Dict[str, str] = {
    "jsonl": "application/jsonl",
    "csv": "text/csv",
    "md": "text/markdown",
    "json": "application/json",
    "yaml": "application/yaml"
}

# Helper function to access dataclass fields
def fields(cls):
    """Get fields from a dataclass.
    
    Args:
        cls: Dataclass to get fields from
        
    Returns:
        List of field objects
    """
    return dataclass_fields(cls)
