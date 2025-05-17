"""
Configuration module for PromptBuilder.

This module defines the configuration structure and provides utilities for loading,
validating, and managing configuration settings throughout the application. It
implements a flexible configuration system with validation, serialization, and
merging capabilities.

Key features:
  - Type-safe configuration through dataclasses
  - Comprehensive validation with detailed error reporting
  - Support for loading from JSON and YAML files
  - Default values for all settings
  - Merging of multiple configuration sources
  - Serialization to various formats

Classes:
    Config: Configuration dataclass with all generator settings

Functions:
    validate_config: Validate configuration settings
    load_config: Load configuration from a file
    merge_configs: Merge multiple configurations

Author: Made in Jurgistan
Version: 2.0.0
License: MIT
"""

import json
import os
import yaml
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Any, Union

from promptbuilder.core.query_components import (
    DifficultyLevel,
    QueryType,
    OutputFormat
)
from promptbuilder.utils.logging import (
    setup_logging,
    log_dict,
    get_logger
)


@dataclass
class Config:
    """Configuration for the PromptBuilder with enhanced validation options.
    
    This class holds all configuration parameters for the training example generator,
    including output settings, generation options, and validation settings.
    
    Attributes:
        num_examples_per_category: Number of examples to generate per combination
        difficulty_levels: List of difficulty levels to generate
        query_types: List of query types to generate
        output_format: Format for output files
        include_metadata: Whether to include metadata in output
        seed: Random seed for reproducibility
        output_file: Path to output file
        validation_output_file: Path to validation output file
        validate_output: Whether to validate generated examples
        selected_domains: List of domains to include (empty for all)
        validate_technology_mapping: Whether to validate technology mappings
        generate_functional_code: Whether to generate functional code examples
        validate_code_examples: Whether to validate code examples
        include_dialogue_turns: Whether to include dialogue turns
        include_feedback_loops: Whether to include feedback loops
        enforce_quality_assurance: Whether to enforce quality assurance
        use_parallel_processing: Whether to use parallel processing for generation
        max_workers: Maximum number of worker threads for parallel processing
        templates_path: Path to custom template files (None for defaults)
        fail_on_quality_issues: Whether to fail validation on quality issues
        verbose: Whether to enable verbose logging
        log_file: Path to log file (None for console only)
        technology_mapping: Mapping of domains to technologies, languages, and frameworks
    """
    # Generation parameters
    num_examples_per_category: int = 5
    difficulty_levels: List[DifficultyLevel] = field(
        default_factory=lambda: list(DifficultyLevel)
    )
    query_types: List[QueryType] = field(
        default_factory=lambda: list(QueryType)
    )
    
    # Output parameters
    output_format: OutputFormat = OutputFormat.JSONL
    include_metadata: bool = True
    output_file: str = "training_data.jsonl"
    validation_output_file: str = "validation_data.jsonl"
    
    # Filtering parameters
    selected_domains: List[str] = field(default_factory=list)
    
    # Quality control parameters
    validate_output: bool = False
    validate_technology_mapping: bool = True
    validate_code_examples: bool = True
    generate_functional_code: bool = True
    enforce_quality_assurance: bool = True
    fail_on_quality_issues: bool = False
    
    # Content generation parameters
    include_dialogue_turns: bool = True
    include_feedback_loops: bool = True
    
    # Performance parameters
    seed: int = 42
    use_parallel_processing: bool = False
    max_workers: Optional[int] = None
    
    # Advanced parameters
    templates_path: Optional[str] = None
    
    # Logging parameters
    verbose: bool = False
    log_file: Optional[str] = None
    
    # Technology mapping
    technology_mapping: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "web_development": {
            "languages": [
                {"name": "JavaScript", "version": "ES2021"},
                {"name": "TypeScript", "version": "4.5"},
                {"name": "Python", "version": "3.10"}
            ],
            "frameworks": [
                {"name": "React", "version": "18.0"},
                {"name": "Angular", "version": "13.0"},
                {"name": "Vue", "version": "3.0"},
                {"name": "Django", "version": "4.0"}
            ],
            "technologies": [
                {"name": "Next.js", "version": "12.0", "description": "React framework for SSR"},
                {"name": "Express", "version": "4.17", "description": "Node.js web framework"},
                {"name": "Flask", "version": "2.0", "description": "Python micro web framework"}
            ],
            "concepts": [
                "Server-side rendering",
                "RESTful APIs",
                "GraphQL",
                "Authentication",
                "State management",
                "Responsive design"
            ]
        },
        "data_science": {
            "languages": [
                {"name": "Python", "version": "3.10"},
                {"name": "R", "version": "4.1"}
            ],
            "frameworks": [
                {"name": "TensorFlow", "version": "2.7"},
                {"name": "PyTorch", "version": "1.10"},
                {"name": "scikit-learn", "version": "1.0"}
            ],
            "technologies": [
                {"name": "Pandas", "version": "1.3", "description": "Data analysis library"},
                {"name": "NumPy", "version": "1.21", "description": "Numerical computing library"},
                {"name": "Matplotlib", "version": "3.5", "description": "Data visualization library"}
            ],
            "concepts": [
                "Machine learning",
                "Deep learning",
                "Data preprocessing",
                "Feature engineering",
                "Model evaluation",
                "Natural language processing"
            ]
        }
    })
    
    def __str__(self) -> str:
        """String representation of the configuration.
        
        Returns a readable string representation showing key settings.
        
        Returns:
            str: String representation of configuration
        """
        items = []
        for key, value in self.__dict__.items():
            # Handle list fields differently to make the output more readable
            if isinstance(value, list) and len(value) > 3:
                items.append(f"{key}=[{len(value)} items]")
            elif isinstance(value, list) and all(hasattr(v, 'value') for v in value):
                # Handle enum lists by showing values
                items.append(f"{key}=[{', '.join(v.value for v in value)}]")
            elif hasattr(value, 'value'):
                # Handle enum values
                items.append(f"{key}={value.value}")
            else:
                items.append(f"{key}={value}")
        
        return f"Config({', '.join(items)})"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to a dictionary.
        
        Handles special conversion of enum values to strings
        for easier serialization.
        
        Returns:
            Dict[str, Any]: Dictionary representation of the configuration
        """
        result = {}
        for key, value in self.__dict__.items():
            # Convert enum values and lists of enum values to strings
            if isinstance(value, list) and value and hasattr(value[0], 'value'):
                result[key] = [item.value for item in value]
            elif hasattr(value, 'value'):
                result[key] = value.value
            else:
                result[key] = value
        
        return result
    
    def to_json(self, indent: int = 2) -> str:
        """Convert the configuration to a JSON string.
        
        Args:
            indent: Number of spaces for indentation
            
        Returns:
            str: JSON representation of the configuration
        """
        return json.dumps(self.to_dict(), indent=indent)
    
    def to_yaml(self) -> str:
        """Convert the configuration to a YAML string.
        
        Returns:
            str: YAML representation of the configuration
        """
        return yaml.dump(self.to_dict(), sort_keys=False)
    
    def save(self, file_path: Union[str, Path]) -> None:
        """Save configuration to a file.
        
        Args:
            file_path: Path to save the configuration to
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        if file_path.suffix.lower() == '.json':
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(self.to_json())
        elif file_path.suffix.lower() in ('.yaml', '.yml'):
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(self.to_yaml())
        else:
            # Default to JSON
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(self.to_json())
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Config':
        """Create a configuration object from a dictionary.
        
        Args:
            data: Dictionary representation of configuration
            
        Returns:
            Config: Configuration object
            
        Raises:
            ValueError: If the data is invalid
        """
        # Convert string values to enums
        if "difficulty_levels" in data and data["difficulty_levels"]:
            if isinstance(data["difficulty_levels"][0], str):
                data["difficulty_levels"] = [
                    DifficultyLevel(d) for d in data["difficulty_levels"]
                ]
        
        if "query_types" in data and data["query_types"]:
            if isinstance(data["query_types"][0], str):
                data["query_types"] = [
                    QueryType(q) for q in data["query_types"]
                ]
        
        if "output_format" in data and isinstance(data["output_format"], str):
            data["output_format"] = OutputFormat(data["output_format"])
        
        # Create the object
        try:
            return cls(**data)
        except TypeError as e:
            raise ValueError(f"Invalid configuration data: {e}")
    
    @classmethod
    def default(cls) -> 'Config':
        """Create a configuration object with default values.
        
        Returns:
            Config: Default configuration object
        """
        return cls()


def validate_config(config: Config) -> List[str]:
    """Validate the provided configuration.
    
    Performs comprehensive validation of the configuration to ensure all
    settings are valid and consistent with each other.
    
    Args:
        config: Configuration to validate
        
    Returns:
        List[str]: List of validation errors, empty if valid
    """
    errors = []
    
    # Validate number of examples
    if config.num_examples_per_category < 1:
        errors.append("Number of examples per category must be at least 1")
    
    # Validate difficulty levels
    if not config.difficulty_levels:
        errors.append("At least one difficulty level must be selected")
    
    # Validate query types
    if not config.query_types:
        errors.append("At least one query type must be selected")
    
    # Validate output file
    if not config.output_file:
        errors.append("Output file must be specified")
    
    # Validate file extension matches format
    output_ext = os.path.splitext(config.output_file)[1].lower()
    format_extensions = {
        OutputFormat.JSONL: '.jsonl',
        OutputFormat.CSV: '.csv',
        OutputFormat.MARKDOWN: '.md'
    }
    expected_ext = format_extensions.get(config.output_format)
    
    if output_ext and expected_ext and output_ext != expected_ext:
        errors.append(f"Output file extension '{output_ext}' does not match format '{expected_ext}'")
    
    # Validate validation output file if validation is enabled
    if config.validate_output and not config.validation_output_file:
        errors.append("Validation output file must be specified when validation is enabled")
    
    # Validate templates path if specified
    if config.templates_path and not os.path.exists(config.templates_path):
        errors.append(f"Templates path does not exist: {config.templates_path}")
    
    # Validate worker count for parallel processing
    if config.use_parallel_processing and config.max_workers is not None:
        if config.max_workers < 1:
            errors.append("Maximum workers must be at least 1")
    
    # Validate log file path
    if config.log_file:
        log_dir = os.path.dirname(config.log_file)
        if log_dir and not os.path.exists(log_dir):
            errors.append(f"Log directory does not exist: {log_dir}")
    
    return errors


def load_config(file_path: Union[str, Path]) -> Config:
    """Load configuration from a file.
    
    Loads configuration from a JSON or YAML file, with automatic format detection
    based on file extension.
    
    Args:
        file_path: Path to the configuration file
        
    Returns:
        Config: Loaded configuration
        
    Raises:
        FileNotFoundError: If the file does not exist
        ValueError: If the file contains invalid JSON or YAML
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        if file_path.suffix.lower() in ('.yaml', '.yml'):
            try:
                data = yaml.safe_load(f)
            except yaml.YAMLError as e:
                raise ValueError(f"Invalid YAML in configuration file: {e}")
        else:  # Default to JSON
            try:
                data = json.load(f)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in configuration file: {e}")
    
    # Validate the loaded data
    if not isinstance(data, dict):
        raise ValueError("Configuration file must contain a JSON or YAML object")
    
    # Create and return the configuration object
    return Config.from_dict(data)


def merge_configs(*configs: Config) -> Config:
    """Merge multiple configurations.
    
    Later configurations override earlier ones. This allows for layering
    configurations from different sources.
    
    Args:
        *configs: Configurations to merge
        
    Returns:
        Config: Merged configuration
    """
    merged_dict = {}
    
    # Merge dictionaries from first to last
    for config in configs:
        merged_dict.update(asdict(config))
    
    # Convert the merged dictionary back to a Config object
    return Config.from_dict(merged_dict)


def create_config_from_file(
    file_path: Union[str, Path],
    overrides: Optional[Dict[str, Any]] = None
) -> Config:
    """Create a configuration from a file with optional overrides.
    
    Args:
        file_path: Path to the configuration file
        overrides: Optional dictionary of values to override
        
    Returns:
        Config: Configuration object
        
    Raises:
        FileNotFoundError: If the file does not exist
        ValueError: If the file format is invalid
    """
    # Load base configuration
    config = load_config(file_path)
    
    # Apply overrides if provided
    if overrides:
        override_dict = {k: v for k, v in overrides.items() if v is not None}
        override_config = Config.from_dict(override_dict)
        config = merge_configs(config, override_config)
    
    return config
