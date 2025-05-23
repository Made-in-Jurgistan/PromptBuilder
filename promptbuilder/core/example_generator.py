"""
Example Generator Module for PromptBuilder.

This module provides the public API and CLI entry point for the Example Generator.
It uses a modular architecture with components for different aspects of generation.

Usage (as a module):
    from promptbuilder.core.example_generator import ExampleGenerator
    ...

Usage (CLI):
    python -m promptbuilder.core.example_generator --config config.json

Author: Made in Jurgistan
Version: 2.1.0
License: MIT
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Union, Tuple

from promptbuilder.core.query_components import Example, QueryType, DifficultyLevel
from promptbuilder.core.generation_pipeline import GenerationPipeline
from promptbuilder.config import Config
from promptbuilder.core.framework_registry import FrameworkRegistry
from promptbuilder.core.template_manager import TemplateManager
from promptbuilder.core.component_generator import ComponentGenerator
from promptbuilder.core.response_generator import ResponseGenerator


def build_generator(config_path: Optional[str] = None) -> 'ExampleGenerator':
    """Factory function to instantiate ExampleGenerator from config file.
    
    Args:
        config_path: Optional path to configuration file
        
    Returns:
        ExampleGenerator: Configured example generator
        
    Raises:
        FileNotFoundError: If config file not found
        json.JSONDecodeError: If config file has invalid JSON
        ValueError: If config has invalid values
    """
    try:
        if config_path:
            # Use the correct config loading method
            from promptbuilder.config import load_config
            config = load_config(config_path)
        else:
            config = Config()
        return ExampleGenerator(config)
    except FileNotFoundError as e:
        logging.error("Config file not found: %s", e)
        raise
    except json.JSONDecodeError as e:
        logging.error("Invalid JSON in config file: %s", e)
        raise
    except ValueError as e:
        logging.error("Invalid configuration values: %s", e)
        raise
    except Exception as e:
        logging.error("Failed to load config: %s", e)
        raise


def main():
    """CLI entry point for generating examples.
    
    This function parses command line arguments, configures logging,
    instantiates the generator, generates examples, validates them if requested,
    and saves the results to a JSON file.
    """
    parser = argparse.ArgumentParser(description="PromptBuilder Example Generator CLI")
    parser.add_argument('--config', type=str, help='Path to config file (JSON)')
    parser.add_argument('--output', type=str, help='Path to output file (JSON)', default='examples.json')
    parser.add_argument('--log-level', type=str, default='INFO', 
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)')
    parser.add_argument('--validate', action='store_true', help='Validate examples after generation')
    parser.add_argument('--count', type=int, default=None, help='Number of examples to generate')
    parser.add_argument('--format', type=str, default='json', choices=['json', 'csv', 'jsonl'],
                        help='Output format (json, csv, jsonl)')
    args = parser.parse_args()

    # Configure logging
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('example_generator.log')
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info("Starting CLI with args: %s", vars(args))

    try:
        # Build the generator
        generator = build_generator(args.config)
        logger.info("Generator built successfully")
        
        # Generate examples
        if args.count:
            examples = generator.generate_examples(count=args.count)
        else:
            examples = generator.generate_examples()
        logger.info("Generated %d examples", len(examples))

        # Validate examples if requested
        if args.validate:
            from promptbuilder.core.example_validator import ExampleValidator
            validator = ExampleValidator(
                supported_languages=generator.supported_languages,
                supported_technologies=generator.supported_technologies
            )
            # Convert dicts to Example objects if needed
            examples = generator._ensure_examples_are_objects(examples)
            valid_examples, metrics = validator.validate_examples(examples)
            logger.info("Validation results: %s", metrics)
            examples = valid_examples

        # Create output directory if it doesn't exist
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save examples in the specified format
        if args.format == 'json':
            with output_path.open('w', encoding='utf-8') as f:
                json.dump([ex if isinstance(ex, dict) else ex.to_dict() for ex in examples], f, indent=2, ensure_ascii=False)
        elif args.format == 'jsonl':
            with output_path.open('w', encoding='utf-8') as f:
                for ex in examples:
                    f.write(json.dumps(ex if isinstance(ex, dict) else ex.to_dict(), ensure_ascii=False) + '\n')
        elif args.format == 'csv':
            try:
                import csv
                # Ensure all are Example objects for CSV
                examples_obj = generator._ensure_examples_are_objects(examples)
                with output_path.open('w', newline='', encoding='utf-8') as f:
                    if examples_obj:
                        # Get field names from the first example
                        fieldnames = examples_obj[0].to_dict().keys()
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        writer.writeheader()
                        for ex in examples_obj:
                            writer.writerow(ex.to_dict())
            except ImportError:
                logger.error("CSV format requires csv module, which is not available")
                sys.exit(1)
        
        logger.info("Saved %d examples to %s", len(examples), output_path)
        print(f"Generated {len(examples)} examples and saved to {output_path}")

    except KeyboardInterrupt:
        logger.info("Generation interrupted by user")
        print("\nGeneration interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error("CLI failed: %s", e, exc_info=True)
        print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()


class ExampleGenerator:
    """Generates sophisticated training examples for coding assistant LLMs.

    This class acts as a facade for the example generation process, delegating
    to specialized components for different aspects of generation.

    Attributes:
        config: Generator configuration
        pipeline: Generation pipeline orchestrator
        logger: Logger instance
        supported_languages: Set of supported programming languages
        supported_technologies: Set of supported technologies
    """

    def __init__(self, config=None):
        """Initialize the example generator.
        
        Args:
            config: Configuration for the generator
        """
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        
        # Extract supported languages and technologies from data
        self.supported_languages = self._extract_supported_languages()
        self.supported_technologies = self._extract_supported_technologies()
        
        self.logger.debug("Extracted supported languages: %s", set(self.supported_languages))
        self.logger.debug("Extracted supported technologies: %s", set(self.supported_technologies))
        
        # Set up the pipeline components
        framework_registry = FrameworkRegistry()
        template_manager = TemplateManager(getattr(config, 'templates_path', None) if hasattr(config, 'templates_path') else None)
        component_generator = ComponentGenerator()
        response_generator = ResponseGenerator(framework_registry)
        
        # Initialize the generation pipeline
        self.pipeline = GenerationPipeline(
            framework_registry=framework_registry,
            component_generator=component_generator,
            template_manager=template_manager,
            response_generator=response_generator,
            config_name=getattr(config, 'name', 'default') if hasattr(config, 'name') else 'default'
        )
        
        self.logger.info("ExampleGenerator initialized with %d languages and %d technologies",
                        len(self.supported_languages), len(self.supported_technologies))

    def _extract_supported_languages(self) -> Set[str]:
        """Extract supported programming languages from the configuration.
        
        Returns:
            Set[str]: Set of supported programming languages
        """
        languages = set()
        default_languages = {"python", "javascript", "typescript", "r"}
        config = self.config
        # Support both dict and object config
        tech_mapping = getattr(config, 'technology_mapping', None)
        if tech_mapping is None and isinstance(config, dict):
            tech_mapping = config.get('technology_mapping', None)
        if tech_mapping:
            for domain_data in tech_mapping.values():
                # domain_data can be dict or list
                if isinstance(domain_data, dict):
                    langs = domain_data.get('languages', [])
                    if isinstance(langs, list):
                        for lang in langs:
                            if isinstance(lang, dict) and 'name' in lang:
                                languages.add(lang['name'].lower())
                            elif isinstance(lang, str):
                                languages.add(lang.lower())
        return languages or default_languages

    def _extract_supported_technologies(self) -> Set[str]:
        """Extract supported technologies from the configuration.
        
        Returns:
            Set[str]: Set of supported technologies
        """
        technologies = set()
        default_technologies = {
            "matplotlib", "numpy", "pandas", "pytorch", "scikit-learn", "tensorflow",
            "react", "vue", "angular", "next.js",
            "express", "flask", "django"
        }
        config = self.config
        tech_mapping = getattr(config, 'technology_mapping', None)
        if tech_mapping is None and isinstance(config, dict):
            tech_mapping = config.get('technology_mapping', None)
        if tech_mapping:
            for domain_data in tech_mapping.values():
                if isinstance(domain_data, dict):
                    techs = domain_data.get('technologies', [])
                    if isinstance(techs, list):
                        for tech in techs:
                            if isinstance(tech, dict) and 'name' in tech:
                                technologies.add(tech['name'].lower())
                            elif isinstance(tech, str):
                                technologies.add(tech.lower())
        return technologies or default_technologies

    def generate_examples(
        self,
        count: int = 1,
        domains: Optional[List[str]] = None,
        difficulties: Optional[List[str]] = None,
        query_types: Optional[List[str]] = None,
        parallel: bool = False
    ) -> List[Dict[str, Any]]:
        """Generate training examples.
        
        Args:
            count: Number of examples to generate per category
            domains: List of domains to generate examples for
            difficulties: List of difficulty levels to generate examples for
            query_types: List of query types to generate examples for
            parallel: Whether to use parallel processing
            
        Returns:
            List[Dict[str, Any]]: List of generated examples
        """
        self.logger.info("Starting example generation process")
        
        # Set up query types
        selected_query_types = []
        if query_types:
            for qt in query_types:
                try:
                    selected_query_types.append(QueryType.from_str(qt))
                except ValueError:
                    self.logger.warning("Ignoring unknown query type: %s", qt)
        
        # Set up difficulty levels
        selected_difficulties = []
        if difficulties:
            difficulty_mapping = {
                "basic": DifficultyLevel.BASIC,
                "intermediate": DifficultyLevel.INTERMEDIATE,
                "advanced": DifficultyLevel.ADVANCED
            }
            for d in difficulties:
                if d.lower() in difficulty_mapping:
                    selected_difficulties.append(difficulty_mapping[d.lower()])
                else:
                    self.logger.warning("Ignoring unknown difficulty level: %s", d)
        
        # Generate examples using the pipeline
        return self.pipeline.generate_examples(
            count=count,
            domains=domains,
            difficulties=selected_difficulties,
            query_types=selected_query_types,
            parallel=parallel
        )

    def get_statistics(self) -> Dict[str, Any]:
        """Get generation statistics.
        
        Retrieves statistics about the generated examples from the pipeline.
        
        Returns:
            Dict[str, Any]: Statistics dictionary with counts by type, 
                difficulty, domain, etc.
        """
        stats = self.pipeline.get_statistics()
        self.logger.debug("Retrieved statistics: %s", stats)
        return stats
    
    def export_examples(self, examples: List[Any], output_path: str, format: str = 'json') -> None:
        """Export examples to a file in the specified format."""
        if not examples:
            self.logger.warning("No examples to export")
            return
        self.logger.info("Exporting %d examples to %s in %s format", len(examples), output_path, format)
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            if format == 'json':
                with path.open('w', encoding='utf-8') as f:
                    json.dump([ex.to_dict() if hasattr(ex, 'to_dict') else ex for ex in examples], f, indent=2, ensure_ascii=False)
            elif format == 'jsonl':
                with path.open('w', encoding='utf-8') as f:
                    for ex in examples:
                        f.write(json.dumps(ex.to_dict() if hasattr(ex, 'to_dict') else ex, ensure_ascii=False) + '\n')
            elif format == 'csv':
                import csv
                # Ensure all are Example objects for CSV
                examples_obj = self._ensure_examples_are_objects(examples)
                with path.open('w', newline='', encoding='utf-8') as f:
                    fieldnames = examples_obj[0].to_dict().keys()
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    for ex in examples_obj:
                        writer.writerow(ex.to_dict())
            else:
                raise ValueError(f"Unsupported export format: {format}")
            self.logger.info("Successfully exported examples to %s", output_path)
        except Exception as e:
            self.logger.error("Failed to export examples: %s", e, exc_info=True)
            raise

    def filter_examples(self, examples: List[Any], **filters) -> List[Any]:
        """Filter examples based on various criteria."""
        self.logger.info("Filtering %d examples with criteria: %s", len(examples), filters)
        def get_attr(ex, attr):
            # Try nested access for Example/query_info/components, else dict
            if hasattr(ex, 'query_info'):
                # Example object
                if hasattr(ex.query_info, 'components'):
                    comp = ex.query_info.components
                    if hasattr(comp, attr):
                        return getattr(comp, attr)
                if hasattr(ex.query_info, attr):
                    return getattr(ex.query_info, attr)
            if hasattr(ex, attr):
                return getattr(ex, attr)
            if isinstance(ex, dict):
                if attr in ex:
                    return ex[attr]
                # Try nested dicts
                if 'query_info' in ex and isinstance(ex['query_info'], dict):
                    if attr in ex['query_info']:
                        return ex['query_info'][attr]
                    if 'components' in ex['query_info'] and isinstance(ex['query_info']['components'], dict):
                        if attr in ex['query_info']['components']:
                            return ex['query_info']['components'][attr]
            return None
        
        filtered = examples
        for key, value in filters.items():
            if not value:
                continue
            if key == 'query_type':
                filtered = [ex for ex in filtered if get_attr(ex, 'query_type') == value]
            elif key == 'difficulty':
                filtered = [ex for ex in filtered if get_attr(ex, 'difficulty') == value]
            elif key == 'domain':
                filtered = [ex for ex in filtered if get_attr(ex, 'domain') == value]
            elif key == 'language':
                filtered = [ex for ex in filtered if (get_attr(ex, 'language') or '').lower() == value.lower()]
            elif key == 'technology':
                filtered = [ex for ex in filtered if (get_attr(ex, 'technology') or '').lower() == value.lower()]
            elif key == 'min_token_count':
                filtered = [ex for ex in filtered if get_attr(ex, 'token_count') is not None and get_attr(ex, 'token_count') >= value]
            elif key == 'max_token_count':
                filtered = [ex for ex in filtered if get_attr(ex, 'token_count') is not None and get_attr(ex, 'token_count') <= value]
            elif key == 'has_code_snippet':
                has_code = bool(value)
                filtered = [ex for ex in filtered if bool(get_attr(ex, 'code_snippet')) == has_code]
            elif key == 'framework':
                filtered = [ex for ex in filtered if (get_attr(ex, 'framework') or '').lower() == value.lower()]
            elif key == 'expertise_level':
                filtered = [ex for ex in filtered if get_attr(ex, 'expertise_level') == value]
            elif key == 'custom':
                if callable(value):
                    filtered = [ex for ex in filtered if value(ex)]
        self.logger.info("Filtered to %d examples", len(filtered))
        return filtered

    def _generate_code_snippet(self, language, technology, difficulty, concept="", has_bug=False):
        """Generate a code snippet based on language, technology, and difficulty."""
        # Create a structured code snippet template
        template = f"# {difficulty.value.capitalize()} level snippet\n"
        
        if language.lower() == "python":
            template += "# Python snippet for {}\n".format(technology)
            if has_bug:
                template += "# Contains a bug: {}\n".format("ZeroDivisionError" if has_bug is True else has_bug)
                # Simple Python code with proper indentation
                template += "def process_data(data):\n"
                template += "    result = []\n"
                template += "    for item in data:\n"
                template += "        # Bug here\n"
                template += "        value = 100 / (item - item)  # ZeroDivisionError\n"
                template += "        result.append(value)\n"
                template += "    return result\n"
            else:
                template += "# Implementation for {}\n".format(concept if concept else "data processing")
                # Simple Python code with proper indentation
                template += "def process_data(data):\n"
                template += "    result = []\n"
                template += "    for item in data:\n"
                template += "        value = item * 2\n"
                template += "        result.append(value)\n"
                template += "    return result\n"
        elif language.lower() == "javascript" or language.lower() == "typescript":
            ext = "js" if language.lower() == "javascript" else "ts"
            template += f"// {ext} snippet for {technology}\n"
            if has_bug:
                template += "// Contains a bug: {}\n".format("TypeError" if has_bug is True else has_bug)
                template += "function processData(data) {\n"
                template += "  const result = [];\n"
                template += "  for (const item of data) {\n"
                template += "    // Bug here\n"
                template += "    const obj = null;\n"
                template += "    console.log(obj.property);  // TypeError: Cannot read property\n"
                template += "    result.push(item * 2);\n"
                template += "  }\n"
                template += "  return result;\n"
                template += "}\n"
            else:
                template += "// Implementation for {}\n".format(concept if concept else "data processing")
                template += "function processData(data) {\n"
                template += "  const result = [];\n"
                template += "  for (const item of data) {\n"
                template += "    result.push(item * 2);\n"
                template += "  }\n"
                template += "  return result;\n"
                template += "}\n"
        elif language.lower() == "r":
            template += "# R snippet for {}\n".format(technology)
            if has_bug:
                template += "# Contains a bug: {}\n".format("TypeError" if has_bug is True else has_bug)
                template += "process_data <- function(data) {\n"
                template += "  result <- c()\n"
                template += "  for (item in data) {\n"
                template += "    # Bug here\n"
                template += "    value <- 100 / (item - item)  # Division by zero\n"
                template += "    result <- c(result, value)\n"
                template += "  }\n"
                template += "  return(result)\n"
                template += "}\n"
            else:
                template += "# Implementation for {}\n".format(concept if concept else "data processing")
                template += "process_data <- function(data) {\n"
                template += "  result <- c()\n"
                template += "  for (item in data) {\n"
                template += "    value <- item * 2\n"
                template += "    result <- c(result, value)\n"
                template += "  }\n"
                template += "  return(result)\n"
                template += "}\n"
        else:
            # Default template for other languages
            template += f"// Code snippet for {technology} in {language}\n"
            template += "// Simple implementation\n"
        
        return template

    def _ensure_examples_are_objects(self, examples: List[Any]) -> List[Example]:
        """Ensure all examples are Example objects."""
        if not examples:
            return []
        if isinstance(examples[0], Example):
            return examples
        elif isinstance(examples[0], dict):
            return [Example.from_dict(ex) if not isinstance(ex, Example) else ex for ex in examples]
        return examples
