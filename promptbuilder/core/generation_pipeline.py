"""
Generation Pipeline Module for PromptBuilder.

This module provides a pipeline orchestrator for the generation process,
coordinating the various components involved in generating and
validating examples. It handles the complexity of working with different
query types, difficulty levels, and frameworks while providing comprehensive
instrumentation and statistics.

Usage:
    from promptbuilder.core.generation_pipeline import GenerationPipeline
    pipeline = GenerationPipeline(config)
    examples = pipeline.generate_examples()

Author: Made in Jurgistan
Version: 3.0.0
License: MIT
"""

import logging
import random
import time
import traceback
import concurrent.futures
import os
from typing import Dict, List, Any, Optional, Set, Tuple
import uuid
from dataclasses import asdict

from promptbuilder.core.query_components import (
    QueryComponents, 
    QueryInfo, 
    Example,
    QueryType, 
    DifficultyLevel
)
from promptbuilder.core.framework_registry import FrameworkRegistry
from promptbuilder.core.template_manager import TemplateManager
from promptbuilder.core.component_generator import ComponentGenerator
from promptbuilder.core.response_generator import ResponseGenerator
from promptbuilder.config import Config


class GenerationPipeline:
    """Orchestrates the generation process by coordinating various components.
    
    This class brings together all the components involved in generating examples,
    including domain selection, component generation, query formatting, and response
    generation. It provides a clean interface for generating examples and handles
    the complexity of coordinating the various components.
    
    Attributes:
        config: Generator configuration
        logger: Logger instance
        framework_registry: Registry of reasoning frameworks
        template_manager: Manager for query templates
        component_generator: Generator for query components
        response_generator: Generator for responses
        statistics: Runtime statistics for instrumentation
    """
    
    def __init__(
        self,
        framework_registry: Any,
        component_generator: Any,
        template_manager: Any,
        response_generator: Any = None,
        config_name: str = "unnamed"
    ):
        """Initialize the generation pipeline.
        
        Args:
            framework_registry: Registry for reasoning frameworks
            component_generator: Generator for query components
            template_manager: Manager for templates
            response_generator: Generator for responses
            config_name: Name of the configuration
        """
        self.logger = logging.getLogger(__name__)
        self.framework_registry = framework_registry
        self.component_generator = component_generator
        self.template_manager = template_manager
        self.response_generator = response_generator
        self.config_name = config_name
        
        # Initialize statistics
        self.stats = {
            "complexity": {
                "min_query_length": None,
                "max_query_length": None,
                "avg_query_length": 0,
                "total_query_length": 0
            },
            "domains": {},
            "examples_count": 0,
            "generation_time": 0,
            "examples_per_second": 0
        }
        
        self.logger.info("GenerationPipeline initialized with configuration: %s", config_name)
    
    def _reset_statistics(self) -> Dict[str, Any]:
        """Reset statistics to default values.
        
        Returns:
            Dict[str, Any]: Default statistics
        """
        return {
            "complexity": {
                "min_query_length": None,
                "max_query_length": None,
                "avg_query_length": 0,
                "total_query_length": 0
            },
            "domains": {},
            "by_query_type": {},
            "by_difficulty": {},
            "examples_count": 0,
            "examples_generated": 0,
            "generation_time": 0,
            "examples_per_second": 0
        }
    
    def generate_examples(
        self,
        count: int = 1,
        domains: Optional[List[str]] = None,
        difficulties: Optional[List[DifficultyLevel]] = None,
        query_types: Optional[List[QueryType]] = None,
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
        start_time = time.time()
        examples = []
        self.stats = self._reset_statistics()
        self.stats['errors'] = {}
        self.config = getattr(self, 'config', Config())

        # Set default query types if not provided
        if not query_types:
            query_types = [
                QueryType.CONCEPT_EXPLANATION,
                QueryType.DEBUGGING,
                QueryType.IMPLEMENTATION_REQUESTS,
                QueryType.OPTIMIZATION
            ]
        # Set default difficulty levels if not provided
        if not difficulties:
            difficulties = [
                DifficultyLevel.BASIC,
                DifficultyLevel.INTERMEDIATE,
                DifficultyLevel.ADVANCED
            ]
        self.logger.info("Starting training example generation for coding assistant LLM")
        total_examples = count * len(query_types) * len(difficulties)
        example_count = 0
        for qt_idx, query_type in enumerate(query_types):
            for diff_idx, difficulty in enumerate(difficulties):
                self._update_stats_structure(query_type, difficulty)
                framework_type = self.framework_registry.select_framework(query_type)
                if parallel:
                    batch = self._generate_examples_in_parallel(query_type, difficulty, framework_type)
                else:
                    batch = self._generate_examples_sequentially(query_type, difficulty, framework_type)
                examples.extend([asdict(ex) if hasattr(ex, 'to_dict') else ex for ex in batch])
                self._update_batch_stats(query_type, difficulty, batch, time.time() - start_time)
        self._finalize_statistics(examples, start_time)
        self.logger.info("Generation complete: %d examples in %.2fs", len(examples), time.time() - start_time)
        return self._post_process_examples(examples)
    
    def _update_stats_structure(self, query_type: QueryType, difficulty: DifficultyLevel) -> None:
        """Initialize statistics structure for a query type and difficulty combination.
        
        Args:
            query_type: Type of query being generated
            difficulty: Difficulty level being generated
        """
        for key, val in [("by_query_type", query_type.value), ("by_difficulty", difficulty.value)]:
            if key not in self.stats:
                self.stats[key] = {}
            if val not in self.stats[key]:
                self.stats[key][val] = {"count": 0, "generation_time": 0}

    def _update_batch_stats(self, qt: QueryType, diff: DifficultyLevel, batch: List[Example], time_taken: float) -> None:
        """Update statistics after a batch generation.
        
        Args:
            qt: Query type of the batch
            diff: Difficulty level of the batch
            batch: List of examples in the batch
            time_taken: Time taken to generate the batch
        """
        count = len(batch)
        self.stats["by_query_type"][qt.value]["count"] += count
        self.stats["by_query_type"][qt.value]["generation_time"] += time_taken
        self.stats["by_difficulty"][diff.value]["count"] += count
        self.stats["by_difficulty"][diff.value]["generation_time"] += time_taken
        self.logger.info("Generated %d examples in %.2fs", count, time_taken)

    def _finalize_statistics(self, examples, start_time):
        """Finalize statistics after generation."""
        generation_time = time.time() - start_time
        self.stats["generation_time"] = generation_time
        self.stats["examples_count"] = len(examples)
        if generation_time > 0:
            self.stats["examples_per_second"] = len(examples) / generation_time
        if len(examples) > 0:
            total_query_length = sum(len(example.get("query", "")) for example in examples)
            self.stats["complexity"]["avg_query_length"] = total_query_length / len(examples)
        self.logger.info("Generation complete: %d examples in %.2fs", len(examples), self.stats["generation_time"])

    def _generate_examples_sequentially(self, query_type: QueryType, difficulty: DifficultyLevel, framework_type: Any) -> List[Example]:
        """Generate examples sequentially for a query type and difficulty level.
        
        Args:
            query_type: Type of query to generate
            difficulty: Difficulty level
            framework_type: Reasoning framework type
            
        Returns:
            List[Example]: Generated examples
        """
        batch = []
        for _ in range(getattr(self.config, 'num_examples_per_category', 1)):
            try:
                example = self._generate_single_example(query_type, difficulty)
                if example:
                    batch.append(example)
                    self._update_example_stats(example)
            except Exception as e:
                self._log_error("sequential generation", query_type, difficulty, e)
        return batch

    def _generate_examples_in_parallel(self, query_type: QueryType, difficulty: DifficultyLevel, framework_type: Any) -> List[Example]:
        """Generate examples in parallel for a query type and difficulty level.
        
        Args:
            query_type: Type of query to generate
            difficulty: Difficulty level
            framework_type: Reasoning framework type
            
        Returns:
            List[Example]: Generated examples
        """
        batch = []
        max_workers = min(getattr(self.config, 'num_examples_per_category', 1), os.cpu_count() or 4)
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(self._generate_single_example, query_type, difficulty)
                for _ in range(getattr(self.config, 'num_examples_per_category', 1))
            ]
            for future in concurrent.futures.as_completed(futures):
                try:
                    example = future.result()
                    if example:
                        batch.append(example)
                        self._update_example_stats(example)
                except Exception as e:
                    self._log_error("parallel generation", query_type, difficulty, e)
        return batch

    def _update_example_stats(self, example):
        """Update statistics based on an example."""
        if example is None:
            return
        domain = "unknown"
        if isinstance(example, dict) and "metadata" in example and "components" in example["metadata"]:
            components = example["metadata"]["components"]
            if "domain" in components:
                domain = components["domain"]
        query_length = len(example.get("query", ""))
        self.stats["complexity"]["min_query_length"] = min(
            self.stats["complexity"].get("min_query_length") or query_length,
            query_length
        )
        self.stats["complexity"]["max_query_length"] = max(
            self.stats["complexity"].get("max_query_length") or 0,
            query_length
        )
        self.stats["complexity"]["total_query_length"] += query_length
        if domain not in self.stats["domains"]:
            self.stats["domains"][domain] = 0
        self.stats["domains"][domain] += 1

    def _log_error(self, context: str, qt: QueryType, diff: DifficultyLevel, e: Exception) -> None:
        """Log an error during example generation.
        
        Args:
            context: Context in which the error occurred
            qt: Query type being generated
            diff: Difficulty level being generated
            e: The exception that occurred
        """
        error_type = type(e).__name__
        if "errors" not in self.stats:
            self.stats["errors"] = {}
        if error_type not in self.stats["errors"]:
            self.stats["errors"][error_type] = 0
        self.stats["errors"][error_type] += 1
        self.logger.warning("Error in %s for %s at %s: %s", context, qt.value, diff.value, e, exc_info=True)

    def _generate_single_example(self, query_type: QueryType, difficulty: DifficultyLevel, domain: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Generate a single example with all necessary components.
        
        Args:
            query_type: Type of query to generate
            difficulty: Difficulty level of the query
            domain: Domain for the query (optional)
            
        Returns:
            Dict[str, Any]: Generated example or None if generation failed
        """
        try:
            framework = self.framework_registry.select_framework(query_type)
            self.logger.debug(
                "Selected framework %s for query type %s",
                framework.value, query_type.value
            )
            components = self.component_generator.generate_components(query_type, difficulty, domain)
            if not components:
                return None
            query_info = QueryInfo(
                query_type=query_type,
                difficulty=difficulty,
                components=components
            )
            query_info.framework = framework
            # Generate internal reasoning and external response
            internal_reasoning = self.response_generator.generate_internal_reasoning(query_info) if self.response_generator else ""
            external_response = self.response_generator.generate_external_response(query_info, internal_reasoning) if self.response_generator else ""
            # Construct the example
            example = {
                "id": str(uuid.uuid4()),
                "query": getattr(query_info, 'query', ''),
                "internal_reasoning": internal_reasoning,
                "external_response": external_response,
                "metadata": {
                    "queryType": query_type.value,
                    "difficulty": difficulty.value,
                    "components": asdict(components),
                    "framework": framework.value
                }
            }
            return example
        except Exception as e:
            self._log_error("example generation", query_type, difficulty, e)
            return None

    def _select_domain(self) -> str:
        """Select a domain based on configuration.
        
        Returns:
            str: Selected domain
        """
        available_domains = list(getattr(self.config, 'technology_mapping', {}).keys())
        selected = [d for d in available_domains if d in getattr(self.config, 'selected_domains', [])]
        if not selected:
            selected = available_domains
        return random.choice(selected) if selected else "unknown"

    def _generate_dialogue_turns(self, query_info: QueryInfo) -> List[Dict[str, str]]:
        """Generate dialogue turns for multi-turn conversations.
        
        Args:
            query_info: Query information
            
        Returns:
            List[Dict[str, str]]: List of dialogue turns
        """
        qt = query_info.query_type
        comp = query_info.components
        turns = []
        
        # Generate dialogue turns based on query type
        if qt == QueryType.CONCEPT_EXPLANATION:
            concept = getattr(comp, 'concept', 'concept')
            turns.append({
                "assistant": f"Can you clarify which aspect of {concept} you're interested in?",
                "user": "I'd like to know about its practical applications."
            })
        elif qt == QueryType.DEBUGGING:
            error_message = getattr(comp, 'error_message', 'error')
            error_pattern = getattr(comp, 'error_occurrence_pattern', 'randomly')
            turns.append({
                "assistant": f"Does the error '{error_message}' occur consistently?",
                "user": f"Yes, {error_pattern}."
            })
        elif qt == QueryType.PROBLEM_SOLVING:
            problem = getattr(comp, 'problem_area', 'issue')
            turns.append({
                "assistant": f"Could you provide more details about the {problem} you're facing?",
                "user": "We need a solution that works with our existing architecture."
            })
        elif qt == QueryType.OPTIMIZATION:
            target = getattr(comp, 'optimization_target', 'performance')
            turns.append({
                "assistant": f"What specific metrics are you using to measure {target}?",
                "user": "We're primarily concerned with response time under load."
            })
        elif qt == QueryType.STRATEGIC_DECISIONS:
            option_a = getattr(comp, 'strategic_option_a', 'option A')
            option_b = getattr(comp, 'strategic_option_b', 'option B')
            turns.append({
                "assistant": f"What are your main criteria for choosing between {option_a} and {option_b}?",
                "user": "Long-term maintainability is our primary concern."
            })
        # elif qt == QueryType.IMPLEMENTATION:
        #     pass
        # elif qt == QueryType.CODE_REVIEW:
        #     pass
        # elif qt == QueryType.ARCHITECTURE_DESIGN:
        #     pass
        # elif qt == QueryType.TESTING:
        #     pass
        # elif qt == QueryType.REFACTORING:
        #     pass
        else:
            pass
        return turns
    
    def _post_process_examples(self, examples: List[Any]) -> List[Any]:
        """Apply post-processing to generated examples.
        
        Args:
            examples: List of examples to post-process
            
        Returns:
            List[Example]: Post-processed examples
        """
        seen = set()
        unique = []
        for ex in examples:
            key = (ex.get("query", None), ex.get("external_response", None))
            if key not in seen:
                seen.add(key)
                unique.append(ex)
        self.logger.info("Post-processing: reduced from %d to %d examples (removed %d duplicates)",
                        len(examples), len(unique), len(examples) - len(unique))
        min_query_length = getattr(self.config, 'minimum_query_length', 0)
        if min_query_length > 0:
            unique = [ex for ex in unique if len(ex.get("query", "")) >= min_query_length]
            self.logger.info("Post-processing: filtered to %d examples after minimum query length check",
                            len(unique))
        min_response_length = getattr(self.config, 'minimum_response_length', 0)
        if min_response_length > 0:
            unique = [ex for ex in unique if len(ex.get("external_response", "")) >= min_response_length]
            self.logger.info("Post-processing: filtered to %d examples after minimum response length check",
                            len(unique))
        return unique

    def get_statistics(self) -> Dict[str, Any]:
        """Get generation statistics.
        
        Returns:
            Dict[str, Any]: Statistics dictionary containing detailed information about the
                           generation process, including counts by query type, difficulty level,
                           domain, language, error rates, average lengths, and other metrics.
                           This data can be used for monitoring, reporting, and analysis of 
                           the generation process.
        """
        return self.stats
