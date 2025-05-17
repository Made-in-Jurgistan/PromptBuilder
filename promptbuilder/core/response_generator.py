"""
Response Generator Module for PromptBuilder.

This module handles the generation of responses to queries, including internal
reasoning and external responses. It encapsulates the response generation logic
to reduce complexity in the main generator.

Usage:
    from promptbuilder.core.response_generator import ResponseGenerator
    response_generator = ResponseGenerator(framework_registry)
    internal_reasoning = response_generator.generate_internal_reasoning(query_info)
    external_response = response_generator.generate_external_response(query_info, internal_reasoning)

Author: Made in Jurgistan
Version: 3.0.0
License: MIT
"""

import logging
from typing import Dict, List, Any, Optional

from promptbuilder.core.query_components import QueryInfo, FrameworkType, QueryType
from promptbuilder.core.framework_registry import FrameworkRegistry


class ResponseGenerator:
    """Generates responses for queries including reasoning and explanations.
    
    This class encapsulates the logic for generating internal reasoning and external
    responses for different query types and difficulty levels. It uses different
    reasoning frameworks based on the query type to generate appropriate responses.
    
    Attributes:
        framework_registry: Registry of reasoning frameworks
        logger: Logger instance
    """
    
    def __init__(self, framework_registry: FrameworkRegistry):
        """Initialize the response generator.
        
        Args:
            framework_registry: Registry of reasoning frameworks
        """
        self.framework_registry = framework_registry
        self.logger = logging.getLogger(__name__)
        self.logger.info("ResponseGenerator initialized")
    
    def generate_internal_reasoning(self, query_info: QueryInfo) -> str:
        """Generate internal reasoning for a query.
        
        This method generates detailed internal reasoning for a query based on the
        appropriate reasoning framework for the query type. The internal reasoning
        is structured to guide the generation of the external response.
        
        Args:
            query_info: Query information to generate reasoning for
            
        Returns:
            str: Generated internal reasoning
        """
        framework = query_info.framework
        self.logger.debug("Generating reasoning for %s using %s", 
                         query_info.query_type.value, framework.value)
        
        try:
            # Map framework types to their respective reasoning generation methods
            methods = {
                FrameworkType.DEVELOPER_CLARIFICATION: self._generate_clarification_reasoning,
                FrameworkType.CHAIN_OF_THOUGHT: self._generate_cot_reasoning,
                FrameworkType.HYBRID_DECISION_MAKING: self._generate_decision_making_reasoning
            }
            
            # Generate reasoning using the appropriate method or use a default
            if framework in methods:
                return methods[framework](query_info)
            else:
                return f"Default reasoning for {query_info.query_type.value} query at {query_info.difficulty.value} difficulty level."
                
        except Exception as e:
            self.logger.error("Reasoning generation failed: %s", e, exc_info=True)
            return f"Error generating reasoning: {e}"
    
    def generate_external_response(self, query_info: QueryInfo, internal_reasoning: str) -> str:
        """Generate external response based on internal reasoning.
        
        This method generates a user-facing response for a query using the internal
        reasoning as a guide. The response is tailored to the query type and difficulty
        level.
        
        Args:
            query_info: Query information
            internal_reasoning: Internal reasoning to base response on
            
        Returns:
            str: Generated external response
        """
        qt = query_info.query_type
        self.logger.debug("Generating response for %s", qt.value)
        
        try:
            # Map query types to their respective response generation methods
            methods = {
                QueryType.CONCEPT_EXPLANATION: self._generate_concept_explanation_response,
                QueryType.DEBUGGING: self._generate_debugging_response,
                QueryType.PROBLEM_SOLVING: self._generate_problem_solving_response,
                QueryType.OPTIMIZATION: self._generate_optimization_response,
                QueryType.IMPLEMENTATION_REQUESTS: self._generate_implementation_response,
                QueryType.STRATEGIC_DECISIONS: self._generate_strategic_decision_response,
                QueryType.VAGUE_QUERIES: self._generate_vague_query_response,
                QueryType.BEST_PRACTICES: self._generate_best_practices_response,
                QueryType.CODE_REVIEW: self._generate_code_review_response,
                QueryType.RECOMMENDATION: self._generate_recommendation_response
            }
            
            # Generate response using the appropriate method or use a default
            if qt in methods:
                return methods[qt](query_info, internal_reasoning)
            else:
                return "Response to your query: Based on the information provided, here's what you need to know..."
                
        except Exception as e:
            self.logger.error("Response generation failed: %s", e, exc_info=True)
            return "I apologize, but I encountered an error while generating a response. Please try rephrasing your query."
    
    def generate_feedback_response(self, query_info: QueryInfo, external_response: str) -> str:
        """Generate a feedback response for dialogue scenarios.
        
        This method generates a follow-up response to ask for feedback on the
        external response. It's tailored to the query type to make the follow-up
        feel natural and specific.
        
        Args:
            query_info: Query information
            external_response: Initial external response
            
        Returns:
            str: Feedback response
        """
        qt = query_info.query_type
        comp = query_info.components
        
        # Generate feedback response based on query type
        if qt == QueryType.CONCEPT_EXPLANATION:
            concept = getattr(comp, 'concept', 'concept')
            return f"Does this explanation of {concept} meet your needs, or should I elaborate on any specific aspect?"
        
        elif qt == QueryType.DEBUGGING:
            error = getattr(comp, 'error_message', 'error')
            return f"Did the solution for '{error}' work for you, or are you encountering other issues?"
        
        elif qt == QueryType.PROBLEM_SOLVING:
            problem = getattr(comp, 'problem_area', 'problem')
            return f"Does this approach to the {problem} address your needs, or would you like me to explore alternative solutions?"
        
        elif qt == QueryType.OPTIMIZATION:
            target = getattr(comp, 'optimization_target', 'optimization target')
            return f"Would these {target} optimizations work in your environment, or do you need modifications for your specific context?"
        
        elif qt == QueryType.IMPLEMENTATION_REQUESTS:
            task = getattr(comp, 'task_description', 'implementation')
            return f"Does this implementation of {task} meet your requirements, or do you need adjustments?"
        
        elif qt == QueryType.STRATEGIC_DECISIONS:
            return "Does my analysis of these options align with your priorities, or are there other factors I should consider?"
        
        elif qt == QueryType.CODE_REVIEW:
            return "Are there other aspects of the code you'd like me to review or explain in more detail?"
        
        else:
            return "Is this solution helpful for your needs, or would you like me to provide additional information?"
    
    def _generate_clarification_reasoning(self, query_info: QueryInfo) -> str:
        """Generate internal reasoning using the Developer Clarification framework.
        
        Args:
            query_info: Query information
            
        Returns:
            str: Internal reasoning
        """
        comp = query_info.components
        concept = getattr(comp, 'concept', 'unspecified concept')
        lang = getattr(comp, 'language', 'programming')
        rel_concept = getattr(comp, 'related_concept', 'related areas')
        concept_def = getattr(comp, 'concept_definition', 'requires explanation')
        
        return (
            f"# Developer Clarification Model\n\n"
            f"## Definition Component\n"
            f"{concept} is a core concept in {lang} that {concept_def}.\n\n"
            f"## Relevance Justification\n"
            f"This concept is important due to its use in {rel_concept} and impact on code quality.\n\n"
            f"## Comprehensive Breakdown\n"
            f"Key aspects of {concept} include:\n"
            f"1. Its underlying principles\n"
            f"2. Implementation strategies\n"
            f"3. Common usage patterns\n"
            f"4. Integration with other {lang} features\n\n"
            f"## Implementation Demonstration\n"
            f"Example implementation considerations include proper syntax, error handling, and optimization techniques.\n\n"
            f"## Risk Identification & Mitigation\n"
            f"Common mistakes when using {concept} include misunderstanding its scope, improper implementation, and overlooking edge cases.\n\n"
            f"## Knowledge Expansion Pathways\n"
            f"To deepen understanding of {concept}, exploration of advanced use cases, performance implications, and integration patterns would be beneficial."
        )
    
    def _generate_cot_reasoning(self, query_info: QueryInfo) -> str:
        """Generate internal reasoning using the Chain of Thought framework.
        
        Args:
            query_info: Query information
            
        Returns:
            str: Internal reasoning
        """
        comp = query_info.components
        problem = getattr(comp, 'problem_area', 'unspecified problem')
        tech = getattr(comp, 'technology', 'technology')
        lang = getattr(comp, 'language', 'language')
        approach_a = getattr(comp, 'approach_a', 'solution approach')
        
        return (
            f"# Chain of Thought Framework\n\n"
            f"## Problem Definition & Boundary Setting\n"
            f"The issue involves a {problem} in {tech} using {lang}. This appears to be related to {getattr(comp, 'problem_description', 'functionality issues')}.\n\n"
            f"## Comprehensive Decomposition & Structure Analysis\n"
            f"Breaking this down into components:\n"
            f"1. The core {problem} mechanism\n"
            f"2. Interaction with {tech} components\n"
            f"3. {lang} implementation specifics\n"
            f"4. Environmental factors\n\n"
            f"## Systematic Logical Deduction Process\n"
            f"Step 1: Analyze the root cause of the {problem}\n"
            f"Step 2: Consider potential solutions including {approach_a}\n"
            f"Step 3: Evaluate tradeoffs between solutions\n"
            f"Step 4: Formulate implementation strategy\n\n"
            f"## Comprehensive Outcome Evaluation & Testing\n"
            f"The solution should address:\n"
            f"- Primary functionality requirements\n"
            f"- Edge cases and error conditions\n"
            f"- Performance considerations\n"
            f"- Compatibility with {tech} ecosystem\n\n"
            f"## Solution Presentation & Implementation Guidance\n"
            f"The recommended implementation approach is {approach_a}, with specific considerations for {lang} implementation patterns."
        )
    
    def _generate_decision_making_reasoning(self, query_info: QueryInfo) -> str:
        """Generate internal reasoning using the Hybrid Decision-Making framework.
        
        Args:
            query_info: Query information
            
        Returns:
            str: Internal reasoning
        """
        comp = query_info.components
        option_a = getattr(comp, 'strategic_option_a', 'Option A')
        option_b = getattr(comp, 'strategic_option_b', 'Option B')
        constraints = getattr(comp, 'constraints', 'project constraints')
        goal = getattr(comp, 'long_term_goal', 'long-term objectives')
        
        return (
            f"# Hybrid Decision-Making Framework\n\n"
            f"## Context Definition & Objective Setting\n"
            f"The decision involves choosing between {option_a} and {option_b} for a {getattr(comp, 'project_type', 'project')} with {constraints} and {goal} as primary objectives.\n\n"
            f"## Comprehensive Option Generation\n"
            f"Options:\n"
            f"A: {option_a}\n"
            f"B: {option_b}\n"
            f"C: Hybrid approach combining elements of both\n"
            f"D: Alternative solution path\n\n"
            f"## Multidimensional Analysis Framework\n"
            f"Comparing options across dimensions:\n"
            f"1. Alignment with {goal}\n"
            f"2. Feasibility within {constraints}\n"
            f"3. Technical implementation complexity\n"
            f"4. Maintenance and scalability\n"
            f"5. Risk profile\n\n"
            f"## Evidence-Based Selection Protocol\n"
            f"Based on the analysis, {option_a} appears to better align with {goal} while addressing {constraints}. Key factors in this assessment include implementation complexity, scalability, and long-term maintainability.\n\n"
            f"## Comprehensive Implementation Roadmap\n"
            f"Implementation of {option_a} would involve:\n"
            f"1. Initial architecture and design\n"
            f"2. Phased implementation approach\n"
            f"3. Integration testing strategy\n"
            f"4. Migration plan from existing systems\n"
            f"5. Monitoring and evaluation criteria"
        )
    
    def _generate_concept_explanation_response(self, query_info: QueryInfo, internal_reasoning: str) -> str:
        """Generate response for concept explanation queries.
        
        Args:
            query_info: Query information
            internal_reasoning: Internal reasoning
            
        Returns:
            str: External response
        """
        comp = query_info.components
        concept = getattr(comp, 'concept', 'concept')
        lang = getattr(comp, 'language', 'language')
        definition = getattr(comp, 'concept_definition', 'a key programming concept')
        example_code = getattr(comp, 'example_code', '')
        
        response = (
            f"# Understanding {concept.title()} in {lang.title()}\n\n"
            f"{concept} is {definition}. It's widely used in {lang} for building robust and maintainable code.\n\n"
            f"## Key Characteristics\n\n"
            f"- Core principle: Encapsulation of related functionality\n"
            f"- Implementation: Follows standard {lang} patterns\n"
            f"- Use cases: Particularly valuable for application architecture\n\n"
            f"## Practical Application\n\n"
            f"When implementing {concept} in your code, focus on clean interfaces and proper separation of concerns."
        )
        
        # Add example code if available
        if example_code:
            response += f"\n\n## Example Implementation\n\n```{lang.lower()}\n{example_code}\n```"
        
        # Add comparison to related concept if available
        related = getattr(comp, 'related_concept', None)
        if related:
            response += f"\n\n## Comparison with {related}\n\nWhile {concept} focuses on {definition}, {related} is different in that it emphasizes other aspects of programming design."
        
        return response
    
    def _generate_debugging_response(self, query_info: QueryInfo, internal_reasoning: str) -> str:
        """Generate response for debugging queries.
        
        Args:
            query_info: Query information
            internal_reasoning: Internal reasoning
            
        Returns:
            str: External response
        """
        comp = query_info.components
        error = getattr(comp, 'error_message', 'error')
        lang = getattr(comp, 'language', 'code')
        tech = getattr(comp, 'technology', 'technology')
        error_pattern = getattr(comp, 'error_occurrence_pattern', 'under certain conditions')
        debug_code = getattr(comp, 'debug_code_snippet', '')
        
        response = (
            f"# Fixing '{error}' in {lang.title()}\n\n"
            f"This error occurs {error_pattern} when using {tech}. The root cause is typically related to improper handling of resources or incorrect API usage.\n\n"
            f"## Diagnosis\n\n"
            f"The error '{error}' suggests that:"
        )
        
        # Add specific diagnosis points based on error type
        if "null" in error.lower() or "undefined" in error.lower():
            response += "\n- A variable or property is being accessed before it's initialized\n- An expected return value is missing\n- An API call is failing silently"
        elif "syntax" in error.lower():
            response += "\n- There's a syntax error in your code\n- The code structure doesn't follow {lang} conventions\n- A required delimiter or symbol is missing"
        elif "permission" in error.lower() or "access" in error.lower():
            response += "\n- The code lacks necessary permissions\n- Authentication is failing\n- Security constraints are preventing execution"
        else:
            response += "\n- The application state is inconsistent\n- Resource management is improper\n- Error handling is incomplete"
        
        response += (
            f"\n\n## Solution\n\n"
            f"To fix this issue:"
        )
        
        # Add specific solution steps
        response += (
            f"\n1. Check {debug_code or 'the problematic code'} for proper initialization and error handling"
            f"\n2. Ensure that all dependencies are correctly loaded before use"
            f"\n3. Implement proper error handling with try/catch blocks"
            f"\n4. Add logging to trace the execution flow"
        )
        
        # Add code example if available
        if debug_code:
            response += f"\n\n## Code Review\n\n```{lang.lower()}\n{debug_code}\n```\n\nThe issue is likely in how resources are being accessed or managed in this snippet."
        
        return response
    
    def _generate_problem_solving_response(self, query_info: QueryInfo, internal_reasoning: str) -> str:
        """Generate response for problem solving queries.
        
        Args:
            query_info: Query information
            internal_reasoning: Internal reasoning
            
        Returns:
            str: External response
        """
        comp = query_info.components
        problem = getattr(comp, 'problem_area', 'issue')
        lang = getattr(comp, 'language', 'programming language')
        tech = getattr(comp, 'technology', 'technology')
        approach = getattr(comp, 'approach_a', 'a systematic approach')
        code_snippet = getattr(comp, 'code_snippet', '')
        
        response = (
            f"# Solving {problem.title()} in {lang.title()} with {tech}\n\n"
            f"This {problem} can be effectively addressed using {approach}. Let's break down the solution into manageable steps.\n\n"
            f"## Approach\n\n"
            f"1. Analyze the specific requirements and constraints\n"
            f"2. Design a solution architecture that leverages {tech} capabilities\n"
            f"3. Implement core functionality with proper error handling\n"
            f"4. Test against edge cases and performance requirements\n"
            f"5. Refine based on testing results\n\n"
            f"## Implementation Strategy\n\n"
            f"The key to solving this {problem} is to focus on:"
        )
        
        # Add implementation details based on problem type
        if "performance" in problem.lower():
            response += "\n- Optimizing critical code paths\n- Implementing caching mechanisms\n- Reducing computational complexity\n- Minimizing resource utilization"
        elif "integration" in problem.lower():
            response += "\n- Establishing clear API contracts\n- Implementing robust error handling\n- Using appropriate design patterns for integration\n- Ensuring proper data transformation"
        elif "security" in problem.lower():
            response += "\n- Implementing proper authentication and authorization\n- Validating all inputs\n- Protecting sensitive data\n- Following security best practices"
        else:
            response += "\n- Breaking down complex logic into manageable components\n- Ensuring proper separation of concerns\n- Implementing comprehensive error handling\n- Designing for maintainability and extensibility"
        
        # Add code example if available
        if code_snippet:
            response += f"\n\n## Sample Implementation\n\n```{lang.lower()}\n{code_snippet}\n```"
        
        return response
    
    def _generate_optimization_response(self, query_info: QueryInfo, internal_reasoning: str) -> str:
        """Generate response for optimization queries.
        
        Args:
            query_info: Query information
            internal_reasoning: Internal reasoning
            
        Returns:
            str: External response
        """
        comp = query_info.components
        target = getattr(comp, 'optimization_target', 'performance')
        lang = getattr(comp, 'language', 'code')
        tech = getattr(comp, 'technology', 'technology')
        
        response = (
            f"# Optimizing {target.title()} in {lang.title()} with {tech}\n\n"
            f"Improving {target} in your {lang} application using {tech} requires a multi-faceted approach. Here are key strategies to consider:\n\n"
            f"## Analysis Phase\n\n"
            f"1. Profile your application to identify bottlenecks\n"
            f"2. Measure current {target} metrics to establish a baseline\n"
            f"3. Identify high-impact areas for optimization\n\n"
            f"## Optimization Techniques\n\n"
        )
        
        # Add specific optimization techniques based on target
        if "performance" in target.lower():
            response += (
                "### Algorithm Optimization\n"
                "- Replace O(n²) algorithms with O(n log n) alternatives\n"
                "- Minimize expensive operations in loops\n"
                "- Use memoization for repetitive calculations\n\n"
                "### Resource Management\n"
                "- Implement connection pooling\n"
                "- Use appropriate caching strategies\n"
                "- Optimize memory usage patterns\n\n"
                "### Code-Level Improvements\n"
                "- Eliminate redundant computations\n"
                "- Optimize data structures for access patterns\n"
                "- Leverage language-specific performance features"
            )
        elif "memory" in target.lower():
            response += (
                "### Memory Usage Patterns\n"
                "- Minimize object creation in critical paths\n"
                "- Implement object pooling for frequently used objects\n"
                "- Use appropriate data structures for memory efficiency\n\n"
                "### Resource Cleanup\n"
                "- Ensure proper resource disposal\n"
                "- Implement memory leak detection\n"
                "- Use weak references where appropriate\n\n"
                "### Data Processing\n"
                "- Process data in chunks\n"
                "- Implement streaming for large datasets\n"
                "- Use memory-mapped files for large data"
            )
        else:
            response += (
                "### Code Quality\n"
                "- Refactor complex methods into smaller, focused functions\n"
                "- Implement consistent error handling patterns\n"
                "- Follow established design patterns\n\n"
                "### System Architecture\n"
                "- Evaluate component boundaries\n"
                "- Consider microservices for better scaling\n"
                "- Implement appropriate caching strategies\n\n"
                "### Testing and Validation\n"
                "- Implement comprehensive test coverage\n"
                "- Use automated performance testing\n"
                "- Establish clear quality metrics"
            )
        
        response += (
            f"\n\n## Implementation Strategy\n\n"
            f"1. Start with the highest-impact optimizations\n"
            f"2. Measure improvement after each change\n"
            f"3. Test thoroughly to ensure optimization doesn't introduce bugs\n"
            f"4. Document optimization techniques for future reference"
        )
        
        return response
    
    def _generate_implementation_response(self, query_info: QueryInfo, internal_reasoning: str) -> str:
        """Generate response for implementation request queries.
        
        Args:
            query_info: Query information
            internal_reasoning: Internal reasoning
            
        Returns:
            str: External response
        """
        comp = query_info.components
        task = getattr(comp, 'task_description', 'feature')
        lang = getattr(comp, 'language', 'programming language')
        app_type = getattr(comp, 'application_type', 'application')
        tech = getattr(comp, 'technology', 'technology')
        code_snippet = getattr(comp, 'code_snippet', '')
        
        response = (
            f"# Implementing {task.title()} in {lang.title()}\n\n"
            f"Here's a comprehensive guide to implementing {task} for your {app_type} using {lang}"
        )
        
        if tech:
            response += f" and {tech}"
        
        response += (
            f".\n\n"
            f"## Design Considerations\n\n"
            f"Before diving into implementation, consider these key design aspects:\n\n"
            f"1. Interface design: How users or other components will interact with this feature\n"
            f"2. Data requirements: What information needs to be stored or processed\n"
            f"3. Error handling: How to manage edge cases and unexpected inputs\n"
            f"4. Performance implications: Potential impact on system performance\n\n"
            f"## Implementation Steps\n\n"
            f"1. Define clear requirements and acceptance criteria\n"
            f"2. Design the component architecture\n"
            f"3. Implement core functionality\n"
            f"4. Add error handling and validation\n"
            f"5. Write tests to verify behavior\n"
            f"6. Optimize for performance and maintainability\n\n"
        )
        
        # Add implementation specifics based on language
        if lang.lower() in ["javascript", "typescript", "js", "ts"]:
            response += (
                "## JavaScript Implementation Pattern\n\n"
                "For modern JavaScript applications, consider using a modular approach:\n\n"
                "- Create dedicated modules for core functionality\n"
                "- Use modern ES6+ features for cleaner code\n"
                "- Implement proper error boundaries\n"
                "- Consider async/await for asynchronous operations"
            )
        elif lang.lower() in ["python", "py"]:
            response += (
                "## Python Implementation Pattern\n\n"
                "For Python applications, focus on readability and maintainability:\n\n"
                "- Follow PEP 8 style guidelines\n"
                "- Use appropriate design patterns\n"
                "- Leverage Python's built-in features\n"
                "- Implement proper error handling with try/except"
            )
        elif lang.lower() in ["java"]:
            response += (
                "## Java Implementation Pattern\n\n"
                "For Java applications, focus on object-oriented design:\n\n"
                "- Create appropriate class hierarchies\n"
                "- Use interfaces for abstraction\n"
                "- Implement proper exception handling\n"
                "- Consider design patterns like Factory or Strategy"
            )
        else:
            response += (
                f"## {lang.title()} Implementation Pattern\n\n"
                f"When implementing in {lang}, focus on these best practices:\n\n"
                f"- Follow established {lang} conventions\n"
                f"- Use appropriate design patterns\n"
                f"- Implement comprehensive error handling\n"
                f"- Write maintainable, self-documenting code"
            )
        
        # Add code example if available
        if code_snippet:
            response += f"\n\n## Sample Implementation\n\n```{lang.lower()}\n{code_snippet}\n```"
        
        response += (
            f"\n\n## Testing Strategy\n\n"
            f"To ensure your implementation works correctly:\n\n"
            f"1. Write unit tests for core functionality\n"
            f"2. Test edge cases and error conditions\n"
            f"3. Consider performance testing for critical paths\n"
            f"4. Implement integration tests if applicable"
        )
        
        return response
    
    def _generate_strategic_decision_response(self, query_info: QueryInfo, internal_reasoning: str) -> str:
        """Generate response for strategic decision queries.
        
        Args:
            query_info: Query information
            internal_reasoning: Internal reasoning
            
        Returns:
            str: External response
        """
        comp = query_info.components
        option_a = getattr(comp, 'strategic_option_a', 'Option A')
        option_b = getattr(comp, 'strategic_option_b', 'Option B')
        lang = getattr(comp, 'language', 'programming language')
        goal = getattr(comp, 'long_term_goal', 'long-term objectives')
        constraints = getattr(comp, 'constraints', 'constraints')
        
        response = (
            f"# Strategic Decision: {option_a} vs. {option_b} for {lang.title()}\n\n"
            f"Choosing between {option_a} and {option_b} for your {lang} project requires careful consideration of your requirements, constraints, and long-term goals.\n\n"
            f"## Comprehensive Analysis\n\n"
        )
        
        # Compare options across multiple dimensions
        response += (
            "### Performance\n"
            f"- {option_a}: Typically offers {random_performance(option_a)}\n"
            f"- {option_b}: Generally provides {random_performance(option_b)}\n\n"
            
            "### Maintenance\n"
            f"- {option_a}: Requires {random_maintenance(option_a)}\n"
            f"- {option_b}: Needs {random_maintenance(option_b)}\n\n"
            
            "### Ecosystem\n"
            f"- {option_a}: Has {random_ecosystem(option_a)}\n"
            f"- {option_b}: Offers {random_ecosystem(option_b)}\n\n"
            
            "### Learning Curve\n"
            f"- {option_a}: Generally {random_learning(option_a)}\n"
            f"- {option_b}: Typically {random_learning(option_b)}\n\n"
            
            "### Future Outlook\n"
            f"- {option_a}: {random_future(option_a)}\n"
            f"- {option_b}: {random_future(option_b)}\n\n"
        )
        
        # Generate recommendation
        response += (
            f"## Recommendation\n\n"
            f"Based on your stated goal of {goal} and working within {constraints}, "
        )
        
        # Randomly choose recommendation
        import random
        if random.choice([True, False]):
            response += (
                f"**{option_a}** would be the better choice because:\n\n"
                f"1. It better aligns with {goal}\n"
                f"2. It works well within your {constraints}\n"
                f"3. Its strengths in {random_strength(option_a)} are particularly relevant to your context\n\n"
            )
        else:
            response += (
                f"**{option_b}** would be the better choice because:\n\n"
                f"1. It provides superior support for {goal}\n"
                f"2. It addresses your {constraints} more effectively\n"
                f"3. Its approach to {random_strength(option_b)} aligns with your needs\n\n"
            )
        
        response += (
            f"## Implementation Considerations\n\n"
            f"Whichever option you choose, consider these implementation factors:\n\n"
            f"1. Developer familiarity and training requirements\n"
            f"2. Integration with existing systems\n"
            f"3. Support and community resources\n"
            f"4. Total cost of ownership over time"
        )
        
        return response
    
    def _generate_vague_query_response(self, query_info: QueryInfo, internal_reasoning: str) -> str:
        """Generate response for vague queries.
        
        Args:
            query_info: Query information
            internal_reasoning: Internal reasoning
            
        Returns:
            str: External response
        """
        comp = query_info.components
        vague_term = getattr(comp, 'vague_term', 'best')
        tech = getattr(comp, 'technology', 'technology')
        lang = getattr(comp, 'language', 'programming language')
        
        response = (
            f"# Understanding '{vague_term}' Approaches for {tech} with {lang.title()}\n\n"
            f"I notice you're asking about the '{vague_term}' way to use {tech} with {lang}. To provide the most helpful response, let me clarify what '{vague_term}' might mean in this context.\n\n"
            f"## Interpreting '{vague_term.title()}'\n\n"
            f"'{vague_term}' could be interpreted in several ways:\n\n"
        )
        
        # Generate interpretations based on vague term
        if vague_term.lower() in ['best', 'good', 'better', 'ideal']:
            response += (
                "1. **Most efficient**: Approaches that optimize for performance\n"
                "2. **Most maintainable**: Code patterns that are easier to understand and modify\n"
                "3. **Most scalable**: Solutions that work well as your application grows\n"
                "4. **Industry standard**: Widely accepted practices in the community\n"
            )
        elif vague_term.lower() in ['easy', 'simple', 'quick', 'fast']:
            response += (
                "1. **Least complex**: Approaches with minimal moving parts\n"
                "2. **Beginner-friendly**: Patterns accessible to those new to {tech}\n"
                "3. **Quick to implement**: Solutions that can be built rapidly\n"
                "4. **Straightforward to maintain**: Code that's easy to understand later\n"
            )
        elif vague_term.lower() in ['modern', 'current', 'latest']:
            response += (
                "1. **Recent patterns**: Approaches that leverage latest language features\n"
                "2. **Contemporary tools**: Integration with modern tooling ecosystems\n"
                "3. **Community trends**: What leading developers are currently adopting\n"
                "4. **Forward-looking**: Practices that anticipate future developments\n"
            )
        else:
            response += (
                f"1. **Optimized for {vague_term}**: Approaches that prioritize this quality\n"
                f"2. **Industry recognized {vague_term}**: Patterns acknowledged for this attribute\n"
                f"3. **{vague_term.title()} by measurable metrics**: Solutions with quantifiable outcomes\n"
                f"4. **Subjectively {vague_term}**: Based on developer experience and preference\n"
            )
        
        response += (
            f"\n## Recommended Approach\n\n"
            f"Assuming you're looking for widely accepted best practices that balance performance, maintainability, and industry standards, here's my recommendation for using {tech} with {lang}:\n\n"
            f"1. Start with a well-structured foundation\n"
            f"2. Follow the established conventions of the {tech} ecosystem\n"
            f"3. Implement proper error handling and validation\n"
            f"4. Write comprehensive tests for your implementation\n"
            f"5. Document your code and design decisions\n\n"
            f"## Next Steps\n\n"
            f"To provide more specific guidance, it would help if you could clarify:\n\n"
            f"- What specific aspect of {tech} you're working with\n"
            f"- Your primary goals (performance, maintainability, etc.)\n"
            f"- Your team's experience level with {tech} and {lang}\n"
            f"- Any specific challenges you're facing"
        )
        
        return response
    
    def _generate_best_practices_response(self, query_info: QueryInfo, internal_reasoning: str) -> str:
        """Generate response for best practices queries.
        
        Args:
            query_info: Query information
            internal_reasoning: Internal reasoning
            
        Returns:
            str: External response
        """
        comp = query_info.components
        area = getattr(comp, 'best_practice_area', 'coding')
        lang = getattr(comp, 'language', 'programming language')
        tech = getattr(comp, 'technology', 'technology')
        
        response = (
            f"# Best Practices for {area.title()} in {lang.title()} with {tech}\n\n"
            f"Following established best practices for {area} when working with {tech} in {lang} will help you create more maintainable, reliable, and efficient code.\n\n"
            f"## Core Principles\n\n"
        )
        
        # Generate principles based on practice area
        if area.lower() in ['error handling', 'exception handling']:
            response += (
                "1. **Be specific**: Catch and handle specific exceptions rather than using general exception handlers\n"
                "2. **Fail fast**: Detect and report errors as early as possible\n"
                "3. **Provide context**: Include meaningful information in error messages\n"
                "4. **Preserve original errors**: Maintain the stack trace when re-throwing exceptions\n"
                "5. **Log appropriately**: Record errors with enough detail for troubleshooting\n\n"
                
                "## Implementation Guidelines\n\n"
                "### Error Prevention\n"
                "- Validate inputs at system boundaries\n"
                "- Use strong typing and type checking where possible\n"
                "- Implement precondition checks for methods\n\n"
                
                "### Exception Handling Patterns\n"
                "- Use try/catch blocks judiciously\n"
                "- Create custom exception types for domain-specific errors\n"
                "- Implement proper cleanup in finally blocks or equivalent\n\n"
                
                "### Error Reporting\n"
                "- Provide actionable error messages\n"
                "- Include error codes for systematic categorization\n"
                "- Consider different audience needs (users vs. developers)"
            )
        elif area.lower() in ['security', 'secure coding']:
            response += (
                "1. **Validate all inputs**: Never trust user-provided data\n"
                "2. **Implement proper authentication**: Use robust identity verification\n"
                "3. **Apply least privilege**: Restrict permissions to the minimum needed\n"
                "4. **Protect sensitive data**: Encrypt confidential information\n"
                "5. **Defense in depth**: Implement multiple security layers\n\n"
                
                "## Implementation Guidelines\n\n"
                "### Input Validation\n"
                "- Use parameterized queries to prevent injection attacks\n"
                "- Implement whitelist validation rather than blacklisting\n"
                "- Sanitize all user inputs before processing\n\n"
                
                "### Authentication & Authorization\n"
                "- Use established authentication frameworks\n"
                "- Implement proper session management\n"
                "- Apply role-based access control\n\n"
                
                "### Data Protection\n"
                "- Use established encryption standards\n"
                "- Properly manage encryption keys\n"
                "- Implement secure storage of sensitive data"
            )
        elif area.lower() in ['performance', 'optimization']:
            response += (
                "1. **Measure first**: Profile before optimizing\n"
                "2. **Focus on bottlenecks**: Target the most impactful areas\n"
                "3. **Consider algorithms**: Choose appropriate algorithmic complexity\n"
                "4. **Optimize data structures**: Select the right tool for each job\n"
                "5. **Test thoroughly**: Verify optimizations don't introduce bugs\n\n"
                
                "## Implementation Guidelines\n\n"
                "### Profiling and Measurement\n"
                "- Use dedicated profiling tools\n"
                "- Establish performance baselines\n"
                "- Measure real-world scenarios\n\n"
                
                "### Code Optimizations\n"
                "- Minimize expensive operations in loops\n"
                "- Implement caching for frequently accessed data\n"
                "- Use lazy loading for resources\n\n"
                
                "### System-Level Considerations\n"
                "- Optimize database queries\n"
                "- Implement appropriate caching strategies\n"
                "- Consider asynchronous processing for time-consuming operations"
            )
        else:
            response += (
                f"1. **Follow {lang} conventions**: Adhere to language-specific standards\n"
                f"2. **Keep it simple**: Prefer clarity over clever code\n"
                f"3. **Practice DRY (Don't Repeat Yourself)**: Eliminate duplication\n"
                f"4. **Write tests**: Ensure code behaves as expected\n"
                f"5. **Document effectively**: Help others understand your code\n\n"
                
                f"## Implementation Guidelines\n\n"
                f"### Code Organization\n"
                f"- Structure code logically in modules and components\n"
                f"- Maintain clear separation of concerns\n"
                f"- Use consistent naming conventions\n\n"
                
                f"### Quality Assurance\n"
                f"- Implement automated testing\n"
                f"- Use static analysis tools\n"
                f"- Conduct regular code reviews\n\n"
                
                f"### Maintenance Practices\n"
                f"- Document design decisions\n"
                f"- Keep dependencies updated\n"
                f"- Refactor code regularly to manage technical debt"
            )
        
        response += (
            f"\n\n## Tools and Resources\n\n"
            f"To implement these best practices in your {lang} projects with {tech}:\n\n"
            f"1. Use linting tools to enforce coding standards\n"
            f"2. Integrate automated testing in your development workflow\n"
            f"3. Leverage static analysis tools to identify potential issues\n"
            f"4. Consult community resources and documentation for {tech}-specific guidance"
        )
        
        return response
    
    def _generate_code_review_response(self, query_info: QueryInfo, internal_reasoning: str) -> str:
        """Generate response for code review queries.
        
        Args:
            query_info: Query information
            internal_reasoning: Internal reasoning
            
        Returns:
            str: External response
        """
        comp = query_info.components
        lang = getattr(comp, 'language', 'code')
        quality = getattr(comp, 'quality_attribute', 'quality')
        code_snippet = getattr(comp, 'code_snippet', 'No code provided')
        
        response = (
            f"# Code Review: {lang.title()} Implementation\n\n"
            f"Thanks for sharing your {lang} code for review. I'll analyze it with a focus on {quality}.\n\n"
            f"## Code Analysis\n\n"
            f"```{lang.lower()}\n{code_snippet}\n```\n\n"
            f"## Strengths\n\n"
        )
        
        # Generate generic strengths based on the language
        response += (
            "- The code structure is generally clear and logical\n"
            "- You've used meaningful variable and function names\n"
            "- The implementation approach is sound for the intended purpose\n\n"
        )
        
        response += f"## Areas for Improvement\n\n"
        
        # Generate improvement suggestions based on quality attribute
        if quality.lower() in ['performance', 'efficiency', 'speed']:
            response += (
                "### Performance Considerations\n\n"
                "1. **Optimize loops**: Consider restructuring the nested loops to reduce computational complexity\n"
                "2. **Memory usage**: Watch for unnecessary object creation within performance-critical sections\n"
                "3. **Caching**: Implement caching for frequently accessed values\n"
                "4. **Algorithm selection**: Consider whether alternative algorithms might offer better performance\n\n"
                
                "### Specific Suggestions\n\n"
                "- Replace the O(n²) operation with a more efficient approach\n"
                "- Move invariant calculations outside of loops\n"
                "- Consider using more efficient data structures for the access patterns in the code"
            )
        elif quality.lower() in ['security', 'secure']:
            response += (
                "### Security Considerations\n\n"
                "1. **Input validation**: Add validation for all external inputs\n"
                "2. **Sanitization**: Ensure proper sanitization of user-provided data\n"
                "3. **Authentication**: Strengthen the authentication mechanism\n"
                "4. **Data protection**: Review handling of sensitive information\n\n"
                
                "### Specific Suggestions\n\n"
                "- Implement parameterized queries instead of string concatenation\n"
                "- Add input validation before processing user data\n"
                "- Review exception handling to ensure it doesn't expose sensitive information"
            )
        elif quality.lower() in ['maintainability', 'readability', 'clean']:
            response += (
                "### Maintainability Considerations\n\n"
                "1. **Modularization**: Break down larger functions into smaller, focused ones\n"
                "2. **Documentation**: Add comments explaining complex logic or business rules\n"
                "3. **Consistency**: Standardize naming conventions and coding style\n"
                "4. **Test coverage**: Ensure adequate test coverage for critical functionality\n\n"
                
                "### Specific Suggestions\n\n"
                "- Extract the complex logic into separate, well-named functions\n"
                "- Add docstrings/comments explaining the purpose and constraints\n"
                "- Refactor duplicate code into reusable functions"
            )
        else:
            response += (
                "### General Improvements\n\n"
                "1. **Error handling**: Enhance error handling for better resilience\n"
                "2. **Code organization**: Improve structure for better readability\n"
                "3. **Documentation**: Add comments for complex sections\n"
                "4. **Testing**: Ensure adequate test coverage\n\n"
                
                "### Specific Suggestions\n\n"
                "- Add more specific exception handling\n"
                "- Break down larger methods into smaller, focused ones\n"
                "- Add documentation explaining the purpose and usage"
            )
        
        response += (
            f"\n\n## Recommended Refactoring\n\n"
            f"Here's how you might improve this code:\n\n"
            f"1. Add proper validation for inputs\n"
            f"2. Restructure for better separation of concerns\n"
            f"3. Implement comprehensive error handling\n"
            f"4. Add appropriate documentation\n\n"
            f"## Summary\n\n"
            f"Your {lang} code has a good foundation but could benefit from the improvements outlined above to enhance {quality}. The suggested changes will make the code more robust, maintainable, and aligned with best practices."
        )
        
        return response
    
    def _generate_recommendation_response(self, query_info: QueryInfo, internal_reasoning: str) -> str:
        """Generate response for recommendation queries.
        
        Args:
            query_info: Query information
            internal_reasoning: Internal reasoning
            
        Returns:
            str: External response
        """
        comp = query_info.components
        domain = getattr(comp, 'domain', 'technical')
        lang = getattr(comp, 'language', 'programming language')
        criteria = getattr(comp, 'recommendation_criteria', 'general requirements')
        
        response = (
            f"# Recommended {domain.title()} Technologies for {lang.title()}\n\n"
            f"Based on your interest in {domain} technologies for {lang} with a focus on {criteria}, here are my recommendations.\n\n"
            f"## Evaluation Criteria\n\n"
            f"I've evaluated options based on:\n\n"
            f"1. Compatibility with {lang}\n"
            f"2. Support for {criteria}\n"
            f"3. Community adoption and support\n"
            f"4. Maturity and stability\n"
            f"5. Future outlook and development\n\n"
            f"## Top Recommendations\n\n"
        )
        
        # Generate recommendations based on domain and language
        if domain.lower() in ['web', 'frontend', 'front-end', 'ui']:
            if lang.lower() in ['javascript', 'js', 'typescript', 'ts']:
                response += (
                    "### 1. React\n\n"
                    "**Strengths**:\n"
                    "- Component-based architecture promotes reusability\n"
                    "- Large ecosystem and community support\n"
                    "- Excellent performance with virtual DOM\n"
                    "- Strong TypeScript integration (if using TypeScript)\n\n"
                    
                    "**Considerations**:\n"
                    "- Requires additional libraries for state management and routing\n"
                    "- Steeper learning curve for beginners\n\n"
                    
                    "### 2. Vue.js\n\n"
                    "**Strengths**:\n"
                    "- Gentle learning curve\n"
                    "- Comprehensive documentation\n"
                    "- Flexible integration options\n"
                    "- Built-in state management and routing capabilities\n\n"
                    
                    "**Considerations**:\n"
                    "- Smaller ecosystem compared to React\n"
                    "- Fewer enterprise adoption examples\n\n"
                    
                    "### 3. Angular\n\n"
                    "**Strengths**:\n"
                    "- Complete solution with built-in tools\n"
                    "- Strong typing with TypeScript\n"
                    "- Comprehensive testing utilities\n"
                    "- Excellent for large enterprise applications\n\n"
                    
                    "**Considerations**:\n"
                    "- Steeper learning curve\n"
                    "- More verbose than other options"
                )
            elif lang.lower() in ['python', 'py']:
                response += (
                    "### 1. Django\n\n"
                    "**Strengths**:\n"
                    "- Comprehensive web framework with \"batteries included\"\n"
                    "- Strong ORM for database operations\n"
                    "- Built-in admin interface\n"
                    "- Excellent security features\n\n"
                    
                    "**Considerations**:\n"
                    "- More opinionated and potentially heavyweight for simple projects\n"
                    "- Steeper learning curve for beginners\n\n"
                    
                    "### 2. Flask\n\n"
                    "**Strengths**:\n"
                    "- Lightweight and flexible microframework\n"
                    "- Easy to learn and use\n"
                    "- Extensive extension ecosystem\n"
                    "- Good for APIs and microservices\n\n"
                    
                    "**Considerations**:\n"
                    "- Requires more manual configuration for larger projects\n"
                    "- Less built-in functionality compared to Django\n\n"
                    
                    "### 3. FastAPI\n\n"
                    "**Strengths**:\n"
                    "- High performance with async support\n"
                    "- Automatic API documentation\n"
                    "- Type hints and validation\n"
                    "- Modern and growing community\n\n"
                    
                    "**Considerations**:\n"
                    "- Newer than other options\n"
                    "- Smaller ecosystem (though growing rapidly)"
                )
            else:
                response += (
                    f"### 1. Leading Framework for {lang}\n\n"
                    f"**Strengths**:\n"
                    f"- Well-established in the {lang} ecosystem\n"
                    f"- Strong community support\n"
                    f"- Comprehensive documentation\n"
                    f"- Regular updates and maintenance\n\n"
                    
                    f"**Considerations**:\n"
                    f"- May require significant learning investment\n"
                    f"- Could be overkill for simpler projects\n\n"
                    
                    f"### 2. Lightweight Alternative\n\n"
                    f"**Strengths**:\n"
                    f"- Faster learning curve\n"
                    f"- More flexibility and less opinionated\n"
                    f"- Good performance characteristics\n"
                    f"- Easier to integrate with existing code\n\n"
                    
                    f"**Considerations**:\n"
                    f"- Less built-in functionality\n"
                    f"- Might require more manual configuration\n\n"
                    
                    f"### 3. Specialized Solution\n\n"
                    f"**Strengths**:\n"
                    f"- Optimized for specific use cases\n"
                    f"- Better performance in target scenarios\n"
                    f"- Purpose-built features for the domain\n"
                    f"- Often more innovative\n\n"
                    
                    f"**Considerations**:\n"
                    f"- More limited in general application\n"
                    f"- Potentially smaller community and support"
                )
        elif domain.lower() in ['database', 'data', 'storage']:
            response += (
                "### 1. PostgreSQL\n\n"
                "**Strengths**:\n"
                "- Robust, feature-rich relational database\n"
                "- Excellent support for advanced data types\n"
                "- Strong concurrency and ACID compliance\n"
                "- JSON support bridges relational and document models\n\n"
                
                "**Considerations**:\n"
                "- More complex setup and maintenance than some alternatives\n"
                "- Not as horizontally scalable as NoSQL options\n\n"
                
                "### 2. MongoDB\n\n"
                "**Strengths**:\n"
                "- Flexible document-oriented storage\n"
                "- Schema-less design for evolving data models\n"
                "- Horizontal scaling capabilities\n"
                "- Good performance for read-heavy workloads\n\n"
                
                "**Considerations**:\n"
                "- Less mature transaction support\n"
                "- Not ideal for highly relational data\n\n"
                
                "### 3. Redis\n\n"
                "**Strengths**:\n"
                "- Extremely fast in-memory data store\n"
                "- Versatile data structures\n"
                "- Excellent for caching and real-time applications\n"
                "- Simple to set up and use\n\n"
                
                "**Considerations**:\n"
                "- Primary in-memory nature requires appropriate sizing\n"
                "- Not intended as a primary database for all use cases"
            )
        else:
            response += (
                f"### 1. Primary Recommendation\n\n"
                f"**Strengths**:\n"
                f"- Strong alignment with {criteria}\n"
                f"- Excellent integration with {lang}\n"
                f"- Robust community support\n"
                f"- Active development and maintenance\n\n"
                
                f"**Considerations**:\n"
                f"- Learning curve may be steeper\n"
                f"- Requires understanding of key concepts\n\n"
                
                f"### 2. Alternative Option\n\n"
                f"**Strengths**:\n"
                f"- Easier entry point for beginners\n"
                f"- More flexible implementation options\n"
                f"- Good balance of features and simplicity\n"
                f"- Solid documentation\n\n"
                
                f"**Considerations**:\n"
                f"- May lack some advanced features\n"
                f"- Smaller ecosystem than the primary recommendation\n\n"
                
                f"### 3. Specialized Solution\n\n"
                f"**Strengths**:\n"
                f"- Optimized for specific aspects of {criteria}\n"
                f"- Innovative approach to common challenges\n"
                f"- Excellent performance in target scenarios\n"
                f"- Modern design principles\n\n"
                
                f"**Considerations**:\n"
                f"- More focused use cases\n"
                f"- May require integration with other tools for complete solution"
            )
        
        response += (
            f"\n\n## Implementation Guidance\n\n"
            f"When adopting any of these technologies, consider:\n\n"
            f"1. **Start small**: Begin with a pilot project to gain experience\n"
            f"2. **Evaluate thoroughly**: Test against your specific requirements\n"
            f"3. **Consider ecosystem**: Look at available libraries and integrations\n"
            f"4. **Plan for growth**: Ensure the technology can scale with your needs\n\n"
            f"## Conclusion\n\n"
            f"For your {lang} project focusing on {criteria} within the {domain} space, my primary recommendation would be the first option listed above. It offers the best balance of features, community support, and alignment with your stated criteria."
        )
        
        return response

# Utility functions for generating realistic content variations
def random_performance(option):
    import random
    performances = [
        "excellent performance for most use cases",
        "good performance with some optimization required",
        "strong performance in typical scenarios",
        "variable performance depending on implementation",
        "superior performance for specific workloads"
    ]
    return random.choice(performances)

def random_maintenance(option):
    import random
    maintenance = [
        "minimal maintenance overhead",
        "moderate maintenance with regular updates",
        "standard maintenance comparable to alternatives",
        "careful attention to updates and dependencies",
        "ongoing maintenance for security and features"
    ]
    return random.choice(maintenance)

def random_ecosystem(option):
    import random
    ecosystems = [
        "a robust ecosystem of plugins and extensions",
        "a growing community with good library support",
        "an established ecosystem with enterprise adoption",
        "a specialized ecosystem focused on core functionality",
        "a comprehensive ecosystem covering most needs"
    ]
    return random.choice(ecosystems)

def random_learning(option):
    import random
    learning = [
        "has a gentle learning curve for beginners",
        "requires moderate effort to learn effectively",
        "has an approachable learning path with good documentation",
        "demands deeper technical understanding",
        "balances ease of entry with advanced capabilities"
    ]
    return random.choice(learning)

def random_future(option):
    import random
    futures = [
        "Shows strong momentum and ongoing development",
        "Has a stable roadmap with regular updates",
        "Demonstrates commitment to backward compatibility",
        "Continues to evolve with industry trends",
        "Maintains a balance of stability and innovation"
    ]
    return random.choice(futures)

def random_strength(option):
    import random
    strengths = [
        "performance optimization",
        "developer productivity",
        "code maintainability",
        "scalability",
        "integration capabilities",
        "community support",
        "documentation quality"
    ]
    return random.choice(strengths)