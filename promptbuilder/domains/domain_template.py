"""
Expanded Standardized Domain Module Template for PromptBuilder.

This module defines the enhanced structure and requirements for domain knowledge mapping
within the PromptBuilder system. All domain modules MUST follow this template precisely 
to ensure standardization while providing more comprehensive coverage of each domain area.

REQUIREMENTS:
1. All sections are mandatory and must appear in the exact order specified
2. Section names must be exactly as specified (no variations)
3. Each section must contain EXACTLY the number of entries specified (increased for comprehensiveness)
4. All fields marked as required must be present with appropriate values
5. Naming conventions and formatting must be consistent across all domains
6. Content quality standards must be maintained throughout
7. Descriptions must be complete sentences with proper punctuation
8. All lengths and counts must be strictly adhered to

This template serves as the definitive enhanced standard for all domain modules.

Author: Made in Jurgistan
Version: 2.0.0
License: MIT
"""

from typing import Dict, Any, List

def get_domain_mapping() -> Dict[str, Any]:
    """
    Get domain technology mapping with enhanced comprehensiveness.
    
    Returns:
        Dict[str, Any]: Domain mapping with standardized comprehensive information
    """
    return {
        # DOMAIN METADATA (REQUIRED)
        "name": "Domain Name",  # REQUIRED: Official name of the domain (1-3 words)
        "description": "Concise description of the domain with its key characteristics and scope, highlighting main purposes and applications along with its significance in the broader technology landscape.",  # REQUIRED: 2-3 sentences, 30-50 words
        
        # LANGUAGES (EXACTLY 6 entries)
        "languages": [
            {
                "name": "Language Name",  # REQUIRED: Official name
                "version": "X.Y",  # REQUIRED: Current stable version in semantic versioning format
                "description": "Role and purpose of this language in the domain context, including its strengths and typical implementation scenarios, highlighting key features that make it suitable for this domain.",  # REQUIRED: 2-3 complete sentences, 25-40 words
                "popularity": "high",  # REQUIRED: Must be exactly "high", "medium", or "low"
                "typical_uses": ["Use case 1", "Use case 2", "Use case 3", "Use case 4"]  # REQUIRED: Exactly 4 use cases, each 2-6 words
            },
            # REQUIRED: Include exactly 6 language entries with the same structure
        ],
        
        # FRAMEWORKS (EXACTLY 8 entries)
        "frameworks": [  # NOTE: This section name is standardized as "frameworks" for all domains
            {
                "name": "Framework Name",  # REQUIRED: Official name
                "version": "X.Y",  # REQUIRED: Current stable version in semantic versioning format
                "description": "Purpose and unique features of this framework, highlighting what distinguishes it from alternatives and its main technical capabilities, including typical scenarios where it outperforms other solutions.",  # REQUIRED: 2-3 complete sentences, 30-45 words
                "language": "Implementation language",  # REQUIRED: Primary language(s) used, separated by slashes if multiple
                "typical_uses": ["Use case 1", "Use case 2", "Use case 3", "Use case 4"],  # REQUIRED: Exactly 4 use cases, each 2-6 words
                "learning_curve": "moderate",  # REQUIRED: Must be exactly "steep", "moderate", or "gentle"
                "category": "Subcategory"  # REQUIRED: Specific subcategory within the domain, 1-3 words
            },
            # REQUIRED: Include exactly 8 framework entries with the same structure
        ],
        
        # TECHNOLOGIES (EXACTLY 8 entries)
        "technologies": [
            {
                "name": "Technology Name",  # REQUIRED: Official name
                "version": "X.Y",  # REQUIRED: Current version or "N/A" if not applicable
                "description": "Role and significance of this technology in the domain, including its core functionality and impact on development or implementation processes, with details on integration points with other domain technologies.",  # REQUIRED: 2-3 complete sentences, 25-40 words
                "category": "Category",  # REQUIRED: Broad category, 1-3 words
                "maturity": "established"  # REQUIRED: Must be exactly "emerging", "established", "mature", or "legacy"
            },
            # REQUIRED: Include exactly 8 technology entries with the same structure
        ],
        
        # CONCEPTS (EXACTLY 50 entries)
        "concepts": [
            "Concept 1",  # Each concept should be a domain-specific term or principle, 1-5 words
            "Concept 2",
            # Continue until exactly 50 concepts are listed in alphabetical order or logical grouping
            "Concept 50"
        ],
        
        # BEST PRACTICES (EXACTLY 10 entries)
        "best_practices": [
            {
                "name": "Best Practice Name",  # REQUIRED: Clear, descriptive name, 2-6 words
                "description": "Detailed explanation of the practice and why it's important for successful implementation or development in this domain, including how it addresses common challenges or enhances outcomes.",  # REQUIRED: 2-3 complete sentences, 30-50 words
                "benefits": ["Benefit 1", "Benefit 2", "Benefit 3", "Benefit 4"],  # REQUIRED: Exactly 4 benefits, each 2-5 words
                "related_concepts": ["Related concept 1", "Related concept 2", "Related concept 3", "Related concept 4"]  # REQUIRED: Exactly 4 concepts from the concepts list
            },
            # REQUIRED: Include exactly 10 best practice entries with the same structure
        ],
        
        # COMMON PROBLEMS (EXACTLY 10 entries)
        "common_problems": [
            {
                "name": "Problem Name",  # REQUIRED: Clear, descriptive name, 2-5 words
                "description": "Concise description of the problem and its impact on development, implementation, or outcomes in this domain.",  # REQUIRED: 1-2 complete sentences, 20-30 words
                "indicators": ["Indicator 1", "Indicator 2", "Indicator 3", "Indicator 4"],  # REQUIRED: Exactly 4 indicators, each 2-8 words
                "causes": ["Cause 1", "Cause 2", "Cause 3", "Cause 4", "Cause 5"],  # REQUIRED: Exactly 5 causes, each 2-8 words
                "solutions": ["Solution 1", "Solution 2", "Solution 3", "Solution 4", "Solution 5"]  # REQUIRED: Exactly 5 solutions, each 2-10 words
            },
            # REQUIRED: Include exactly 10 common problem entries with the same structure
        ],
        
        # CODE EXAMPLES (EXACTLY 8 entries)
        "code_examples": [
            {
                "task": "Task Name",  # REQUIRED: Clear, descriptive name, 3-8 words
                "description": "What this code accomplishes and when to use it, including specific use cases and expected outcomes.",  # REQUIRED: 2-3 sentences, 25-40 words
                "language": "Language Name",  # REQUIRED: Programming language used
                "code": """
# Example code here with proper formatting and indentation
def example_function():
    # Include 75-250 lines of actual working code
    # that demonstrates a common task in the domain
    # with proper error handling, commenting, and best practices
    # Code should be production-quality and address edge cases
    return "example"
                """,  # REQUIRED: 75-250 lines of code
                "explanation": "Detailed explanation of how the code works, key techniques used, and why certain approaches were chosen. Should include references to domain-specific concepts and best practices demonstrated in the code. Include information about potential optimizations, edge cases handled, and performance considerations."  # REQUIRED: 3-5 paragraphs, 200-350 words
            },
            # REQUIRED: Include exactly 8 code example entries with the same structure
        ],
        
        # PROJECT CONTEXTS (EXACTLY 20 entries)
        "project_contexts": [
            "Project context 1",  # Each entry should be a common project type or scenario, 3-10 words
            "Project context 2",
            # Continue until exactly 20 project contexts are listed in order of frequency or significance
            "Project context 20"
        ],
        
        # ARCHITECTURE PATTERNS (EXACTLY 12 entries)
        "architecture_patterns": [
            {
                "name": "Pattern Name",  # REQUIRED: Official or common name, 2-6 words
                "description": "Concise explanation of the architecture pattern, its core principles, and when it should be applied.",  # REQUIRED: 2-3 sentences, 30-50 words
                "use_cases": ["Use case 1", "Use case 2", "Use case 3", "Use case 4"],  # REQUIRED: Exactly 4 use cases, each 2-8 words
                "pros": ["Advantage 1", "Advantage 2", "Advantage 3", "Advantage 4", "Advantage 5"],  # REQUIRED: Exactly 5 pros, each 2-8 words
                "cons": ["Disadvantage 1", "Disadvantage 2", "Disadvantage 3", "Disadvantage 4", "Disadvantage 5"]  # REQUIRED: Exactly 5 cons, each 2-8 words
            },
            # REQUIRED: Include exactly 12 architecture pattern entries with the same structure
        ],
        
        # DOMAIN RESOURCES (EXACTLY 10 entries)
        "resources": [
            {
                "name": "Resource Name",  # REQUIRED: Official name
                "type": "documentation",  # REQUIRED: Must be exactly "documentation", "tutorial", "book", "community", or "tool"
                "description": "Brief explanation of what this resource offers and why it's valuable for practitioners or learners in this domain.",  # REQUIRED: 1-2 sentences, 15-30 words
                "url": "https://example.com/resource"  # REQUIRED: Valid URL to the resource or its information page
            },
            # REQUIRED: Include exactly 10 resource entries with the same structure, with at least 2 of each type
        ]
    }
