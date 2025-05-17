"""
Domain mapping initialization module for PromptBuilder.

This module imports all domain modules and provides a unified interface
for accessing the domain technology mappings.

Author: Made in Jurgistan
Version: 2.0.0
License: MIT
"""

from typing import Dict, Any, List, Callable

# Import available domain modules
import importlib
import logging
import os
from pathlib import Path

# Define all possible domain modules
domain_modules = {
    "web_development": "Web Development",
    "ai_ml": "AI/ML",
    "devops": "DevOps",
    "cybersecurity": "Cybersecurity",
    "data_science": "Data Science",
    "cloud_computing": "Cloud Computing",
    "blockchain": "Blockchain",
    "mobile_development": "Mobile Development",
    "iot": "IoT",
    "game_development": "Game Development"
}

# Initialize logger for this module
logger = logging.getLogger(__name__)

# Import available domain modules dynamically
available_domains = {}

def _load_available_domains():
    """Load all available domain modules dynamically.
    
    Returns:
        Dict[str, Callable]: Dictionary mapping domain names to their get_*_mapping functions
    """
    domains = {}
    
    # Get the directory where this module is located
    module_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    
    # Check each potential domain module
    for module_name, domain_name in domain_modules.items():
        module_path = module_dir / f"{module_name}.py"
        
        if module_path.exists():
            try:
                # Dynamically import the module
                module = importlib.import_module(f"domains.{module_name}")
                getter_name = f"get_{module_name}_mapping"
                
                if hasattr(module, getter_name):
                    domains[domain_name] = getattr(module, getter_name)
                    logger.debug(f"Successfully loaded domain module: {module_name}")
                else:
                    logger.warning(f"Module {module_name} does not have required function {getter_name}")
            except ImportError as e:
                logger.warning(f"Could not import domain module {module_name}: {e}")
        else:
            logger.debug(f"Domain module file not found: {module_path}")
    
    # Ensure we have at least one domain
    if not domains:
        logger.warning("No domain modules found. Using fallback minimal domain.")
        # Could add a fallback minimal domain here if needed
    
    return domains

def get_available_domains() -> List[str]:
    """Get a list of available domain names.
    
    Returns:
        List[str]: List of available domain names
    """
    global available_domains
    if not available_domains:
        available_domains = _load_available_domains()
    
    return list(available_domains.keys())

def get_domain_mapping() -> Dict[str, Dict[str, Any]]:
    """Get all domain technology mappings.
    
    Returns:
        Dict[str, Dict[str, Any]]: Mapping of domain names to their technology mappings
    """
    global available_domains
    if not available_domains:
        available_domains = _load_available_domains()
    
    result = {}
    for domain_name, mapping_func in available_domains.items():
        try:
            result[domain_name] = mapping_func()
        except Exception as e:
            logger.error(f"Error getting mapping for domain {domain_name}: {e}")
    
    return result
