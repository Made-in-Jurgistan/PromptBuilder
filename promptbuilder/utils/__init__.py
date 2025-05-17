"""
Utilities module for PromptBuilder.

This package contains utility functions and helpers used throughout 
the PromptBuilder system, including logging configuration,
file handling, and general helper functions.

Functions:
    setup_logging: Configure logging with consistent formatting
    log_dict: Helper for logging dictionary contents
    get_logger: Get a configured logger instance

Author: Made in Jurgistan
Version: 2.0.0
License: MIT
"""

from promptbuilder.utils.logging import (
    setup_logging,
    log_dict,
    get_logger
)

__all__ = [
    'setup_logging',
    'log_dict',
    'get_logger'
]