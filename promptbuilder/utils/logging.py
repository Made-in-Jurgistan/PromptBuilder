"""
Logging utilities for PromptBuilder.

This module provides a comprehensive set of logging utilities for configuring,
managing, and extending the Python logging system within the PromptBuilder platform. 
It enables consistent logging across the application with flexible output formats, 
multiple destinations, and customizable log levels.

Key features:
  - Centralized logging configuration
  - Console and file logging support
  - Customizable log formats and levels
  - JSON logging for machine-readable output
  - Special utilities for logging complex data structures
  - Context managers for scoped logging configuration
  - Performance monitoring through timed logging

Functions:
    setup_logging: Configure logging with consistent formatting
    get_logger: Get a configured logger instance
    log_dict: Helper for logging dictionary contents
    log_exception: Helper for enhanced exception logging
    timed_operation: Context manager for timing operations

Classes:
    JsonFormatter: JSON formatter for structured logging
    LoggingContext: Context manager for temporary logging configuration

Author: Made in Jurgistan
Version: 2.1.0
License: MIT
"""

import json
import logging
import os
import sys
import time
import traceback
import socket
import inspect
import threading
import hashlib
from contextlib import contextmanager
from datetime import datetime
from functools import wraps
from typing import Dict, Any, Optional, Union, List, Callable, TypeVar, Iterator, TextIO, cast, Set

# Type variables for generic functions
T = TypeVar('T')
R = TypeVar('R')

# Default log message format
DEFAULT_LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# Default maximum recursion depth for structure logging
DEFAULT_MAX_DEPTH = 5


class JsonFormatter(logging.Formatter):
    """JSON formatter for structured logging.
    
    Formats log records as JSON objects for easy parsing and analysis by
    log management systems. Supports customizable fields and structured
    exception information.
    
    Attributes:
        include_timestamp: Whether to include timestamp in output
        timestamp_format: Format string for timestamp
        include_hostname: Whether to include hostname in output
        include_thread: Whether to include thread info in output
        include_process: Whether to include process info in output
        include_path: Whether to include file path info in output
        extra_fields: Additional fields to include in all log entries
        hostname: Cached hostname value when include_hostname is True
    """
    
    def __init__(
        self,
        include_timestamp: bool = True,
        timestamp_format: str = "%Y-%m-%d %H:%M:%S.%f",
        include_hostname: bool = False,
        include_thread: bool = False,
        include_process: bool = False,
        include_path: bool = False,
        include_corr_id: bool = True,
        extra_fields: Optional[Dict[str, Any]] = None,
        sanitize_keys: Optional[Set[str]] = None
    ):
        """Initialize the formatter.
        
        Args:
            include_timestamp: Whether to include timestamp
            timestamp_format: Format string for timestamp
            include_hostname: Whether to include hostname
            include_thread: Whether to include thread name and ID
            include_process: Whether to include process name and ID
            include_path: Whether to include file path and line number
            include_corr_id: Whether to include correlation ID
            extra_fields: Additional fields to include in every log entry
            sanitize_keys: Set of keys to sanitize in log output
        """
        super().__init__()
        self.include_timestamp = include_timestamp
        self.timestamp_format = timestamp_format
        self.include_hostname = include_hostname
        self.include_thread = include_thread
        self.include_process = include_process
        self.include_path = include_path
        self.include_corr_id = include_corr_id
        self.extra_fields = extra_fields or {}
        self.sanitize_keys = sanitize_keys or {"password", "secret", "token", "key", "auth", "credential"}
        
        # Cache hostname if needed
        if self.include_hostname:
            try:
                self.hostname = socket.gethostname()
            except Exception:
                self.hostname = "unknown"
    
    def format(self, record: logging.LogRecord) -> str:
        """Format the log record as JSON.
        
        Converts a LogRecord into a structured JSON format with all
        configured fields and attributes.
        
        Args:
            record: Log record to format
            
        Returns:
            str: JSON-formatted log entry
        """
        # Start with base log data
        log_data = {
            "level": record.levelname,
            "logger": record.name,
            "message": self._sanitize_message(record.getMessage()),
        }
        
        # Include correlation ID if available and requested
        if self.include_corr_id:
            corr_id = getattr(record, "correlation_id", None)
            if corr_id:
                log_data["correlation_id"] = corr_id
            
            # Also include operation_id if present for backwards compatibility
            op_id = getattr(record, "operation_id", None)
            if op_id and not corr_id:
                log_data["correlation_id"] = op_id
        
        # Add timestamp if requested
        if self.include_timestamp:
            log_time = datetime.fromtimestamp(record.created)
            log_data["timestamp"] = log_time.strftime(self.timestamp_format)
        
        # Add hostname if requested
        if self.include_hostname:
            log_data["hostname"] = self.hostname
        
        # Add thread information if requested
        if self.include_thread:
            log_data["thread"] = {
                "id": record.thread,
                "name": record.threadName
            }
        
        # Add process information if requested
        if self.include_process:
            log_data["process"] = {
                "id": record.process,
                "name": record.processName
            }
        
        # Add file path and line number if requested
        if self.include_path:
            log_data["path"] = {
                "file": record.pathname,
                "line": record.lineno,
                "function": record.funcName
            }
        
        # Add extra fields
        log_data.update(self.extra_fields)
        
        # Add extra attributes from the record
        if hasattr(record, "extra") and isinstance(record.extra, dict):
            extra_data = self._sanitize_data(record.extra)
            log_data.update(extra_data)
        
        # Add exception info if present
        if record.exc_info:
            exc_type, exc_value, _ = record.exc_info
            
            # Use traceback text if already rendered, otherwise format it
            if record.exc_text:
                tb_text = record.exc_text
            else:
                tb_text = ''.join(traceback.format_exception(*record.exc_info))
            
            log_data["exception"] = {
                "type": exc_type.__name__,
                "message": str(exc_value),
                "traceback": tb_text
            }
            
            # Add exception attributes as additional context if available
            exc_attrs = self._get_exception_attributes(exc_value)
            if exc_attrs:
                log_data["exception"]["attributes"] = exc_attrs
        
        # Format as JSON with fallback for non-serializable objects
        try:
            return json.dumps(log_data, default=str)
        except (TypeError, ValueError) as e:
            # Fallback for serialization errors
            return json.dumps({
                "level": "ERROR",
                "logger": record.name,
                "message": f"Error serializing log record: {e}",
                "original_message": str(record.getMessage())
            })
    
    def _get_exception_attributes(self, exc: Exception) -> Dict[str, Any]:
        """Extract non-standard attributes from an exception.
        
        Args:
            exc: Exception to extract attributes from
            
        Returns:
            Dict[str, Any]: Dictionary of exception attributes
        """
        result = {}
        
        # Skip standard exception attributes
        standard_attrs = {"args", "__cause__", "__context__", "__traceback__"}
        
        for attr in dir(exc):
            # Skip private attributes and methods
            if attr.startswith("_") or attr in standard_attrs:
                continue
                
            try:
                value = getattr(exc, attr)
                # Skip callable attributes (methods)
                if callable(value):
                    continue
                    
                # Add attribute to result
                result[attr] = self._sanitize_data({attr: value})[attr]
            except Exception:
                # Skip attributes that can't be accessed
                continue
                
        return result
    
    def _sanitize_message(self, message: str) -> str:
        """Sanitize potentially sensitive information in log messages.
        
        Args:
            message: Log message to sanitize
            
        Returns:
            str: Sanitized message
        """
        # Simple sanitization approach - could be extended with regex patterns
        sanitized = message
        
        # Sanitize common patterns like key=value where key is sensitive
        for key in self.sanitize_keys:
            # Match patterns like 'password=abc123' or 'token: xyz'
            patterns = [
                f"{key}=",
                f"{key}:",
                f'"{key}":',
                f"'{key}':",
            ]
            
            for pattern in patterns:
                pattern_lower = pattern.lower()
                
                # Look for the pattern in the message (case insensitive)
                i = sanitized.lower().find(pattern_lower)
                while i >= 0:
                    # Find the value part
                    start = i + len(pattern)
                    if start < len(sanitized):
                        # Find end of value (space, comma, etc.)
                        end = start
                        in_quotes = False
                        quote_char = None
                        
                        # Check if value starts with a quote
                        if start < len(sanitized) and sanitized[start] in ('"', "'"):
                            in_quotes = True
                            quote_char = sanitized[start]
                            end += 1
                            
                        # Find end of value
                        while end < len(sanitized):
                            if in_quotes:
                                if sanitized[end] == quote_char and sanitized[end-1] != '\\':
                                    end += 1
                                    break
                            elif sanitized[end] in ' ,}])\n\r\t':
                                break
                            end += 1
                            
                        # Replace value with asterisks
                        value_length = end - start
                        if value_length > 0:
                            if value_length < 8:
                                sanitized = sanitized[:start] + "****" + sanitized[end:]
                            else:
                                sanitized = sanitized[:start] + "********" + sanitized[end:]
                                
                    # Find next occurrence
                    i = sanitized.lower().find(pattern_lower, i + 1)
        
        return sanitized
    
    def _sanitize_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize potentially sensitive data in dictionaries.
        
        Args:
            data: Dictionary to sanitize
            
        Returns:
            Dict[str, Any]: Sanitized dictionary
        """
        result = {}
        
        for key, value in data.items():
            # Check if key contains any sensitive keywords
            key_lower = key.lower()
            is_sensitive = any(sensitive in key_lower for sensitive in self.sanitize_keys)
            
            if is_sensitive:
                # Mask sensitive values
                if isinstance(value, str):
                    if value:  # Only mask non-empty strings
                        result[key] = "********"
                    else:
                        result[key] = ""
                elif value is None:
                    result[key] = None
                else:
                    result[key] = "[REDACTED]"
            elif isinstance(value, dict):
                # Recursively sanitize nested dictionaries
                result[key] = self._sanitize_data(value)
            else:
                # Pass through non-sensitive values
                result[key] = value
                
        return result


class SafeJsonEncoder(json.JSONEncoder):
    """JSON encoder that safely handles non-serializable objects.
    
    Provides fallback serialization for complex objects like sets,
    custom classes, and other non-JSON-serializable types.
    """
    
    def __init__(self, max_collection_size: int = 50, **kwargs):
        """Initialize the encoder.
        
        Args:
            max_collection_size: Maximum collection size before summarizing
            **kwargs: Additional arguments for JSONEncoder
        """
        super().__init__(**kwargs)
        self.max_collection_size = max_collection_size
        
    def default(self, obj: Any) -> Any:
        """Provide custom serialization for non-serializable objects.
        
        Args:
            obj: Object to serialize
            
        Returns:
            Serializable version of the object
        """
        try:
            # Handle objects with to_dict method
            if hasattr(obj, 'to_dict') and callable(obj.to_dict):
                return obj.to_dict()
                
            # Handle datetime objects
            if hasattr(obj, 'isoformat') and callable(obj.isoformat):
                return obj.isoformat()
                
            # Handle sets and frozensets
            if isinstance(obj, (set, frozenset)):
                if len(obj) > self.max_collection_size:
                    return f"<{obj.__class__.__name__} with {len(obj)} items>"
                return list(obj)
                
            # Handle large collections
            if isinstance(obj, (list, tuple)):
                if len(obj) > self.max_collection_size:
                    return f"<{obj.__class__.__name__} with {len(obj)} items>"
                return list(obj)
                
            # Handle custom objects with __dict__
            if hasattr(obj, '__dict__'):
                return {k: v for k, v in obj.__dict__.items() if not k.startswith('_')}
                
            # Handle bytes
            if isinstance(obj, bytes):
                try:
                    return obj.decode('utf-8')
                except UnicodeDecodeError:
                    return f"<binary data ({len(obj)} bytes)>"
                    
            # Default to string representation
            return str(obj)
            
        except Exception as e:
            # Fallback for any errors
            return f"<{obj.__class__.__name__} (error: {e})>"


def generate_correlation_id(prefix: str = "op") -> str:
    """Generate a correlation ID for tracking related log entries.
    
    Args:
        prefix: Optional prefix for the correlation ID
        
    Returns:
        str: Generated correlation ID
    """
    # Create a hash from current time and thread ID for uniqueness
    timestamp = datetime.now().isoformat()
    thread_id = threading.get_ident()
    
    # Create a short hash
    hash_input = f"{timestamp}-{thread_id}-{os.getpid()}"
    hash_value = hashlib.md5(hash_input.encode()).hexdigest()[:8]
    
    return f"{prefix}-{hash_value}"


def setup_logging(
    log_level: int = logging.INFO,
    log_file: Optional[str] = None,
    log_format: Optional[str] = None,
    date_format: str = "%Y-%m-%d %H:%M:%S",
    include_timestamp: bool = True,
    console_output: bool = True,
    console_level: Optional[int] = None,
    file_level: Optional[int] = None,
    file_mode: str = 'a',
    file_max_bytes: int = 10 * 1024 * 1024,  # 10 MB
    file_backup_count: int = 5,
    json_format: bool = False,
    json_options: Optional[Dict[str, Any]] = None,
    propagate: bool = True,
    log_libraries: Optional[List[str]] = None,
    library_level: int = logging.WARNING,
    correlation_id: Optional[str] = None
) -> None:
    """Set up logging with comprehensive configuration options.
    
    Configures the Python logging system with standardized formatting,
    multiple output destinations, and flexible level settings.
    
    Args:
        log_level: The logging level to use (default: logging.INFO)
        log_file: Optional file path for logging to a file
        log_format: Optional custom log format string
        date_format: Date format string for timestamp formatting
        include_timestamp: Whether to include timestamp in log format
        console_output: Whether to output logs to console
        console_level: Specific level for console logging (default: log_level)
        file_level: Specific level for file logging (default: log_level)
        file_mode: File opening mode ('a' for append, 'w' for overwrite)
        file_max_bytes: Maximum log file size before rotation
        file_backup_count: Number of backup files to keep
        json_format: Whether to format logs as JSON
        json_options: Options for JSON formatter
        propagate: Whether to propagate logs to parent loggers
        log_libraries: List of third-party libraries to configure logging for
        library_level: Logging level for third-party libraries
        correlation_id: Optional correlation ID for tracing related log entries
        
    Returns:
        None
    """
    # Create the root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Remove existing handlers to prevent duplicate logs
    for handler in root_logger.handlers[:]:
        handler.close()
        root_logger.removeHandler(handler)
    
    # Generate correlation ID if needed
    if correlation_id is None:
        correlation_id = generate_correlation_id()
    
    # Default log format if not specified
    if log_format is None:
        if include_timestamp:
            log_format = DEFAULT_LOG_FORMAT
        else:
            log_format = '%(name)s - %(levelname)s - %(message)s'
    
    # Use specific levels or fall back to the main log level
    console_level = console_level if console_level is not None else log_level
    file_level = file_level if file_level is not None else log_level
    
    # Configure formatters
    if json_format:
        json_options = json_options or {}
        formatter = JsonFormatter(
            include_timestamp=include_timestamp,
            timestamp_format=date_format,
            **json_options
        )
    else:
        formatter = logging.Formatter(log_format, datefmt=date_format)
    
    # Add console handler if requested
    if console_output:
        try:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(console_level)
            console_handler.setFormatter(formatter)
            root_logger.addHandler(console_handler)
        except Exception as e:
            # Log error but continue with other handlers
            print(f"Error setting up console logging: {e}")
    
    # Add a file handler if requested
    if log_file:
        try:
            # Create directory for log file if it doesn't exist
            log_dir = os.path.dirname(log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)
            
            # Use rotating file handler if rotation is enabled
            if file_max_bytes > 0:
                from logging.handlers import RotatingFileHandler
                file_handler = RotatingFileHandler(
                    log_file,
                    mode=file_mode,
                    maxBytes=file_max_bytes,
                    backupCount=file_backup_count,
                    encoding='utf-8'
                )
            else:
                file_handler = logging.FileHandler(
                    log_file,
                    mode=file_mode,
                    encoding='utf-8'
                )
                
            file_handler.setLevel(file_level)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
        except Exception as e:
            # Log error but continue with other handlers
            if console_output:
                root_logger.error(f"Error setting up file logging: {e}")
            else:
                print(f"Error setting up file logging: {e}")
    
    # Configure specific libraries
    if log_libraries:
        for library in log_libraries:
            try:
                lib_logger = logging.getLogger(library)
                lib_logger.setLevel(library_level)
                lib_logger.propagate = propagate
            except Exception as e:
                root_logger.warning(f"Error configuring logging for library {library}: {e}")
    
    # Create a logger for this function and log the initialization
    logger = get_logger("logging_utils", extra={"correlation_id": correlation_id})
    logger.info(f"Logging initialized with level: {logging.getLevelName(log_level)}")
    if log_file:
        logger.info(f"Logging to file: {log_file}")
    
    # Log correlation ID for tracing
    logger.info(f"Correlation ID: {correlation_id}")


def get_logger(
    name: str,
    level: Optional[int] = None,
    propagate: bool = True,
    extra: Optional[Dict[str, Any]] = None,
    correlation_id: Optional[str] = None
) -> Union[logging.Logger, logging.LoggerAdapter]:
    """Get a logger with the specified name and configuration.
    
    Creates or retrieves a logger with the specified name and applies
    optional configuration settings.
    
    Args:
        name: Name for the logger
        level: Optional specific log level for this logger
        propagate: Whether to propagate logs to parent loggers
        extra: Optional extra attributes to include in all log records
        correlation_id: Optional correlation ID for tracing related log entries
        
    Returns:
        Union[logging.Logger, logging.LoggerAdapter]: Configured logger
    """
    logger = logging.getLogger(name)
    
    if level is not None:
        logger.setLevel(level)
    
    logger.propagate = propagate
    
    # Prepare extra context dictionary
    context = extra or {}
    
    # Add correlation ID if provided
    if correlation_id and "correlation_id" not in context:
        context["correlation_id"] = correlation_id
    
    # Add extra context if provided
    if context:
        logger = logging.LoggerAdapter(logger, {"extra": context})
    
    return logger


def log_dict(
    logger: Union[logging.Logger, logging.LoggerAdapter],
    message: str,
    data: Dict[str, Any],
    level: int = logging.INFO,
    indent: int = 2,
    max_depth: int = DEFAULT_MAX_DEPTH,
    filter_keys: Optional[List[str]] = None,
    mask_values: Optional[Dict[str, str]] = None,
    max_collection_size: int = 50,
    correlation_id: Optional[str] = None
) -> None:
    """Log a dictionary with proper formatting and security controls.
    
    Formats dictionary contents for readable logging, with options
    for controlling indentation, maximum depth, and key filtering.
    Includes security features for masking sensitive values.
    
    Args:
        logger: Logger instance to use
        message: Message to prepend to the dictionary
        data: Dictionary to log
        level: Logging level to use
        indent: Number of spaces for indentation
        max_depth: Maximum depth to traverse nested dictionaries
        filter_keys: Optional list of keys to exclude from logging
        mask_values: Optional dictionary mapping keys to mask values
        max_collection_size: Maximum size for collections before truncation
        correlation_id: Optional correlation ID for tracing
        
    Returns:
        None
    """
    if data is None:
        logger.log(level, f"{message}: <null>")
        return
        
    if not isinstance(data, dict):
        logger.log(level, f"{message}: <not a dictionary, got {type(data).__name__}>")
        return
        
    if not data:
        logger.log(level, f"{message}: <empty dictionary>")
        return
    
    # Add correlation ID to context if provided
    extra = {}
    if correlation_id:
        extra = {"correlation_id": correlation_id}
    
    # Apply context if needed
    if extra and not isinstance(logger, logging.LoggerAdapter):
        logger = logging.LoggerAdapter(logger, {"extra": extra})
    
    # Copy the data to avoid modifying the original
    filtered_data = {}
    filter_keys = filter_keys or []
    mask_values = mask_values or {}
    
    # Always filter sensitive keys
    default_sensitive_keys = {"password", "secret", "token", "api_key", "auth_token", "credentials"}
    expanded_filter_keys = set(filter_keys) | default_sensitive_keys
    
    # Filter and mask the data
    for key, value in data.items():
        # Skip filtered keys
        if key in expanded_filter_keys:
            continue
        
        # Mask sensitive values
        if key in mask_values:
            filtered_data[key] = mask_values[key]
        elif any(sensitive in key.lower() for sensitive in default_sensitive_keys):
            # Automatically mask sensitive keys
            if value is not None and value != "":
                filtered_data[key] = "********"
            else:
                filtered_data[key] = value
        else:
            filtered_data[key] = value
    
    # Format dictionary with indentation and limited depth
    try:
        # Use custom encoder for handling complex objects
        formatted_data = json.dumps(
            filtered_data,
            indent=indent,
            cls=SafeJsonEncoder,
            default=str,
            ensure_ascii=False,
            sort_keys=True,
            max_collection_size=max_collection_size
        )
        logger.log(level, f"{message}:\n{formatted_data}")
    except (TypeError, ValueError) as e:
        # Fallback for non-serializable dictionaries
        formatted_data = _format_dict_fallback(
            filtered_data, 
            indent=indent, 
            max_depth=max_depth,
            max_collection_size=max_collection_size,
            current_depth=0,
            current_indent=0
        )
        logger.log(level, f"{message}:\n{formatted_data}")
        logger.debug(f"JSON serialization error: {e}")


def _format_dict_fallback(
    data: Dict[str, Any],
    indent: int = 2,
    max_depth: int = DEFAULT_MAX_DEPTH,
    current_depth: int = 0,
    current_indent: int = 0,
    max_collection_size: int = 50
) -> str:
    """Format dictionary recursively with fallback approach.
    
    Internal helper for log_dict when JSON output is not possible.
    
    Args:
        data: Dictionary to format
        indent: Indentation spaces
        max_depth: Maximum recursion depth
        current_depth: Current recursion depth
        current_indent: Current indentation level
        max_collection_size: Maximum number of collection items to display
        
    Returns:
        str: Formatted string representation
    """
    
    if not isinstance(data, dict):
        if isinstance(data, (list, tuple, set, frozenset)):
            if len(data) > max_collection_size:
                return " " * current_indent + f"<{data.__class__.__name__} with {len(data)} items>"
            
            items = []
            item_indent = " " * (current_indent + indent)
            
            for item in data:
                if isinstance(item, dict):
                    items.append(_format_dict_fallback(
                        item, indent, max_depth, current_depth + 1, current_indent + indent,
                        max_collection_size
                    ))
                else:
                    items.append(f"{item_indent}{str(item)}")
                    
            return "\n".join(items)
        else:
            return " " * current_indent + str(data)


def log_exception(
    logger: Union[logging.Logger, logging.LoggerAdapter],
    exception: Exception,
    message: str = "An error occurred",
    level: int = logging.ERROR,
    include_traceback: bool = True,
    include_context: bool = True,
    context_data: Optional[Dict[str, Any]] = None,
    correlation_id: Optional[str] = None
) -> None:
    """Log an exception with enhanced details.
    
    Provides detailed exception logging with optional context information
    and traceback formatting.
    
    Args:
        logger: Logger instance to use
        exception: The exception to log
        message: Message to include with the exception
        level: Logging level to use
        include_traceback: Whether to include the traceback
        include_context: Whether to include exception context
        context_data: Additional context data to include
        correlation_id: Optional correlation ID for tracing
        
    Returns:
        None
    """
    if not isinstance(exception, Exception):
        logger.log(level, f"Invalid exception object: {type(exception).__name__}")
        return
        
    # Add correlation ID to context if provided
    extra = {}
    if correlation_id:
        extra = {"correlation_id": correlation_id}
    
    # Apply context if needed
    if extra and not isinstance(logger, logging.LoggerAdapter):
        logger = logging.LoggerAdapter(logger, {"extra": extra})
    
    exc_type = type(exception).__name__
    exc_message = str(exception)
    
    # Base log message
    log_message = f"{message}: {exc_type}: {exc_message}"
    
    # Add context information if requested
    if include_context:
        context = {}
        
        # Add exception attributes
        for attr in dir(exception):
            if not attr.startswith('_') and attr not in ('args', 'with_traceback'):
                try:
                    value = getattr(exception, attr)
                    if not callable(value):
                        context[attr] = value
                except Exception:
                    pass
        
        # Add custom context data
        if context_data:
            context.update(context_data)
        
        # Add additional context from stack frames
        if include_traceback:
            try:
                # Get the current stack frame
                frame = inspect.currentframe()
                if frame:
                    # Go back a few frames to capture caller context
                    for _ in range(2):
                        if frame.f_back:
                            frame = frame.f_back
                    
                    # Extract local variables from the frame
                    locals_dict = frame.f_locals.copy()
                    
                    # Filter sensitive information and add to context
                    filtered_locals = {}
                    for k, v in locals_dict.items():
                        # Skip self, logger, and special variables
                        if k in ('self', 'logger', 'exception') or k.startswith('__'):
                            continue
                            
                        # Include only simple types to avoid circular references
                        if isinstance(v, (str, int, float, bool, type(None))):
                            filtered_locals[k] = v
                            
                    if filtered_locals:
                        context['locals'] = filtered_locals
            except Exception:
                # Ignore errors in stack frame processing
                pass
        
        # Include context in log if not empty
        if context:
            log_dict(logger, f"{log_message}\nContext", context, level=level)
        else:
            logger.log(level, log_message)
    else:
        logger.log(level, log_message)
    
    # Log traceback if requested
    if include_traceback:
        tb_str = ''.join(traceback.format_exception(type(exception), exception, exception.__traceback__))
        logger.log(level, f"Traceback:\n{tb_str}")


@contextmanager
def timed_operation(
    logger: Union[logging.Logger, logging.LoggerAdapter],
    operation_name: str,
    level: int = logging.INFO,
    log_start: bool = True,
    log_exception: bool = True,
    threshold_ms: Optional[int] = None,
    include_args: bool = False,
    args: Optional[tuple] = None,
    kwargs: Optional[dict] = None,
    correlation_id: Optional[str] = None
) -> Iterator[Dict[str, Any]]:
    """Context manager for timing operations.
    
    Measures the execution time of a code block and logs it, with options
    for logging only operations that exceed a threshold.
    
    Args:
        logger: Logger instance to use
        operation_name: Name of the operation for logging
        level: Logging level to use
        log_start: Whether to log when the operation starts
        log_exception: Whether to log exceptions
        threshold_ms: Only log operations that exceed this duration in ms
        include_args: Whether to include operation arguments in logs
        args: Positional arguments to the operation (for logging)
        kwargs: Keyword arguments to the operation (for logging)
        correlation_id: Optional correlation ID for tracing
        
    Yields:
        Dict[str, Any]: Context dictionary with timing information
        
    Example:
        with timed_operation(logger, "Database query") as context:
            results = db.execute_query(...)
            context["row_count"] = len(results)
    """
    # Add correlation ID to context if provided
    extra = {}
    if correlation_id:
        extra = {"correlation_id": correlation_id}
    
    # Apply context if needed
    if extra and not isinstance(logger, logging.LoggerAdapter):
        logger = logging.LoggerAdapter(logger, {"extra": extra})
    
    # Create context dictionary for the operation
    context = {
        "operation": operation_name,
        "start_time": time.time(),
        "metrics": {}
    }
    
    # Log operation start if requested
    if log_start:
        if include_args and (args or kwargs):
            args_str = ", ".join(repr(a) for a in (args or ()))
            kwargs_str = ", ".join(f"{k}={repr(v)}" for k, v in (kwargs or {}).items())
            params = f"{args_str}{', ' if args and kwargs else ''}{kwargs_str}"
            logger.log(level, f"Starting {operation_name}({params})")
        else:
            logger.log(level, f"Starting {operation_name}")
    
    try:
        # Yield context to caller
        yield context
        
    except Exception as e:
        # Calculate duration
        end_time = time.time()
        duration_ms = (end_time - context["start_time"]) * 1000
        context["duration_ms"] = duration_ms
        
        # Log exception with duration
        logger.log(
            level, 
            f"{operation_name} failed after {duration_ms:.2f}ms: {type(e).__name__}: {e}"
        )
        
        # Log full exception details if requested
        if log_exception:
            log_exception(
                logger, 
                e, 
                message=f"{operation_name} failed",
                context_data={"duration_ms": duration_ms}
            )
            
        # Re-raise the exception
        raise
        
    else:
        # Calculate duration
        end_time = time.time()
        duration_ms = (end_time - context["start_time"]) * 1000
        context["duration_ms"] = duration_ms
        
        # Log completion based on threshold
        if threshold_ms is None or duration_ms >= threshold_ms:
            # Include any metrics added to the context
            metrics_str = ""
            if context["metrics"]:
                metrics_items = [f"{k}={v}" for k, v in context["metrics"].items()]
                metrics_str = f" ({', '.join(metrics_items)})"
                
            logger.log(level, f"{operation_name} completed in {duration_ms:.2f}ms{metrics_str}")


@contextmanager
def log_scope(
    logger: Union[logging.Logger, logging.LoggerAdapter],
    scope_name: str,
    level: int = logging.INFO,
    log_entry: bool = True,
    log_exit: bool = True,
    log_time: bool = True,
    correlation_id: Optional[str] = None
) -> Iterator[Dict[str, Any]]:
    """Context manager for scoped logging.
    
    Creates a logging scope with entry/exit logs and timing information.
    Useful for tracking logical sections of code execution.
    
    Args:
        logger: Logger instance to use
        scope_name: Name of the scope for logging
        level: Logging level to use
        log_entry: Whether to log entry into the scope
        log_exit: Whether to log exit from the scope
        log_time: Whether to log the time spent in the scope
        correlation_id: Optional correlation ID for tracing
        
    Yields:
        Dict[str, Any]: Context dictionary with scope information
        
    Example:
        with log_scope(logger, "Processing file", correlation_id="abc123") as scope:
            process_file(filename)
            scope["file_size"] = os.path.getsize(filename)
    """
    # Add correlation ID to context if provided
    extra = {}
    if correlation_id:
        extra = {"correlation_id": correlation_id}
    
    # Apply context if needed
    if extra and not isinstance(logger, logging.LoggerAdapter):
        logger = logging.LoggerAdapter(logger, {"extra": extra})
    
    # Create context dictionary for the scope
    context = {
        "scope": scope_name,
        "start_time": time.time(),
        "attributes": {}
    }
    
    # Log scope entry
    if log_entry:
        logger.log(level, f"Entering scope: {scope_name}")
    
    try:
        # Yield context to caller
        yield context
        
    except Exception as e:
        # Calculate time in scope
        duration_sec = time.time() - context["start_time"]
        
        # Log scope exit with exception
        if log_exit:
            if log_time:
                logger.log(level, f"Exiting scope: {scope_name} with exception after {duration_sec:.2f}s: {type(e).__name__}: {e}")
            else:
                logger.log(level, f"Exiting scope: {scope_name} with exception: {type(e).__name__}: {e}")
                
        # Re-raise the exception
        raise
        
    else:
        # Calculate time in scope
        duration_sec = time.time() - context["start_time"]
        context["duration_sec"] = duration_sec
        
        # Log scope exit
        if log_exit:
            # Include any attributes added to the context
            attr_str = ""
            if context["attributes"]:
                attr_items = [f"{k}={v}" for k, v in context["attributes"].items()]
                attr_str = f" ({', '.join(attr_items)})"
                
            if log_time:
                logger.log(level, f"Exiting scope: {scope_name} after {duration_sec:.2f}s{attr_str}")
            else:
                logger.log(level, f"Exiting scope: {scope_name}{attr_str}")


class LoggingContext:
    """Context manager for temporary logging configuration.
    
    Allows for temporarily changing logging configuration within a scope,
    automatically restoring the previous configuration when exiting the scope.
    
    Attributes:
        logger: Logger instance to modify
        level: Logging level to set
        handler: Optional handler to add
        close_handler: Whether to close the handler when exiting the context
        formatter: Optional formatter to use with the handler
    """
    
    def __init__(
        self,
        logger: logging.Logger,
        level: Optional[int] = None,
        handler: Optional[logging.Handler] = None,
        close_handler: bool = True,
        formatter: Optional[logging.Formatter] = None,
        correlation_id: Optional[str] = None
    ):
        """Initialize the context manager.
        
        Args:
            logger: Logger instance to modify
            level: Logging level to set
            handler: Optional handler to add
            close_handler: Whether to close the handler when exiting the context
            formatter: Optional formatter to use with the handler
            correlation_id: Optional correlation ID for the logging context
        """
        self.logger = logger
        self.level = level
        self.handler = handler
        self.close_handler = close_handler
        self.formatter = formatter
        self.correlation_id = correlation_id
        
        # Save original configuration
        self._old_level = logger.level
        self._prev_handlers = list(logger.handlers)
        self._logger_adapter = None
    
    def __enter__(self) -> Union[logging.Logger, logging.LoggerAdapter]:
        """Enter the context, applying the new logging configuration.
        
        Returns:
            Union[logging.Logger, logging.LoggerAdapter]: The configured logger
        """
        # Set new level if specified
        if self.level is not None:
            self.logger.setLevel(self.level)
        
        # Add new handler if specified
        if self.handler is not None:
            # Set formatter if specified
            if self.formatter is not None:
                self.handler.setFormatter(self.formatter)
            
            self.logger.addHandler(self.handler)
        
        # Create logger adapter if correlation ID is provided
        if self.correlation_id is not None:
            self._logger_adapter = logging.LoggerAdapter(
                self.logger,
                {"extra": {"correlation_id": self.correlation_id}}
            )
            return self._logger_adapter
            
        return self.logger
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit the context, restoring the previous logging configuration."""
        # Remove any handlers added during the context
        if self.handler is not None:
            self.logger.removeHandler(self.handler)
            if self.close_handler:
                try:
                    self.handler.close()
                except Exception:
                    pass
        
        # Remove any other new handlers
        current_handlers = list(self.logger.handlers)
        for handler in current_handlers:
            if handler not in self._prev_handlers:
                self.logger.removeHandler(handler)
        
        # Restore original handlers
        for handler in self._prev_handlers:
            if handler not in self.logger.handlers:
                self.logger.addHandler(handler)
        
        # Restore original level
        self.logger.setLevel(self._old_level)


def log_call(
    logger: Optional[Union[logging.Logger, logging.LoggerAdapter]] = None,
    level: int = logging.DEBUG,
    log_args: bool = True,
    log_result: bool = True,
    log_duration: bool = True,
    include_self: bool = False,
    mask_arg_names: Optional[List[str]] = None,
    mask_result: bool = False,
    correlation_id_arg: Optional[str] = None
) -> Callable:
    """Decorator to log function calls with arguments and return values.
    
    Args:
        logger: Logger instance to use (if None, a logger will be created)
        level: Logging level to use
        log_args: Whether to log function arguments
        log_result: Whether to log function return value
        log_duration: Whether to log function execution duration
        include_self: Whether to include self argument for methods
        mask_arg_names: List of argument names to mask in logs
        mask_result: Whether to mask the return value in logs
        correlation_id_arg: Name of the argument containing correlation ID
        
    Returns:
        Callable: Decorator function
    
    Example:
        @log_call(logger, log_result=False)
        def process_data(user_id, data):
            # Function implementation
            return result
    """
    def decorator(func: Callable) -> Callable:
        # Get function name and create logger if needed
        func_name = func.__qualname__ if hasattr(func, '__qualname__') else func.__name__
        
        if logger is None:
            # Create logger with function module name
            module_name = func.__module__ if hasattr(func, '__module__') else 'unknown'
            local_logger = get_logger(f"{module_name}.{func_name}")
        else:
            local_logger = logger
            
        # Set up masking
        safe_mask_arg_names = mask_arg_names or ["password", "secret", "token", "key"]
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Extract correlation ID from arguments if specified
            correlation_id = None
            if correlation_id_arg and correlation_id_arg in kwargs:
                correlation_id = kwargs[correlation_id_arg]
            
            # Create logger adapter if needed for correlation ID
            call_logger = local_logger
            if correlation_id and not isinstance(local_logger, logging.LoggerAdapter):
                call_logger = logging.LoggerAdapter(
                    local_logger,
                    {"extra": {"correlation_id": correlation_id}}
                )
            
            # Format arguments for logging
            if log_args:
                # Process positional arguments
                if include_self or len(args) <= 0:
                    args_str = ", ".join(repr(a) for a in args)
                else:
                    # Skip self parameter for instance methods
                    args_str = ", ".join(repr(a) for a in args[1:])
                
                # Process keyword arguments with masking for sensitive keys
                kwargs_items = []
                for k, v in kwargs.items():
                    if any(sensitive in k.lower() for sensitive in safe_mask_arg_names):
                        # Mask sensitive values
                        mask_val = "********" if v else ""
                        kwargs_items.append(f"{k}={repr(mask_val)}")
                    else:
                        kwargs_items.append(f"{k}={repr(v)}")
                        
                kwargs_str = ", ".join(kwargs_items)
                
                # Combine args and kwargs
                if args_str and kwargs_str:
                    params_str = f"{args_str}, {kwargs_str}"
                elif args_str:
                    params_str = args_str
                elif kwargs_str:
                    params_str = kwargs_str
                else:
                    params_str = ""
                    
                call_logger.log(level, f"Calling {func_name}({params_str})")
            else:
                call_logger.log(level, f"Calling {func_name}")
            
            # Execute function and time it
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                elapsed_ms = (time.time() - start_time) * 1000
                
                # Log result
                if log_result:
                    result_repr = "********" if mask_result else repr(result)
                    if log_duration:
                        call_logger.log(level, f"{func_name} returned {result_repr} in {elapsed_ms:.2f}ms")
                    else:
                        call_logger.log(level, f"{func_name} returned {result_repr}")
                elif log_duration:
                    call_logger.log(level, f"{func_name} completed in {elapsed_ms:.2f}ms")
                    
                return result
                
            except Exception as e:
                elapsed_ms = (time.time() - start_time) * 1000
                call_logger.log(logging.ERROR, f"{func_name} raised {type(e).__name__}({str(e)}) in {elapsed_ms:.2f}ms")
                raise
                
        return wrapper
    
    return decorator


def configure_library_logging(
    libraries: List[str],
    level: int = logging.WARNING,
    handler: Optional[logging.Handler] = None,
    formatter: Optional[logging.Formatter] = None,
    propagate: bool = False
) -> None:
    """Configure logging for third-party libraries.
    
    Sets up logging configuration for external libraries to prevent
    them from being too verbose or to capture their logs for analysis.
    
    Args:
        libraries: List of library names to configure
        level: Logging level to set
        handler: Optional handler to add
        formatter: Optional formatter to use with the handler
        propagate: Whether logs should propagate to parent loggers
        
    Returns:
        None
    """
    for library in libraries:
        try:
            lib_logger = logging.getLogger(library)
            lib_logger.setLevel(level)
            lib_logger.propagate = propagate
            
            # Remove existing handlers
            for h in lib_logger.handlers[:]:
                h.close()
                lib_logger.removeHandler(h)
            
            # Add the new handler if provided
            if handler:
                if formatter:
                    handler.setFormatter(formatter)
                lib_logger.addHandler(handler)
                
            logging.debug(f"Configured logging for library: {library}")
        except Exception as e:
            logging.warning(f"Error configuring logging for library {library}: {e}")


def setup_json_logging(
    log_level: int = logging.INFO,
    log_file: Optional[str] = None,
    console_output: bool = True,
    include_hostname: bool = False,
    include_path: bool = False,
    include_thread: bool = False,
    include_process: bool = False,
    correlation_id: Optional[str] = None
) -> None:
    """Set up JSON-formatted logging.
    
    Configures the logging system to output logs in JSON format,
    which is useful for log processing and analysis tools.
    
    Args:
        log_level: The logging level to use
        log_file: Optional file path for logging to a file
        console_output: Whether to output logs to console
        include_hostname: Whether to include hostname in logs
        include_path: Whether to include file path in logs
        include_thread: Whether to include thread info in logs
        include_process: Whether to include process info in logs
        correlation_id: Optional correlation ID for log entries
        
    Returns:
        None
    """
    json_options = {
        "include_hostname": include_hostname,
        "include_path": include_path,
        "include_thread": include_thread,
        "include_process": include_process
    }
    
    setup_logging(
        log_level=log_level,
        log_file=log_file,
        console_output=console_output,
        json_format=True,
        json_options=json_options,
        correlation_id=correlation_id
    )
