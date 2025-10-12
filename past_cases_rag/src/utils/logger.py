"""
Logging utilities for the Legal Assistant RAG system.
"""

import logging
import sys
from pathlib import Path
from rich.console import Console
from rich.logging import RichHandler

console = Console()

def setup_logging(level: str = "INFO", log_file: str = None) -> logging.Logger:
    """
    Set up logging with rich formatting.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional log file path
    
    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger("legal_rag")
    logger.setLevel(getattr(logging, level.upper()))
    
    # Remove existing handlers
    logger.handlers.clear()
    
    # Create rich handler for console output
    rich_handler = RichHandler(
        console=console,
        rich_tracebacks=True,
        show_time=True,
        show_path=False
    )
    rich_handler.setLevel(getattr(logging, level.upper()))
    
    # Create formatter
    formatter = logging.Formatter(
        "%(message)s"
    )
    rich_handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(rich_handler)
    
    # Add file handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, level.upper()))
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger

def get_logger(name: str = None) -> logging.Logger:
    """Get a logger instance."""
    return logging.getLogger(name or "legal_rag")