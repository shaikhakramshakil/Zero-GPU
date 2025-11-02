"""
Utility functions for cloud connector operations.
"""

import os
from typing import Dict, Any, Optional
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)


def load_env_variables() -> Dict[str, str]:
    """
    Load environment variables from .env file.
    
    Returns:
        Dictionary of environment variables
    """
    load_dotenv()
    
    env_vars = {
        'COLAB_AUTH_TOKEN': os.getenv('COLAB_AUTH_TOKEN'),
        'COLAB_PROJECT_ID': os.getenv('COLAB_PROJECT_ID'),
        'KAGGLE_USERNAME': os.getenv('KAGGLE_USERNAME'),
        'KAGGLE_KEY': os.getenv('KAGGLE_KEY'),
        'PREFERRED_PLATFORM': os.getenv('PREFERRED_PLATFORM', 'colab'),
        'TIMEOUT_SECONDS': int(os.getenv('TIMEOUT_SECONDS', '300')),
        'VERBOSE': os.getenv('VERBOSE', 'true').lower() == 'true'
    }
    
    return env_vars


def get_connector(platform: str = 'colab', **kwargs):
    """
    Factory function to create appropriate connector.
    
    Args:
        platform: 'colab' or 'kaggle'
        **kwargs: Additional arguments for connector
        
    Returns:
        CloudConnector instance
    """
    from .colab_connector import ColabConnector
    from .kaggle_connector import KaggleConnector
    
    platform = platform.lower()
    
    if platform == 'colab':
        return ColabConnector(**kwargs)
    elif platform == 'kaggle':
        return KaggleConnector(**kwargs)
    else:
        raise ValueError(f"Unknown platform: {platform}")


def measure_execution_time(func):
    """Decorator to measure function execution time."""
    import time
    from functools import wraps
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        execution_time = end_time - start_time
        logger.info(f"{func.__name__} executed in {execution_time:.2f} seconds")
        
        return result
    
    return wrapper


def validate_code(code: str) -> bool:
    """
    Basic validation of Python code.
    
    Args:
        code: Python code to validate
        
    Returns:
        True if code appears valid
    """
    try:
        compile(code, '<string>', 'exec')
        return True
    except SyntaxError as e:
        logger.error(f"Syntax error in code: {e}")
        return False


def format_code(code: str) -> str:
    """
    Format Python code (basic formatting).
    
    Args:
        code: Python code to format
        
    Returns:
        Formatted code
    """
    # Remove leading/trailing whitespace
    code = code.strip()
    
    # Normalize indentation
    lines = code.split('\n')
    
    return '\n'.join(lines)


def create_progress_bar(total: int, current: int) -> str:
    """
    Create a text-based progress bar.
    
    Args:
        total: Total items
        current: Current item
        
    Returns:
        Progress bar string
    """
    bar_length = 30
    percent = current / total
    filled = int(bar_length * percent)
    bar = '█' * filled + '-' * (bar_length - filled)
    
    return f"[{bar}] {percent*100:.1f}%"


def estimate_execution_time(code_size: int, complexity: str = 'medium') -> int:
    """
    Estimate code execution time based on size and complexity.
    
    Args:
        code_size: Size of code in characters
        complexity: 'low', 'medium', or 'high'
        
    Returns:
        Estimated time in seconds
    """
    base_time = code_size / 1000  # 1 second per 1000 chars
    
    complexity_multiplier = {
        'low': 1,
        'medium': 2,
        'high': 5
    }
    
    multiplier = complexity_multiplier.get(complexity, 2)
    
    return max(5, int(base_time * multiplier))
