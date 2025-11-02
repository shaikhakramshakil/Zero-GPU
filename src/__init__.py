"""
AI on Cloud - Run AI models on cloud notebooks from your local IDE.
"""

from .cloud_connector import CloudConnector
from .colab_connector import ColabConnector
from .kaggle_connector import KaggleConnector
from .utils import get_connector, load_env_variables

__version__ = "1.0.0"
__author__ = "Your Name"
__all__ = [
    'CloudConnector',
    'ColabConnector',
    'KaggleConnector',
    'get_connector',
    'load_env_variables'
]
