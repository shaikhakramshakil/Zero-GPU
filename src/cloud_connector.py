"""
Base connector class for cloud notebook services.
This module provides the abstract interface for connecting to cloud compute platforms.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CloudConnector(ABC):
    """Abstract base class for cloud notebook connectors."""
    
    def __init__(self, timeout: int = 300, verbose: bool = True):
        """
        Initialize the cloud connector.
        
        Args:
            timeout: Timeout in seconds for cloud operations
            verbose: Enable verbose logging
        """
        self.timeout = timeout
        self.verbose = verbose
        self.authenticated = False
    
    @abstractmethod
    def authenticate(self) -> bool:
        """
        Authenticate with the cloud service.
        
        Returns:
            True if authentication was successful, False otherwise
        """
        pass
    
    @abstractmethod
    def execute(self, code: str, **kwargs) -> Dict[str, Any]:
        """
        Execute code on the cloud platform.
        
        Args:
            code: Python code to execute
            **kwargs: Additional arguments
            
        Returns:
            Dictionary with execution results
        """
        pass
    
    @abstractmethod
    def upload_file(self, local_path: str, remote_path: str) -> bool:
        """
        Upload a file to the cloud platform.
        
        Args:
            local_path: Local file path
            remote_path: Remote file path
            
        Returns:
            True if upload was successful
        """
        pass
    
    @abstractmethod
    def download_file(self, remote_path: str, local_path: str) -> bool:
        """
        Download a file from the cloud platform.
        
        Args:
            remote_path: Remote file path
            local_path: Local file path
            
        Returns:
            True if download was successful
        """
        pass
    
    def _log(self, message: str, level: str = "info"):
        """Internal logging method."""
        if self.verbose:
            if level == "info":
                logger.info(message)
            elif level == "error":
                logger.error(message)
            elif level == "warning":
                logger.warning(message)
