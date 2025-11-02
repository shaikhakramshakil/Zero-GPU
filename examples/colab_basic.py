"""
Basic example of using Colab connector from local IDE.

This example demonstrates:
- Connecting to Google Colab
- Executing Python code remotely
- Retrieving GPU information
- Downloading results
"""

from src.colab_connector import ColabConnector
import os
from dotenv import load_dotenv


def main():
    """Run basic Colab example."""
    
    # Load environment variables
    load_dotenv()
    
    print("=" * 60)
    print("Google Colab Connector - Basic Example")
    print("=" * 60)
    
    # Initialize Colab connector
    print("\n1. Initializing Colab Connector...")
    colab = ColabConnector(verbose=True)
    
    # Authenticate
    print("\n2. Authenticating with Google Colab...")
    if not colab.authenticate():
        print("❌ Authentication failed!")
        return
    print("✅ Authentication successful!")
    
    # Get GPU information
    print("\n3. Checking GPU availability...")
    gpu_info = colab.get_gpu_info()
    print(f"   GPU Available: {gpu_info.get('gpu_available', 'Unknown')}")
    print(f"   GPU Model: {gpu_info.get('gpu_name', 'Unknown')}")
    print(f"   GPU Memory: {gpu_info.get('gpu_memory', 'Unknown')} bytes")
    
    # Execute simple code
    print("\n4. Executing sample code on Colab...")
    code = """
import numpy as np
import torch

# NumPy operations
arr = np.array([1, 2, 3, 4, 5])
print(f"Array sum: {np.sum(arr)}")
print(f"Array mean: {np.mean(arr)}")

# PyTorch check
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
"""
    
    result = colab.execute(code)
    
    if result.get('success'):
        print("✅ Code executed successfully!")
        print("\nOutput:")
        print(result.get('output', 'No output'))
    else:
        print(f"❌ Execution failed: {result.get('error', 'Unknown error')}")
    
    print("\n" + "=" * 60)
    print("Example completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
