"""
Basic example of using Kaggle connector from local IDE.

This example demonstrates:
- Connecting to Kaggle
- Executing Python code remotely
- Retrieving GPU information
- Listing notebooks
"""

from src.kaggle_connector import KaggleConnector
import os
from dotenv import load_dotenv


def main():
    """Run basic Kaggle example."""
    
    # Load environment variables
    load_dotenv()
    
    print("=" * 60)
    print("Kaggle Connector - Basic Example")
    print("=" * 60)
    
    # Initialize Kaggle connector
    print("\n1. Initializing Kaggle Connector...")
    kaggle = KaggleConnector(verbose=True)
    
    # Authenticate
    print("\n2. Authenticating with Kaggle...")
    if not kaggle.authenticate():
        print("❌ Authentication failed!")
        print("   Make sure kaggle.json is in ~/.kaggle/ directory")
        return
    print("✅ Authentication successful!")
    
    # List notebooks
    print("\n3. Listing your Kaggle notebooks...")
    notebooks = kaggle.list_notebooks()
    if notebooks.get('success'):
        print(f"   Found {len(notebooks.get('notebooks', []))} notebooks")
    else:
        print(f"   Could not list notebooks: {notebooks.get('error')}")
    
    # Get GPU information
    print("\n4. Checking GPU availability...")
    gpu_info = kaggle.get_gpu_info()
    print(f"   GPU Available: {gpu_info.get('gpu_available', 'Unknown')}")
    print(f"   GPU Model: {gpu_info.get('gpu_name', 'Unknown')}")
    
    # Execute simple code
    print("\n5. Executing sample code on Kaggle...")
    code = """
import pandas as pd
import numpy as np

# Create sample dataframe
df = pd.DataFrame({
    'A': np.random.randn(5),
    'B': np.random.randn(5),
    'C': np.random.randn(5)
})

print("Sample DataFrame:")
print(df)
print(f"\\nDataFrame shape: {df.shape}")
print(f"Mean values:\\n{df.mean()}")
"""
    
    result = kaggle.execute(code)
    
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
