"""
Advanced usage examples for cloud connectors.

This example demonstrates:
- Multi-platform execution
- Conditional execution based on platform capabilities
- Error handling and recovery
- Performance optimization
"""

from src.colab_connector import ColabConnector
from src.kaggle_connector import KaggleConnector
from src.utils import load_env_variables, measure_execution_time
import time


@measure_execution_time
def train_on_best_platform():
    """
    Train model on the best available platform.
    Falls back if primary platform fails.
    """
    
    print("=" * 60)
    print("Advanced: Multi-Platform Training")
    print("=" * 60)
    
    env_vars = load_env_variables()
    preferred_platform = env_vars.get('PREFERRED_PLATFORM', 'colab')
    
    # Try preferred platform first
    connectors = []
    if preferred_platform == 'colab':
        connectors = [
            ('Colab', ColabConnector(verbose=True)),
            ('Kaggle', KaggleConnector(verbose=True))
        ]
    else:
        connectors = [
            ('Kaggle', KaggleConnector(verbose=True)),
            ('Colab', ColabConnector(verbose=True))
        ]
    
    # Try each platform until one works
    for platform_name, connector in connectors:
        print(f"\n1. Attempting to connect to {platform_name}...")
        
        try:
            if connector.authenticate():
                print(f"✅ Successfully connected to {platform_name}!")
                
                # Execute training
                print(f"\n2. Training on {platform_name}...")
                result = run_training(connector)
                
                if result.get('success'):
                    print(f"✅ Training successful on {platform_name}!")
                    return result
                else:
                    print(f"❌ Training failed on {platform_name}: {result.get('error')}")
            else:
                print(f"❌ Could not authenticate with {platform_name}")
        
        except Exception as e:
            print(f"❌ Error with {platform_name}: {str(e)}")
    
    print("\n❌ Failed to train on any platform")
    return {'success': False, 'error': 'All platforms failed'}


def run_training(connector):
    """Run training on the given connector."""
    
    training_code = """
import torch
import time

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    device = torch.device('cpu')
    print("Using CPU")

# Simple computation
start = time.time()
x = torch.randn(1000, 1000, device=device)
y = torch.randn(1000, 1000, device=device)
z = torch.matmul(x, y)
elapsed = time.time() - start

print(f"Matrix multiplication completed in {elapsed:.4f} seconds")
"""
    
    return connector.execute(training_code, timeout=120)


@measure_execution_time
def batch_processing_with_retry(cloud, code: str, max_retries: int = 3):
    """
    Execute code with automatic retry on failure.
    
    Args:
        cloud: Cloud connector
        code: Code to execute
        max_retries: Maximum number of retries
        
    Returns:
        Result dictionary
    """
    
    for attempt in range(1, max_retries + 1):
        print(f"\nAttempt {attempt}/{max_retries}...")
        
        try:
            result = cloud.execute(code, timeout=300)
            
            if result.get('success'):
                print(f"✅ Success on attempt {attempt}")
                return result
            else:
                print(f"❌ Failed on attempt {attempt}: {result.get('error')}")
        
        except Exception as e:
            print(f"❌ Exception on attempt {attempt}: {str(e)}")
        
        if attempt < max_retries:
            wait_time = 2 ** attempt  # Exponential backoff
            print(f"Waiting {wait_time} seconds before retry...")
            time.sleep(wait_time)
    
    return {'success': False, 'error': 'Max retries exceeded'}


def monitor_execution(cloud, code: str, check_interval: int = 5):
    """
    Execute code and monitor progress with periodic checks.
    
    Args:
        cloud: Cloud connector
        code: Code to execute
        check_interval: Interval in seconds between checks
        
    Returns:
        Result dictionary
    """
    
    print("Starting execution with monitoring...")
    start_time = time.time()
    
    result = cloud.execute(code, timeout=300)
    
    elapsed_time = time.time() - start_time
    
    if result.get('success'):
        print(f"✅ Execution completed in {elapsed_time:.2f} seconds")
    else:
        print(f"❌ Execution failed: {result.get('error')}")
    
    return result


def main():
    """Run advanced examples."""
    
    print("\n" + "=" * 60)
    print("Advanced Usage Examples")
    print("=" * 60)
    
    # Example 1: Multi-platform training
    print("\nExample 1: Multi-Platform Training")
    result = train_on_best_platform()
    if result.get('success'):
        print("✅ Multi-platform training successful!")
    
    # Example 2: Retry logic
    print("\n\nExample 2: Batch Processing with Retry")
    env_vars = load_env_variables()
    cloud = KaggleConnector(verbose=True)
    
    if cloud.authenticate():
        code = """
import numpy as np
print("Processing batch...")
data = np.random.randn(1000, 100)
result = np.mean(data, axis=0)
print(f"Processed {len(result)} features")
"""
        result = batch_processing_with_retry(cloud, code, max_retries=2)
    
    print("\n" + "=" * 60)
    print("Advanced examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
