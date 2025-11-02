"""
Data processing example using cloud compute.

This example demonstrates:
- Large-scale data processing on cloud
- Data transformation and cleaning
- Parallel processing
- Results download
"""

from src.utils import get_connector
from dotenv import load_dotenv


def main():
    """Run data processing example."""
    
    load_dotenv()
    
    print("=" * 60)
    print("Large-scale Data Processing on Cloud")
    print("=" * 60)
    
    # Initialize connector
    print("\n1. Initializing cloud connector...")
    cloud = get_connector(platform='kaggle', verbose=True)
    
    if not cloud.authenticate():
        print("❌ Authentication failed!")
        return
    print("✅ Connected to cloud platform!")
    
    # Upload data
    print("\n2. Uploading data to cloud...")
    local_data = "./sample_data.csv"
    if not cloud.upload_file(local_data, "input_data.csv"):
        print("⚠️  Could not upload data (this is expected in demo mode)")
    
    # Processing code
    print("\n3. Preparing data processing code...")
    processing_code = """
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import time

print("Loading data...")
# In real scenario, load from uploaded file
# df = pd.read_csv('input_data.csv')

# Create sample dataset for demonstration
np.random.seed(42)
df = pd.DataFrame({
    'feature_1': np.random.randn(10000),
    'feature_2': np.random.randn(10000),
    'feature_3': np.random.randn(10000),
    'target': np.random.randint(0, 2, 10000)
})

print(f"Dataset shape: {df.shape}")
print(f"\\nFirst few rows:\\n{df.head()}")

# Data cleaning
print("\\nCleaning data...")
df = df.dropna()
df = df[df['feature_1'] != df['feature_1'].max()]  # Remove outliers

# Feature scaling
print("Scaling features...")
scaler = StandardScaler()
features = ['feature_1', 'feature_2', 'feature_3']
df[features] = scaler.fit_transform(df[features])

# Statistics
print("\\nData Statistics:")
print(df.describe())

# Class distribution
print("\\nClass distribution:")
print(df['target'].value_counts())

# Save processed data
print("\\nSaving processed data...")
df.to_csv('processed_data.csv', index=False)
print(f"Processed data saved: {len(df)} rows")
"""
    
    # Execute processing
    print("\n4. Executing data processing on cloud...")
    result = cloud.execute(processing_code, timeout=300)
    
    if result.get('success'):
        print("✅ Data processing completed successfully!")
        print("\nProcessing Output:")
        print(result.get('output', 'No output'))
    else:
        print(f"❌ Processing failed: {result.get('error', 'Unknown error')}")
    
    # Download results
    print("\n5. Downloading processed data...")
    if cloud.download_file('processed_data.csv', './processed_data.csv'):
        print("✅ Processed data downloaded to ./processed_data.csv")
    else:
        print("⚠️  Could not download results (this is expected in demo mode)")
    
    print("\n" + "=" * 60)
    print("Data processing example completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
