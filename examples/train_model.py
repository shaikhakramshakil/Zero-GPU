"""
Model training example using cloud connectors.

This example demonstrates:
- Training a neural network on cloud GPU
- Monitoring training progress
- Saving and downloading model
- Performance metrics
"""

from src.utils import get_connector
from dotenv import load_dotenv


def main():
    """Run model training example."""
    
    load_dotenv()
    
    print("=" * 60)
    print("Model Training on Cloud GPU")
    print("=" * 60)
    
    # Initialize connector (default: Colab)
    print("\n1. Initializing cloud connector...")
    cloud = get_connector(platform='colab', verbose=True)
    
    if not cloud.authenticate():
        print("❌ Authentication failed!")
        return
    print("✅ Connected to cloud platform!")
    
    # Training code
    print("\n2. Preparing training code...")
    training_code = """
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import time

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

# Define model
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialize
model = SimpleNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Training loop
print("\\nStarting training...")
start_time = time.time()

num_epochs = 3
for epoch in range(num_epochs):
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
        
        if (batch_idx + 1) % 100 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}, Loss: {loss.item():.4f}")
    
    epoch_loss = total_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    print(f"Epoch {epoch+1} - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")

training_time = time.time() - start_time
print(f"\\nTraining completed in {training_time:.2f} seconds")

# Save model
torch.save(model.state_dict(), 'model.pth')
print("Model saved to model.pth")
"""
    
    # Execute training
    print("\n3. Executing training on cloud GPU...")
    result = cloud.execute(training_code, timeout=600)
    
    if result.get('success'):
        print("✅ Training completed successfully!")
        print("\nTraining Output:")
        print(result.get('output', 'No output'))
    else:
        print(f"❌ Training failed: {result.get('error', 'Unknown error')}")
    
    # Download trained model
    print("\n4. Downloading trained model...")
    if cloud.download_file('model.pth', './model.pth'):
        print("✅ Model downloaded to ./model.pth")
    else:
        print("❌ Failed to download model")
    
    print("\n" + "=" * 60)
    print("Training example completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
