import numpy as np 
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils import data
from tqdm import tqdm
import torch.backends.cudnn as cudnn
from torch.cuda.amp import autocast

# Enable CUDA optimizations
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

# Verify CUDA is being used
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
print(f'GPU Device: {torch.cuda.get_device_name(0)}')

def print_gpu_memory():
    print(f"GPU Memory Usage:")
    print(f"Allocated: {torch.cuda.memory_allocated() / 1024**2:.1f}MB")
    print(f"Cached: {torch.cuda.memory_reserved() / 1024**2:.1f}MB")

class ComplexNet(nn.Module):
    def __init__(self):
        super(ComplexNet, self).__init__()
        self.flatten = nn.Flatten()
        self.features = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 10)
        )
    
    def forward(self, x):
        x = self.flatten(x)
        return self.features(x)

model = ComplexNet().to(device)
print_gpu_memory()

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def main():
    
    BATCH_SIZE = 1024
    NUM_EPOCHS = 15

    normalize = transforms.Normalize(mean=[.5], std=[.5])
    transform = transforms.Compose([transforms.ToTensor(), normalize])

    # download and load the data
    train_dataset = torchvision.datasets.MNIST(root='./mnist/', train=True, transform=transform, download=True)
    test_dataset = torchvision.datasets.MNIST(root='./mnist/', train=False, transform=transform, download=False)

# encapsulate them into dataloader form
    train_loader = data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True
    )
    test_loader = data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)

    # Add Mixed Precision training (significant speedup on RTX cards)
    from torch.amp import autocast, GradScaler
    scaler = GradScaler('cuda')

    # Print model device
    print(f"Model device: {next(model.parameters()).device}")

    # For checking data device, do it inside your training loop
    for batch_data, batch_labels in train_loader:  # or whatever your loader is called
        batch_data = batch_data.to(device)
        batch_labels = batch_labels.to(device)
        print(f"Batch data device: {batch_data.device}")
        print(f"Batch labels device: {batch_labels.device}")
        break  # Just check the first batch and break

    # Initialize lists to store metrics
    train_accuracies = []
    test_accuracies = []
    
    # Training loop
    num_epochs = NUM_EPOCHS
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        correct_train = 0
        total_train = 0
        
        for batch_data, batch_labels in train_loader:
            batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_data)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            
            _, predicted = torch.max(outputs.data, 1)
            total_train += batch_labels.size(0)
            correct_train += (predicted == batch_labels).sum().item()
        
        train_accuracy = 100 * correct_train / total_train
        train_accuracies.append(train_accuracy)
        
        # Testing phase
        model.eval()
        correct_test = 0
        total_test = 0
        
        with torch.no_grad():
            for batch_data, batch_labels in test_loader:
                batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
                outputs = model(batch_data)
                _, predicted = torch.max(outputs.data, 1)
                total_test += batch_labels.size(0)
                correct_test += (predicted == batch_labels).sum().item()
        
        test_accuracy = 100 * correct_test / total_test
        test_accuracies.append(test_accuracy)
        
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Train Accuracy: {train_accuracy:.2f}%')
        print(f'Test Accuracy: {test_accuracy:.2f}%')
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs + 1), train_accuracies, label='Training Accuracy')
    plt.plot(range(1, num_epochs + 1), test_accuracies, label='Testing Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (log scale)')
    plt.yscale('log')
    plt.title('Training and Testing Accuracy over Epochs')
    plt.legend()
    plt.show()

if __name__ == "__main__":main()

