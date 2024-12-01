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
from collections import defaultdict

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

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.flatten = nn.Flatten()
        self.features = nn.Sequential(
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
    
    def forward(self, x):
        x = self.flatten(x)
        return self.features(x)

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

class DeepNet(nn.Module):
    def __init__(self):
        super(DeepNet, self).__init__()
        self.flatten = nn.Flatten()
        self.features = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 10)
        )
    
    def forward(self, x):
        x = self.flatten(x)
        return self.features(x)

def train_model(model, train_loader, test_loader, device, num_epochs):
    # Move model to device
    model = model.to(device)
    
    # Initialize optimizer and criterion for this model
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    train_accuracies = []
    test_accuracies = []
    
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
    
    return train_accuracies, test_accuracies

def visualize_network_structure(model, ax, title):
    """Visualize network structure as a simple layer diagram"""
    layers = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.BatchNorm1d, nn.ReLU, nn.Dropout)):
            layers.append((type(module).__name__, module))
    
    y_spacing = 1
    y_positions = np.arange(len(layers)) * y_spacing
    
    # Draw boxes for each layer
    for idx, (layer_name, layer) in enumerate(layers):
        if isinstance(layer, nn.Linear):
            label = f'{layer_name}\n{layer.in_features}→{layer.out_features}'
        elif isinstance(layer, nn.Dropout):
            label = f'{layer_name}\np={layer.p}'
        else:
            label = layer_name
            
        ax.add_patch(plt.Rectangle((-0.5, y_positions[idx]-0.3), 1, 0.6, 
                                 fill=True, alpha=0.3))
        ax.text(0, y_positions[idx], label, ha='center', va='center',fontsize='x-small')
    
    # Set plot properties
    ax.set_title(title)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, max(y_positions) + 1)
    ax.axis('off')

def main():
    BATCH_SIZE = 1024
    NUM_EPOCHS = 5
    
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

    # For checking data device, do it inside your training loop
    for batch_data, batch_labels in train_loader:  # or whatever your loader is called
        batch_data = batch_data.to(device)
        batch_labels = batch_labels.to(device)
        print(f"Batch data device: {batch_data.device}")
        print(f"Batch labels device: {batch_labels.device}")
        break  # Just check the first batch and break

    # Define models to compare
    models = {
        'Simple': SimpleNet(),
        'Complex': ComplexNet(),
        'Deep': DeepNet()
    }
    
    # Dictionary to store results
    results = defaultdict(dict)
    
    # Train each model
    for model_name, model in models.items():
        print(f"\nTraining {model_name} Network...")
        train_acc, test_acc = train_model(model, train_loader, test_loader, device, NUM_EPOCHS)
        results[model_name]['train'] = train_acc
        results[model_name]['test'] = test_acc
    
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 10), tight_layout=True)
    
    # Performance plot (top)
    ax1 = fig.add_subplot(2, 1, 1)
    for model_name in results:
        ax1.plot(range(1, NUM_EPOCHS + 1), results[model_name]['train'], 
                label=f'{model_name} (Train)', linestyle='--')
        ax1.plot(range(1, NUM_EPOCHS + 1), results[model_name]['test'], 
                label=f'{model_name} (Test)')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Model Accuracy Comparison')
    ax1.legend()

    # Network structure plots (bottom)
    for idx, (model_name, model) in enumerate(models.items()):
        ax = fig.add_subplot(2, 3, idx + 4)
        visualize_network_structure(model, ax, model_name)

    plt.show()

if __name__ == "__main__":
    main()

