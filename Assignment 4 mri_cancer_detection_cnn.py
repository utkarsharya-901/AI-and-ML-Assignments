import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os

# 1. Define CNN Architecture for Brain Tumor MRI Detection
class BrainTumorCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(BrainTumorCNN, self).__init__()
        # Input shape: (Batch, 3, 128, 128) - MRI scans resized
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.3)
        
        # After 3 pooling layers: 128x128 -> 64x64 -> 32x32 -> 16x16
        self.fc1 = nn.Linear(128 * 16 * 16, 256)
        self.fc2 = nn.Linear(256, num_classes) # Binary classification: Tumor (1) vs No Tumor (0)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        x = x.view(x.size(0), -1) # Flatten
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# Synthetic Dataset Generator for Testing / Demo Purposes
class DummyMRIDataset(Dataset):
    def __init__(self, num_samples=200, transform=None):
        self.num_samples = num_samples
        self.transform = transform
        # Random synthetic images mimicking 128x128 MRI scans
        self.data = np.random.randint(0, 256, (num_samples, 128, 128, 3), dtype=np.uint8)
        self.labels = np.random.randint(0, 2, num_samples)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        img = Image.fromarray(self.data[idx])
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

def main():
    print("1. Initializing Brain Tumor MRI Cancer Detection Project...")
    
    # 2. Data Augmentation and Transforms for Medical Scans
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print("2. Loading MRI Dataset...")
    # Replace with torchvision.datasets.ImageFolder('path/to/mri_dataset', transform=transform) for real data
    dataset = DummyMRIDataset(num_samples=300, transform=transform)
    
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # 3. Model, Loss, and Optimizer Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BrainTumorCNN(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    print(f"\n3. Training CNN Model on {device}...")
    epochs = 5
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = (correct / total) * 100
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {epoch_loss:.4f} - Accuracy: {epoch_acc:.2f}%")

    print("\n4. Evaluating Model on Test Set...")
    model.eval()
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()

    print(f"Test Accuracy: {(test_correct / test_total) * 100:.2f}%")

if __name__ == '__main__':
    main()
