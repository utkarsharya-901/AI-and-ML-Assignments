import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import numpy as np

# 1. Define CNN Architecture for LFW Face Recognition
class FaceRecognitionCNN(nn.Module):
    def __init__(self, num_classes):
        super(FaceRecognitionCNN, self).__init__()
        # Input shape: (Batch, 1, 62, 47) - Grayscale images of LFW
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.4)
        
        # After 3 pooling layers: 62x47 -> 31x23 -> 15x11 -> 7x5
        self.fc1 = nn.Linear(128 * 7 * 5, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        x = x.view(x.size(0), -1) # Flatten
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

def main():
    print("1. Loading Labeled Faces in the Wild (LFW) Dataset...")
    # Fetch LFW dataset filtering for individuals with at least 70 images
    lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
    
    n_samples, h, w = lfw_people.images.shape
    X = lfw_people.images
    y = lfw_people.target
    target_names = lfw_people.target_names
    num_classes = len(target_names)

    print(f"Dataset summary:")
    print(f" - Total samples: {n_samples}")
    print(f" - Image resolution: {h}x{w}")
    print(f" - Number of classes (people): {num_classes}")
    for i, name in enumerate(target_names):
        print(f"   Class {i}: {name}")

    # 2. Preprocessing & Normalization
    # Reshape X to (N, C, H, W) for PyTorch CNN (Grayscale: C=1)
    X = X.reshape((n_samples, 1, h, w))
    # Normalize pixel values to [0, 1]
    X = X / 255.0

    # Train-test split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Convert to PyTorch Tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.long)

    # DataLoader setup
    train_dataset = TensorDataset(X_train_t, y_train_t)
    test_dataset = TensorDataset(X_test_t, y_test_t)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 3. Model, Loss, and Optimizer Initialization
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FaceRecognitionCNN(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print(f"\n2. Training CNN Model on {device}...")
    epochs = 25
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            
        epoch_loss = running_loss / len(train_loader.dataset)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{epochs}] - Loss: {epoch_loss:.4f}")

    print("\n3. Evaluating Model...")
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    print(f"\nAccuracy: {acc * 100:.2f}%\n")
    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=target_names))

if __name__ == '__main__':
    main()
