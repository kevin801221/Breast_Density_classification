import os
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, mean_squared_error, mean_absolute_error, r2_score
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

class BreastDensityDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths, self.labels = self.load_data()

    def load_data(self):
        image_paths, labels = [], []
        for i in range(1, 5):
            class_dir = os.path.join(self.root_dir, f"breast_density{i}")
            for img_name in os.listdir(class_dir):
                image_paths.append(os.path.join(class_dir, img_name))
                labels.append(i-1)
        return image_paths, labels

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = plt.imread(self.image_paths[idx])
        label = self.labels[idx] / 3
        if self.transform:
            image = self.transform(image)
        return image, label

import torch
import torch.nn as nn
from torchvision import models

def create_resnet50_regression_model():
    model = models.resnet50(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 1)
    return model


from torchvision import transforms
from torch.utils.data import DataLoader
data_dir = "/home/kevinluo/breast_density_classification/datasets/"
input_size = (640, 640)
batch_size = 32


def create_dataloaders(data_dir, input_size=(640, 640), batch_size=32):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = BreastDensityDataset(os.path.join(data_dir, "train"), transform=transform)
    valid_dataset = BreastDensityDataset(os.path.join(data_dir, "valid"), transform=transform)
    test_dataset = BreastDensityDataset(os.path.join(data_dir, "test"), transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return {"train": train_loader, "valid": valid_loader, "test": test_loader}
dataloaders = create_dataloaders(data_dir, input_size, batch_size)

class RegressionModel(nn.Module):
    def __init__(self, num_classes=1):
        super(RegressionModel, self).__init__()
        self.base_model = models.resnet50(pretrained=True)
        num_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.base_model(x)

def train_model(model, criterion, optimizer, dataloaders, num_epochs=25):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_losses, valid_losses = [], []
    train_accuracies, valid_accuracies = [], []
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        print("-" * 10)

        for phase in ["train", "valid"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss, running_corrects = 0.0, 0
            data_loader = dataloaders[phase]
            with tqdm(data_loader, unit="batch") as tepoch:
                for inputs, labels in tepoch:
                    tepoch.set_description(f"{phase.capitalize()} Epoch {epoch+1}")
                    inputs, labels = inputs.to(device), labels.to(device)

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == "train"):
                        outputs = model(inputs)
                        loss = criterion(outputs.view(-1), labels.float())

                        if phase == "train":
                            loss.backward()
                            optimizer.step()

                        preds = (outputs.view(-1) * 3).round()
                        running_corrects += torch.sum(preds == labels.view(-1) * 3).item()

                    running_loss += loss.item() * inputs.size(0)

                epoch_loss = running_loss / len(data_loader.dataset)
                epoch_acc = running_corrects / len(data_loader.dataset)

                if phase == "train":
                    train_losses.append(epoch_loss)
                    train_accuracies.append(epoch_acc)
                else:
                    valid_losses.append(epoch_loss)
                    valid_accuracies.append(epoch_acc)

                print(f"{phase.capitalize()} Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")

    return model, train_losses, valid_losses, train_accuracies, valid_accuracies

def plot_metrics(train_losses, valid_losses, train_accuracies, valid_accuracies):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    ax1.plot(train_losses, label="Training Loss")
    ax1.plot(valid_losses, label="Validation Loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend()

    ax2.plot(train_accuracies, label="Training Accuracy")
    ax2.plot(valid_accuracies, label="Validation Accuracy")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Accuracy")
    ax2.legend()

    plt.show()

def evaluate_model(model, test_loader):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    y_true, y_pred = [], []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = (outputs.view(-1) * 3).round()

            y_true.extend(labels.cpu().numpy() * 3)
            y_pred.extend(preds.cpu().numpy())

    return y_true, y_pred


def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu", cbar=False)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

def print_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Mean Absolute Error: {mae:.4f}")
    print(f"R2 Score: {r2:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))

def main():
    data_dir = "/home/kevinluo/breast_density_classification/datasets/"
    input_size = 640
    batch_size = 16
    num_epochs = 1
    learning_rate = 0.001

    dataloaders = create_dataloaders(data_dir, input_size, batch_size)
    model = create_resnet50_regression_model()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    trained_model, train_losses, valid_losses, train_accuracies, valid_accuracies = train_model(model, criterion, optimizer, dataloaders, num_epochs)
    # 評估模型
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for inputs, labels in dataloaders['test']:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    # 計算指標
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:\n", cm)

    cr = classification_report(y_true, y_pred, target_names=["breast_density1", "breast_density2", "breast_density3", "breast_density4"])
    print("Classification Report:\n", cr)

    acc = accuracy_score(y_true, y_pred)
    print("Accuracy:", acc)

    plot_metrics(train_losses, valid_losses, train_accuracies, valid_accuracies)

    test_loader = dataloaders["test"]
    y_true, y_pred = evaluate_model(trained_model, test_loader)

    plot_confusion_matrix(y_true, y_pred)

    print_metrics(y_true, y_pred)

if __name__ == "__main__":
    main()


