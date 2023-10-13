from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import torch
import os
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.optim import lr_scheduler
import time
import copy
import tqdm
data_dir = '/home/kevinluo/breast_density_classification/BD_data_newdis'
batch_size = 16
learning_rate = 0.001
num_epochs = 100
# Define the data transformations
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(416),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize(416),
        transforms.CenterCrop(416),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(416),
        transforms.CenterCrop(416),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

# Load the datasets
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'valid', 'test']}

# Create data loaders
dataloaders = {x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train', 'valid', 'test']}

from torch import nn
from torchvision.models import densenet121
import torch.nn.functional as F

class SEBlock(nn.Module):
    def __init__(self, in_channels, r=16):
        super(SEBlock, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(in_channels, in_channels // r),
            nn.ReLU(),
            nn.Linear(in_channels // r, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
class CustomModel(nn.Module):
    def __init__(self, num_classes=4):
        super(CustomModel, self).__init__()
        base_model = densenet121(pretrained=True)
        self.base_features = base_model.features
        num_planes = 1024
        for name, module in self.base_features.named_modules():
            if isinstance(module, nn.Conv2d):
                module.dilation = (2, 2)
            if isinstance(module, nn.BatchNorm2d):
                module.momentum = 0.9
            if isinstance(module, nn.ReLU):
                module.inplace = True
        self.se_block = SEBlock(num_planes)
        self.classifier = nn.Linear(num_planes, num_classes)

    def forward(self, x):
        features = self.base_features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = self.se_block(out)
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out
device = torch.device("cuda:0" if torch.cuda.is_available() else "cuda:1")
model = CustomModel()
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

def train_model(model, dataloaders, criterion, optimizer, num_epochs=25):
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'valid':
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    model.load_state_dict(best_model_wts)
    return model, val_acc_history
def evaluate_model(model, dataloader, class_names):
    model.eval()  # set model to evaluation mode

    # Initialize the prediction and label lists
    predlist = torch.zeros(0, dtype=torch.long, device='cpu')
    lbllist = torch.zeros(0, dtype=torch.long, device='cpu')

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)  # forward pass
            _, preds = torch.max(outputs, 1)

            # Append batch prediction results
            predlist = torch.cat([predlist, preds.view(-1).cpu()])
            lbllist = torch.cat([lbllist, labels.view(-1).cpu()])

    # Confusion matrix
    conf_mat = confusion_matrix(lbllist.numpy(), predlist.numpy())
    plt.figure(figsize=(10, 10))
    sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig('confusion_matrix.png')
    plt.show()

    # Classification report
    class_report = classification_report(lbllist.numpy(), predlist.numpy(), target_names=class_names)
    print(class_report)



# Load the datasets
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ["train", "valid", "test"]}

# Create data loaders
dataloaders = {x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ["train", "valid", "test"]}

# Get the dataset sizes
dataset_sizes = {x: len(image_datasets[x]) for x in ["train", "valid", "test"]}

# Get the class names
class_names = image_datasets["train"].classes

# Set device to GPU or CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cuda:1")

# Set the criterion and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# Train the model
model, val_acc_history = train_model(model, dataloaders, criterion, optimizer, num_epochs=num_epochs)

# Evaluate the model
evaluate_model(model, dataloaders["test"], class_names)

# Save the model
torch.save(model.state_dict(), "0606model_with_se_block.pth")
