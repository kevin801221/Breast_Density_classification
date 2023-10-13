# Import necessary libraries
import mlflow
from datetime import datetime
from torchvision.models import densenet169
import numpy as np
import os
import time
import copy
import torch
import torchvision
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from tqdm import tqdm
mlflow.set_tracking_uri("http://localhost:5001/")

EXPERIMENT_NAME = "Breast_density_with_SE-attention_balanced_data_0630"

experiment_id = mlflow.get_experiment_by_name(EXPERIMENT_NAME)  # check if the experiment is already exist
if not experiment_id:
    experiment_id = mlflow.create_experiment(EXPERIMENT_NAME)
else:
    experiment_id = experiment_id.experiment_id

model = densenet169()

mlflow.start_run(
    experiment_id=experiment_id,
    run_name=f'test_{datetime.now().strftime("%Y-%m-%d")}',
    tags={
        "type": "mammogran_type",
        "task": "pathology_BD_type"
    }
)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cuda:1")
#Define the SEBlock
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

# Load the pretrained DenseNet model and replace the last fully connected layer
model = models.densenet169(pretrained=True)
num_ftrs = model.classifier.in_features
model.classifier = nn.Linear(num_ftrs, 3)

# Add SEBlock to DenseNet
for name, module in model.named_modules():
    if isinstance(module, nn.Conv2d):
        setattr(module, "se", SEBlock(module.out_channels))

model = model.to(device)

# class SEBlock(nn.Module):
#     def __init__(self, channel, reduction=16):
#         super(SEBlock, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.fc = nn.Sequential(
#             nn.Linear(channel, channel // reduction, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Linear(channel // reduction, channel, bias=False),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         b, c, _, _ = x.size()
#         y = self.avg_pool(x).view(b, c)
#         y = self.fc(y).view(b, c, 1, 1)
#         return x * y.expand_as(x)


# class CustomDenseLayer(nn.Module):
#     def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
#         super(CustomDenseLayer, self).__init__()
#         self.add_module("norm1", nn.BatchNorm2d(num_input_features)),
#         self.add_module("relu1", nn.ReLU(inplace=True)),
#         self.add_module("conv1", nn.Conv2d(num_input_features, bn_size * growth_rate,
#                         kernel_size=1, stride=1, bias=False)),
#         self.add_module("norm2", nn.BatchNorm2d(bn_size * growth_rate)),
#         self.add_module("relu2", nn.ReLU(inplace=True)),
#         self.add_module("conv2", nn.Conv2d(bn_size * growth_rate, growth_rate,
#                         kernel_size=3, stride=1, padding=2, bias=False, dilation=2)),
#         self.drop_rate = drop_rate
#         self.se = SEBlock(growth_rate)

#     def forward(self, *prev_features):
#         concated_features = torch.cat(prev_features, 1)
#         bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))
#         new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
#         new_features = self.se(new_features)
#         if self.drop_rate > 0:
#             new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
#         return new_features

# Now replace dense layers in the DenseNet with your custom layer



# Set the parameters
data_dir = "/home/kevinluo/Benign_Malignant_dataclassifier"
num_epochs = 150
batch_size = 16
learning_rate = 0.001

# Set the transforms
data_transforms = {
    "train": transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    "valid": transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    "test": transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}
epochs = 150
batch_size = 16
lr = 0.001
mlflow.log_params(
    {
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr
    }
)
#train_model
from tqdm import tqdm

def train_model(model, criterion, optimizer, scheduler, num_epochs):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            # Log metrics to MLflow
            mlflow.log_metric("{}_loss".format(phase), epoch_loss, step=epoch)
            mlflow.log_metric("{}_accuracy".format(phase), epoch_acc.item(), step=epoch)

            # deep copy the model
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


#evaluate_model
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
    plt.savefig('confusion_matrix0630.png')
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
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Set the criterion and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# Train the model
model = train_model(model, criterion, optimizer, exp_lr_scheduler, num_epochs=num_epochs)

# Evaluate the model
evaluate_model(model, dataloaders["test"], class_names)

# Save the model
torch.save(model.state_dict(), "model_with_se_block0630.pth")