import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torch.optim import Adam

# 使用注意力機制
class AttentionModule(nn.Module):
    def __init__(self):
        super(AttentionModule, self).__init__()
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x):
        attention = self.softmax(x)
        out = attention * x
        return out

# 定義訓練用的transforms
transform = transforms.Compose([
    transforms.Resize((416, 416)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 加載數據
train_data = datasets.ImageFolder('/home/kevinluo/B_M_dataclassifier/train', transform=transform)
test_data = datasets.ImageFolder('/home/kevinluo/B_M_dataclassifier/test', transform=transform)

train_loader = DataLoader(train_data, batch_size=8, shuffle=True)
test_loader = DataLoader(test_data, batch_size=8, shuffle=False)

# 使用pretrained的ResNet50
model = models.resnet50(pretrained=True)

# 將ResNet的fc層替換為自定義的分類層
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 512),
    nn.ReLU(),
    AttentionModule(),
    nn.Linear(512, 2) # 改為二分類問題
)

# 定義損失函數和優化器
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001)

# 將模型移至GPU
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
model.to(device)

# 訓練模型
for epoch in range(10):  # 10個epochs，可以根據需要調整
    model.train()
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()

    # 每個epoch結束後，計算並輸出accuracy
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy after epoch {epoch + 1}: {correct / total * 100:.2f}%')

# 保存模型
torch.save(model.state_dict(), 'path_to_save_model')

