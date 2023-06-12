import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Define the ResNet-34 model
class ResNet34(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet34, self).__init__()
        self.model = torchvision.models.resnet34(pretrained=False)
        num_filters = self.model.fc.in_features
        self.model.fc = nn.Linear(num_filters, num_classes)

    def forward(self, x):
        return self.model(x)


# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
batch_size = 128

# Load CIFAR-10 dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Initialize the model
model = ResNet34().to(device)

# Load pre-trained weights
model.load_state_dict(torch.load("resnet34_cifar10.pth"))
model.eval()

# Extract features from the model
features = []
labels = []

# Iterate over the training dataset to collect features and labels
for images, target in train_loader:
    images = images.to(device)
    features.append(model.model.conv1(images).detach().cpu().numpy())
    labels.append(target.numpy())

# Convert the collected features and labels into tensors
features = torch.cat(features, dim=0)
labels = torch.cat(labels, dim=0)

# Prepare the attack dataset
attack_dataset = torch.utils.data.TensorDataset(features, labels)
attack_loader = torch.utils.data.DataLoader(attack_dataset, batch_size=batch_size, shuffle=True)

# Perform membership inference attack
attack_model = nn.Linear(model.model.conv1.out_channels, 2).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(attack_model.parameters(), lr=0.001)

# Training loop for the attack model
num_epochs = 10
for epoch in range(num_epochs):
    for features, labels in attack_loader:
        features = features.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = attack_model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# Testing the attack model
attack_model.eval()
correct = 0
total = 0

for features, labels in test_loader:
    features = features.to(device)
    labels = labels.to(device)

    outputs = attack_model(features)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Attack model accuracy on test dataset: {accuracy:.2f}%")

