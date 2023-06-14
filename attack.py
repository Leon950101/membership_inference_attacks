import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import pickle
import torchvision.models as models

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet34
import multiprocessing
import pickle

device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Define the shadow model architecture
class ShadowModel(nn.Module):
    def __init__(self):
        super(ShadowModel, self).__init__()
        self.resnet = resnet34(pretrained=False, num_classes=10) # Using ResNet34 as the base model

    def forward(self, x):
        x = self.resnet(x)
        return x

DATA_PATH = '../pickle/cifar10/resnet34/eval.p'

with open(DATA_PATH, "rb") as f:
    test_dataset = pickle.load(f)

# train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=multiprocessing.cpu_count())

# Create shadow model instance
shadow_model = ShadowModel()

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(shadow_model.parameters(), lr=0.01, momentum=0.9)

correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = shadow_model(images)
        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Accuracy of the trained model: {accuracy:.2f}%")


batch_size = 64
# Prepare the attack dataset
attack_dataset = [] # TODO

DATA_PATH = '../pickle/cifar10/resnet34/test.p'

with open(DATA_PATH, "rb") as f:
    test_dataset = pickle.load(f)

# train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

# attack_loader = torch.utils.data.DataLoader(attack_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Perform membership inference attack
attack_model = nn.Linear(11, 2).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(attack_model.parameters(), lr=0.001)

# Training loop for the attack model
# num_epochs = 10
# for epoch in range(num_epochs):
#     for features, labels in attack_loader:
#         features = features.to(device)
#         labels = labels.to(device)

#         optimizer.zero_grad()
#         outputs = attack_model(features)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()

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

