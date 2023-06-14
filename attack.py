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

# Generate dataset for attack model
# in_member = []
# out_member = []
# model.eval()
# with torch.no_grad():
#     for data in train_loader:
#         images, labels = data[0].to(device), data[1].to(device)
#         outputs = model(images)
#         in_member.append([outputs.data, labels])
#     print(len(in_member), len(in_member[0]), len(in_member[1]), len(in_member[2]))

#     for data in test_loader:
#         images, labels = data[0].to(device), data[1].to(device)
#         outputs = model(images)
#         for idx in range(len(data)):
#             out_member.append([outputs[idx].data, labels[idx], 0])
#     print(len(out_member), len(out_member[0]), len(out_member[1]), len(out_member[2]))

device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

MODEL_PATH = 'shadow_resnet_cifar10.pth'

shadow_model = models.resnet34(num_classes=200).to(device)
# Change num_classes to 200 when you use the Tiny ImageNet dataset

state_dict = torch.load(MODEL_PATH, map_location=device)
print(state_dict)
# target_model.load_state_dict(state_dict['net']) # acc: 68.49%, epoch: 199
# # print(target_model)

# # print(target_model)
# target_model.eval()

# Hyperparameters
batch_size = 64

DATA_PATH = '../pickle/cifar10/resnet34/shadow.p'

with open(DATA_PATH, "rb") as f:
    test_dataset = pickle.load(f)

# test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


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

