import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import pickle
import torchvision.models as models

MODEL_PATH = '../models/resnet34_cifar10.pth'
# Change the MODEL_PATH to your local model path

device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

target_model = models.resnet34(num_classes=10).to(device)
# Change num_classes to 200 when you use the Tiny ImageNet dataset

state_dict = torch.load(MODEL_PATH, map_location=device)
target_model.load_state_dict(state_dict['net'])

# print(target_model)
target_model.eval()

# Hyperparameters
batch_size = 64

# # Load CIFAR-10 dataset
# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
# ])

DATA_PATH = '../pickle/cifar10/resnet34/shadow.p'

with open(DATA_PATH, "rb") as f:
    test_dataset = pickle.load(f)

# test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = target_model(images)
        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Accuracy of the trained model: {accuracy:.2f}%")

