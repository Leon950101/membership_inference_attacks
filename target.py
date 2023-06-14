import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import pickle
import torchvision.models as models

MODEL_PATH = '../models/mobilenetv2_tinyimagenet.pth'
# Change the MODEL_PATH to your local model path

device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

target_model = models.resnet34(num_classes=200).to(device)
# Change num_classes to 200 when you use the Tiny ImageNet dataset

state_dict = torch.load(MODEL_PATH, map_location=device)
print(state_dict)
# target_model.load_state_dict(state_dict['net']) # acc: 68.49%, epoch: 199
# print(target_model)

# print(target_model)
# target_model.eval()

# # Hyperparameters
# batch_size = 64

# DATA_PATH = '../pickle/cifar10/resnet34/shadow.p'

# with open(DATA_PATH, "rb") as f:
#     test_dataset = pickle.load(f)

# # test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
# test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
