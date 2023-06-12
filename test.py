import pickle
import torch
import matplotlib.pyplot as plt
import torchvision.models as models

## For shadow.p
# DATA_PATH = '../pickle/cifar10/resnet34/shadow.p'

# device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# with open(DATA_PATH, "rb") as f:
#     dataset = pickle.load(f)

# print(type(dataset), len(dataset))
# print(type(dataset[0]), len(dataset[0]))

# print(type(dataset[0][0]), dataset[0][0].shape)
# print(type(dataset[0][1]), dataset[0][1])

# dataloader = torch.utils.data.DataLoader(
#     dataset, batch_size=64, shuffle=False, num_workers=1)

# for batch_idx, (img, label) in enumerate(dataloader):
#     img = img.to(device)

## For eval.p
# DATA_PATH = '../pickle/cifar10/resnet34/eval.p'
# # Change the DATA_PATH to your local pickle file path

# device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# with open(DATA_PATH, "rb") as f:
#     dataset = pickle.load(f)

# print(type(dataset), len(dataset))
# print(type(dataset[0]), len(dataset[0]))

# print(type(dataset[0][0]), dataset[0][0].shape)
# print(type(dataset[0][1]), dataset[0][1])
# print(type(dataset[0][2]), dataset[0][2])

# dataloader = torch.utils.data.DataLoader(
#     dataset, batch_size=100, shuffle=False, num_workers=1)

# for batch_idx, (img, label, isMemeber) in enumerate(dataloader):
#     img = img.to(device)

## Model load
# MODEL_PATH = '../models/resnet34_cifar10.pth'
# # Change the MODEL_PATH to your local model path

# device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# target_model = models.resnet34(num_classes=10).to(device)
# # Change num_classes to 200 when you use the Tiny ImageNet dataset

# state_dict = torch.load(MODEL_PATH, map_location=device)
# target_model.load_state_dict(state_dict['net'])

# print(target_model)

## Result save
# DATA_PATH = '../pickle/cifar10/resnet34/test.p'

# with open(DATA_PATH, "rb") as f:
#     dataset = pickle.load(f)

# import numpy as np

# prediction = [1 for i in range(len(dataset))]
# print(prediction)

# np.save('../results/task0_resnet34_cifar10.npy', prediction)
# np.save('../results/task1_mobilenetv2_cifar10.npy', prediction)
# np.save('../results/task2_resnet34_tinyimagenet.npy', prediction)
# np.save('../results/task3_mobilenetv2_tinyimagenet.npy', prediction)

# test = np.load('../results/task3_mobilenetv2_tinyimagenet.npy')
# print(test)