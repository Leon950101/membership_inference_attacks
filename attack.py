import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet34, mobilenet_v2
import pickle
from torch.utils.data import DataLoader
import sys

device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class AttackModel(nn.Module):
    def __init__(self):
        super(AttackModel, self).__init__()
        self.fc1 = nn.Linear(11, 64)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        out = self.sigmoid(out)
        return out.squeeze()
    
class AttackModel_2(nn.Module):
    def __init__(self):
        super(AttackModel_2, self).__init__()
        self.fc1 = nn.Linear(201, 10)
        # self.relu1 = nn.ReLU()
        # self.fc2 = nn.Linear(512, 256)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(10, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        # out = self.relu1(out)
        # out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        out = self.sigmoid(out)
        return out.squeeze()

if len(sys.argv) > 3 or len(sys.argv) < 3:
    print("Usage: python script_name.py settings")
    sys.exit(1)
else:
    idx = int(sys.argv[1])
    num_epochs = int(sys.argv[2])

settings = [['../models/resnet34_cifar10.pth', '../models/resnet34_cifar10_shadow.pth',
             '../pickle/cifar10/resnet34/shadow_train.p', '../pickle/cifar10/resnet34/shadow_test.p',
             '../pickle/cifar10/resnet34/eval.p', '../pickle/cifar10/resnet34/test.p', 10, 0],
             ['../models/mobilenetv2_cifar10.pth', '../models/mobilenetv2_cifar10_shadow.pth',
             '../pickle/cifar10/mobilenetv2/shadow_train.p', '../pickle/cifar10/mobilenetv2/shadow_test.p',
             '../pickle/cifar10/mobilenetv2/eval.p', '../pickle/cifar10/mobilenetv2/test.p', 10, 1],
             ['../models/resnet34_tinyimagenet.pth', '../models/resnet34_tinyimagenet_shadow.pth',
             '../pickle/tinyimagenet/resnet34/shadow_train.p', '../pickle/tinyimagenet/resnet34/shadow_test.p',
             '../pickle/tinyimagenet/resnet34/eval.p', '../pickle/tinyimagenet/resnet34/test.p',  200, 0],
             ['../models/mobilenetv2_tinyimagenet.pth', '../models/mobilenetv2_tinyimagenet_shadow.pth',
             '../pickle/tinyimagenet/mobilenetv2/shadow_train.p', '../pickle/tinyimagenet/mobilenetv2/shadow_test.p',
             '../pickle/tinyimagenet/mobilenetv2/eval.p', '../pickle/tinyimagenet/mobilenetv2/test.p', 200, 1]
             ]

TARGET_MODEL_PATH = settings[idx][0]
SHADOW_MODEL_PATH = settings[idx][1]
TRAIN_DATA_PATH = settings[idx][2]
TEST_DATA_PATH = settings[idx][3]
EVAL_DATA_PATH = settings[idx][4]
FINAL_TEST = settings[idx][5]

CLASS_NUM = settings[idx][6]

if settings[idx][7] == 0:
    shadow_model = resnet34(num_classes=CLASS_NUM).to(device)
    target_model = resnet34(num_classes=CLASS_NUM).to(device)
else:
    shadow_model = mobilenet_v2(num_classes=CLASS_NUM).to(device)
    target_model = mobilenet_v2(num_classes=CLASS_NUM).to(device)

# Define the architecture for the attack model

with open(TRAIN_DATA_PATH, "rb") as f:
    train_dataset = pickle.load(f)

with open(TEST_DATA_PATH, "rb") as f:
    test_dataset = pickle.load(f)

with open(EVAL_DATA_PATH, "rb") as f:
    eval_dataset = pickle.load(f) # [3*32*32], label, in/out

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=True, num_workers=2)
eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=len(eval_dataset), shuffle=False, num_workers=2)

def divide(dataset, class_num):
    divided_data = [[] for i in range(class_num)]
    for idx in range(len(dataset)):
        label = dataset[idx][1].item()
        divided_data[label].append(dataset[idx])
    return divided_data

def prepare_dataset(dataset, class_num):
    input_data = [[] for i in range(class_num)]
    labels = [[] for i in range(class_num)]
    for i in range(class_num):
        for j in range(len(dataset[i])):
            prediction = dataset[i][j][0].tolist()
            label =  dataset[i][j][1].tolist()
            combined = prediction + [label]
            input_data[i].append(combined)
            labels[i].append(dataset[i][j][2])
    return input_data, labels

if __name__ == '__main__':

    state_dict = torch.load(SHADOW_MODEL_PATH, map_location=device)
    shadow_model.load_state_dict(state_dict)
    # Generate dataset for attack model
    shadow_model.eval()
    with torch.no_grad():
        in_member_all = []
        for data in train_loader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = shadow_model(images)
            for i in range(len(outputs)):
                in_member_all.append([outputs[i], labels[i], 1])
        
        out_member_all = []
        for data in test_loader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = shadow_model(images)
            for i in range(len(outputs)):
                out_member_all.append([outputs[i], labels[i], 0])
        
        # print(len(in_member_all), in_member_all[0][0], in_member_all[0][1], in_member_all[0][2])
        # print(len(out_member_all), out_member_all[0][0], out_member_all[0][1], out_member_all[0][2])

    in_member_divided = divide(in_member_all, CLASS_NUM)
    out_member_divided = divide(out_member_all, CLASS_NUM)
    members_divided = [[] for i in range(CLASS_NUM)]
    for i in range(CLASS_NUM):
        members_divided[i] = in_member_divided[i] + out_member_divided[i]

    # Create a list to store the attack models
    attack_models = []

    # Create ten attack models
    for _ in range(CLASS_NUM):
        if CLASS_NUM == 10:
            model = AttackModel()
            attack_models.append(model)
        else:
            model = AttackModel_2()
            attack_models.append(model)
    # Set the device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Define the loss function and optimizer
    input_data_all, labels_all = prepare_dataset(members_divided, CLASS_NUM)
    # for i in range(CLASS_NUM):
    #     print(len(input_data_all[i]), len(labels_all[i]))
    # Training loop for each attack model
    for idx in range(len(attack_models)):
        model = attack_models[idx]
        model.to(device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(params=model.parameters(), lr=0.001)
        loss = 0.0
        
        input_data = torch.tensor(input_data_all[CLASS_NUM - 1 - idx], dtype=torch.float32).to(device)
        labels = torch.tensor(labels_all[CLASS_NUM - 1 - idx], dtype=torch.float32).to(device)

        for epoch in range(num_epochs):
            # Forward pass
            outputs = model(input_data)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Print the loss for each epoch
        print(f"Attack Model {idx}: Loss: {loss.item()}") # Epoch [{epoch+1}/{num_epochs}],

    # Testing the attack model
    for idx in range(len(attack_models)):
        model = attack_models[idx]
        model.eval()
        
        input_data = torch.tensor(input_data_all[CLASS_NUM - 1 - idx], dtype=torch.float32).to(device)
        labels = torch.tensor(labels_all[CLASS_NUM - 1 - idx], dtype=torch.float32).to(device)
       
        outputs = model(input_data)
        correct = 0
        for i in range(len(outputs)):
            if outputs[i] < 0.5 and labels[i] < 0.5:
                correct += 1
            if outputs[i] >= 0.5 and labels[i] > 0.5:
                correct += 1

        accuracy = 100 * correct / len(labels)
        print(f"Attack model accuracy on train dataset: {accuracy:.2f}%")

    state_dict = torch.load(TARGET_MODEL_PATH, map_location=device)
    target_model.load_state_dict(state_dict["net"])
    target_model.eval()

    predict_members = []
    with torch.no_grad():
        for data in eval_loader:
            images, labels, real_members = data[0].to(device), data[1].to(device), data[2].to(device)
            outputs = target_model(images)

    for i in range(len(labels)):
        a_model = attack_models[labels[i]]
        a_model.eval()
        prediction = outputs[i].tolist()
        label = labels[i].tolist()
        combined = prediction + [label]
        input_data = torch.tensor(combined, dtype=torch.float32).to(device)
        out = a_model(input_data)
        if out > 0.5: predict_members.append(1)
        else: predict_members.append(0)
    
    correct = 0
    for i in range(len(real_members)):
        if predict_members[i] == int(real_members[i]):
            correct += 1
    accuracy = 100 * correct / len(real_members)
    print(f"Attack model accuracy for real: {accuracy:.2f}%")


    ## Result save

    # with open(FINAL_TEST, "rb") as f:
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

