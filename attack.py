import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet34, mobilenet_v2
import pickle
from torch.utils.data import DataLoader
import sys
from sklearn.model_selection import train_test_split
import numpy as np

device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class AttackModel(nn.Module):
    def __init__(self):
        super(AttackModel, self).__init__()
        self.fc1 = nn.Linear(11, 32)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(32, 64)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(64, 32)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        out = self.relu3(out)
        out = self.fc4(out)
        out = self.sigmoid(out)
        return out.squeeze()
    
class AttackModel_2(nn.Module):
    def __init__(self):
        super(AttackModel_2, self).__init__()
        self.fc1 = nn.Linear(201, 1024)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(1024, 256)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(256, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        out = self.sigmoid(out)
        return out.squeeze()

if len(sys.argv) > 2 or len(sys.argv) < 2:
    print("Usage: python script_name.py task_idx")
    sys.exit(1)
else:
    idx = int(sys.argv[1])

settings = [['../models/resnet34_cifar10.pth', '../models/resnet34_cifar10_shadow_3.pth',
             '../pickle/cifar10/resnet34/shadow_train_2.p', '../pickle/cifar10/resnet34/shadow_test_2.p',
             '../pickle/cifar10/resnet34/eval.p', '../pickle/cifar10/resnet34/test.p', 10, 0,
             '../results/task0_resnet34_cifar10.npy', 75, '../models/resnet34_cifar10_shadow_4.pth', 100, 1,
             '../pickle/cifar10/resnet34/shadow_train.p', '../pickle/cifar10/resnet34/shadow_test.p',
             '../models/resnet34_cifar10_shadow_1.pth', '../models/resnet34_cifar10_shadow_2.pth'],
             ['../models/mobilenetv2_cifar10.pth', '../models/mobilenetv2_cifar10_shadow_3.pth',
             '../pickle/cifar10/mobilenetv2/shadow_train_2.p', '../pickle/cifar10/mobilenetv2/shadow_test_2.p',
             '../pickle/cifar10/mobilenetv2/eval.p', '../pickle/cifar10/mobilenetv2/test.p', 10, 1,
             '../results/task1_mobilenetv2_cifar10.npy', 73, '../models/mobilenetv2_cifar10_shadow_4.pth', 100, 1,
             '../pickle/cifar10/mobilenetv2/shadow_train.p', '../pickle/cifar10/mobilenetv2/shadow_test.p',
             '../models/mobilenetv2_cifar10_shadow_1.pth', '../models/mobilenetv2_cifar10_shadow_2.pth'],
             ['../models/resnet34_tinyimagenet.pth', '../models/resnet34_tinyimagenet_shadow_3.pth',
             '../pickle/tinyimagenet/resnet34/shadow_train_2.p', '../pickle/tinyimagenet/resnet34/shadow_test_2.p',
             '../pickle/tinyimagenet/resnet34/eval.p', '../pickle/tinyimagenet/resnet34/test.p',  200, 0,
             '../results/task2_resnet34_tinyimagenet.npy', 98, '../models/resnet34_tinyimagenet_shadow_4.pth', 5, 50,
             '../pickle/tinyimagenet/resnet34/shadow_train.p', '../pickle/tinyimagenet/resnet34/shadow_test.p',
             '../models/resnet34_tinyimagenet_shadow_1.pth', '../models/resnet34_tinyimagenet_shadow_2.pth'],
             ['../models/mobilenetv2_tinyimagenet.pth', '../models/mobilenetv2_tinyimagenet_shadow_3.pth',
             '../pickle/tinyimagenet/mobilenetv2/shadow_train_2.p', '../pickle/tinyimagenet/mobilenetv2/shadow_test_2.p',
             '../pickle/tinyimagenet/mobilenetv2/eval.p', '../pickle/tinyimagenet/mobilenetv2/test.p', 200, 1,
             '../results/task3_mobilenetv2_tinyimagenet.npy', 85.5, '../models/mobilenetv2_tinyimagenet_shadow_4.pth', 5, 50,
             '../pickle/tinyimagenet/mobilenetv2/shadow_train.p', '../pickle/tinyimagenet/mobilenetv2/shadow_test.p',
             '../models/mobilenetv2_tinyimagenet_shadow_1.pth', '../models/mobilenetv2_tinyimagenet_shadow_2.pth']
             ]

# 68% 65% 90% 80%
TARGET_MODEL_PATH = settings[idx][0]
SHADOW_MODEL_PATH_1 = settings[idx][15]
SHADOW_MODEL_PATH_2 = settings[idx][16]
SHADOW_MODEL_PATH_3 = settings[idx][1]
SHADOW_MODEL_PATH_4 = settings[idx][10]
TRAIN_DATA_PATH = settings[idx][13]
TEST_DATA_PATH = settings[idx][14]
TRAIN_DATA_PATH_2 = settings[idx][2]
TEST_DATA_PATH_2 = settings[idx][3]
EVAL_DATA_PATH = settings[idx][4]
FINAL_TEST = settings[idx][5]
CLASS_NUM = settings[idx][6]

if settings[idx][7] == 0:
    shadow_model = resnet34(num_classes=CLASS_NUM).to(device)
    target_model = resnet34(num_classes=CLASS_NUM).to(device)
else:
    shadow_model = mobilenet_v2(num_classes=CLASS_NUM).to(device)
    target_model = mobilenet_v2(num_classes=CLASS_NUM).to(device)

SAVE_NAME = settings[idx][8]
BEST_ACC = settings[idx][9]
num_epochs = settings[idx][11]
label_divide = settings[idx][12]

with open(TRAIN_DATA_PATH, "rb") as f: 
    train_dataset_1 = pickle.load(f)
    
with open(TEST_DATA_PATH, "rb") as f: 
    test_dataset_1 = pickle.load(f)

with open(TRAIN_DATA_PATH, "rb") as f: 
    test_dataset_2 = pickle.load(f)

with open(TEST_DATA_PATH, "rb") as f: 
    train_dataset_2 = pickle.load(f)

with open(TRAIN_DATA_PATH_2, "rb") as f: 
    train_dataset_3 = pickle.load(f)

with open(TEST_DATA_PATH_2, "rb") as f: 
    test_dataset_3 = pickle.load(f)

with open(TRAIN_DATA_PATH_2, "rb") as f: 
    test_dataset_4 = pickle.load(f)

with open(TEST_DATA_PATH_2, "rb") as f: 
    train_dataset_4 = pickle.load(f)

with open(EVAL_DATA_PATH, "rb") as f:
    eval_dataset = pickle.load(f)

with open(FINAL_TEST, "rb") as f:
    final_dataset = pickle.load(f)

train_loader_1 = torch.utils.data.DataLoader(train_dataset_1, batch_size=128, shuffle=True, num_workers=2)
test_loader_1 = torch.utils.data.DataLoader(test_dataset_1, batch_size=128, shuffle=True, num_workers=2)
train_loader_2 = torch.utils.data.DataLoader(train_dataset_2, batch_size=128, shuffle=True, num_workers=2)
test_loader_2 = torch.utils.data.DataLoader(test_dataset_2, batch_size=128, shuffle=True, num_workers=2)
train_loader_3 = torch.utils.data.DataLoader(train_dataset_3, batch_size=128, shuffle=True, num_workers=2)
test_loader_3 = torch.utils.data.DataLoader(test_dataset_3, batch_size=128, shuffle=True, num_workers=2)
train_loader_4 = torch.utils.data.DataLoader(train_dataset_4, batch_size=128, shuffle=True, num_workers=2)
test_loader_4 = torch.utils.data.DataLoader(test_dataset_4, batch_size=128, shuffle=True, num_workers=2)

eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=len(eval_dataset), shuffle=False, num_workers=2)
final_loader = torch.utils.data.DataLoader(final_dataset, batch_size=len(final_dataset), shuffle=False, num_workers=2)

def divide(dataset, class_num):
    divided_data = [[] for i in range(class_num)]
    for idx in range(len(dataset)):
        label = dataset[idx][1].item()
        divided_data[label].append(dataset[idx])
    return divided_data

def prepare_dataset(dataset, class_num):
    input_data = [[] for _ in range(class_num)]
    labels = [[] for _ in range(class_num)]
    for i in range(class_num):
        for j in range(len(dataset[i])):
            prediction = dataset[i][j][0].tolist()
            label =  dataset[i][j][1].tolist() / label_divide
            combined = prediction + [label]
            input_data[i].append(combined)
            labels[i].append(dataset[i][j][2])
    return input_data, labels

def divide_eval(dataset, class_num):
    eval_divided_input_data = [[] for _ in range(class_num)]
    eval_divided_labels = [[] for _ in range(class_num)]
    eval_divided_members = [[] for _ in range(class_num)]
    for idx in range(len(dataset)):
        label = dataset[idx][1]
        eval_divided_input_data[label].append(dataset[idx][0])
        eval_divided_labels[label].append(dataset[idx][1])
        eval_divided_members[label].append(dataset[idx][2])
    # for idx in range(CLASS_NUM):
    #     print(len(eval_divided_input_data[idx]))
    return eval_divided_input_data, eval_divided_labels, eval_divided_members

if __name__ == '__main__':

    shadow_set = [SHADOW_MODEL_PATH_1, SHADOW_MODEL_PATH_2, SHADOW_MODEL_PATH_3, SHADOW_MODEL_PATH_4]
    train_data_set = [train_loader_1, train_loader_2, train_loader_3, train_loader_4]
    test_data_set = [test_loader_1, test_loader_2, test_loader_3, test_loader_4]
    input_data_set = []
    labels_set = []
    num_to_train_start = 0
    num_to_train_end = 4
    
    for idx in range(num_to_train_start, num_to_train_end):
        state_dict = torch.load(shadow_set[idx], map_location=device)
        shadow_model.load_state_dict(state_dict)
        # Generate dataset for attack model
        shadow_model.eval()
        with torch.no_grad():
            in_member_all = []
            train_data = train_data_set[idx]
            for data in train_data:
                images, labels = data[0].to(device), data[1].to(device)
                outputs = shadow_model(images)
                for i in range(len(outputs)):
                    in_member_all.append([outputs[i], labels[i], 1])
            
            out_member_all = []
            test_data = test_data_set[idx]
            for data in test_data:
                images, labels = data[0].to(device), data[1].to(device)
                outputs = shadow_model(images)
                for i in range(len(outputs)):
                    out_member_all.append([outputs[i], labels[i], 0])
            
            # print(len(in_member_all), in_member_all[0][0], in_member_all[0][1], in_member_all[0][2])
            # print(len(out_member_all), out_member_all[0][0], out_member_all[0][1], out_member_all[0][2])

        in_member_divided = divide(in_member_all, CLASS_NUM)
        out_member_divided = divide(out_member_all, CLASS_NUM)
        members_divided = [[] for _ in range(CLASS_NUM)]
        for i in range(CLASS_NUM):
            members_divided[i] = in_member_divided[i] + out_member_divided[i]
        input_data_all, labels_all = prepare_dataset(members_divided, CLASS_NUM)
        input_data_set.append(input_data_all)
        labels_set.append(labels_all)

    # Training loop for each attack model
    not_good = True
    while not_good:
         # Create attack models
        attack_models = []
        for _ in range(CLASS_NUM):
            if CLASS_NUM == 10:
                model = AttackModel()
                attack_models.append(model)
            else:
                model = AttackModel_2()
                attack_models.append(model)
        for epoch in range(num_epochs):
            for idx in range(len(attack_models)):
                model = attack_models[idx]
                model.to(device)
                criterion = nn.BCELoss()
                optimizer = optim.Adam(params=model.parameters(), lr=0.001)
                loss = 0.0
                model.train()
                for i in range(len(labels_set)): 
                    input_data = torch.tensor(input_data_set[i][idx], dtype=torch.float32).to(device)
                    labels = torch.tensor(labels_set[i][idx], dtype=torch.float32).to(device)
                
                    # Forward pass
                    outputs = model(input_data)
                    loss = criterion(outputs, labels)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            # Target Model
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
                label = labels[i].tolist() / label_divide
                combined = prediction + [label]
                input_data = torch.tensor(combined, dtype=torch.float32).to(device)
                out = a_model(input_data)
                if out >= 0.5: predict_members.append(1)
                else: predict_members.append(0)
            
            correct = 0
            for i in range(len(real_members)):
                if predict_members[i] == int(real_members[i]):
                    correct += 1
            accuracy = 100 * correct / len(real_members)
            print(f"Epoch [{epoch+1}/{num_epochs}] | Attack Accuracy: {accuracy:.2f}%")
            if accuracy >= BEST_ACC:
                not_good = False
                break

    # Target Model
    state_dict = torch.load(TARGET_MODEL_PATH, map_location=device)
    target_model.load_state_dict(state_dict["net"])
    target_model.eval()

    predict_members = []
    with torch.no_grad():
        for data in final_loader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = target_model(images)

    for i in range(len(labels)):
        a_model = attack_models[labels[i]]
        a_model.eval()
        prediction = outputs[i].tolist()
        label = labels[i].tolist() / label_divide
        combined = prediction + [label]
        input_data = torch.tensor(combined, dtype=torch.float32).to(device)
        out = a_model(input_data)
        if out >= 0.5: predict_members.append(1)
        else: predict_members.append(0)

    if accuracy >= BEST_ACC:
        np.save(SAVE_NAME, predict_members)
        test = np.load(SAVE_NAME)

