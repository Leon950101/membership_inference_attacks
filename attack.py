import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet34, mobilenet_v2
import pickle
from torch.utils.data import DataLoader

device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

TARGET_MODEL_PATH = '../models/resnet34_cifar10.pth'
SHADOW_MODEL_PATH = '../models/resnet34_cifar10_shadow.pth'
TRAIN_DATA_PATH = '../pickle/cifar10/resnet34/shadow_train.p'
TEST_DATA_PATH = '../pickle/cifar10/resnet34/shadow_test.p'
EVAL_DATA_PATH = '../pickle/cifar10/resnet34/eval.p'
CLASS_NUM = 10

shadow_model = resnet34(num_classes=CLASS_NUM).to(device)
target_model = resnet34(num_classes=CLASS_NUM).to(device)

# Define the architecture for the attack model
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

with open(TRAIN_DATA_PATH, "rb") as f:
    train_dataset = pickle.load(f)

with open(TEST_DATA_PATH, "rb") as f:
    test_dataset = pickle.load(f)

with open(EVAL_DATA_PATH, "rb") as f:
    eval_dataset = pickle.load(f) # [3*32*32], label, in/out

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=True, num_workers=2)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=True, num_workers=2)
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
        for data in train_loader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = shadow_model(images)

        in_member_all = []
        for i in range(len(train_dataset)):
            in_member_all.append([outputs[i], labels[i], 1])

        for data in test_loader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = shadow_model(images)

        out_member_all = []
        for i in range(len(test_dataset)):
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
    for _ in range(10):
        model = AttackModel()
        attack_models.append(model)

    # Set the device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Move the attack models to the device
    for model in attack_models:
        model.to(device)

    # Define the loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(params=model.parameters(), lr=0.001)

    input_data_all, labels_all = prepare_dataset(members_divided, CLASS_NUM)
    # for i in range(CLASS_NUM):
    #     print(len(input_data_all[i]), len(labels_all[i]))
    # Training loop for each attack model
    num_epochs = 100
    for idx in range(len(attack_models)):
        model = attack_models[idx]
        input_data = torch.tensor(input_data_all[idx], dtype=torch.float32).to(device)
        labels = torch.tensor(labels_all[idx], dtype=torch.float32).to(device)
        for epoch in range(num_epochs):
            # Forward pass
            outputs = model(input_data)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Print the loss for each epoch
            print(f"Attack Model {idx}: Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}")

    # Testing the attack model
    for idx in range(len(attack_models)):
        model = attack_models[idx]
        model.eval()
        input_data = torch.tensor(input_data_all[idx], dtype=torch.float32).to(device)
        labels = torch.tensor(labels_all[idx], dtype=torch.float32).to(device)
       
        outputs = model(input_data)
        correct = 0
        for i in range(len(outputs)):
            if outputs[i] < 0.5 and labels[i] < 0.5:
                correct += 1
            if outputs[i] >= 0.5 and labels[i] > 0.5:
                correct += 1

        accuracy = 100 * correct / len(labels)
        print(f"Attack model accuracy on test dataset: {accuracy:.2f}%")

    # Change num_classes to 200 when you use the Tiny ImageNet dataset

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

