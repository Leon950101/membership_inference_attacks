import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet34, mobilenet_v2
import pickle
from sklearn.model_selection import train_test_split
import sys

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# resnet34:     cifar10: 68.49% | tinyimagenet: 19.65%
# mobilenetv2:  cifar10: 72.25% | tinyimagenet: 24.19%

# Check if the required number of arguments is provided and retrive
if len(sys.argv) > 3 or len(sys.argv) < 3:
    print("Usage: python script_name.py settings epochs")
    sys.exit(1)
else:
    idx = int(sys.argv[1])
    num_epochs = int(sys.argv[2])

settings = [['../pickle/cifar10/resnet34/shadow.p', '../models/resnet34_cifar10_shadow_1.pth', 
             '../pickle/cifar10/resnet34/shadow_train.p', '../pickle/cifar10/resnet34/shadow_test.p', 10, 0],
             ['../pickle/cifar10/mobilenetv2/shadow.p', '../models/mobilenetv2_cifar10_shadow_1.pth', 
             '../pickle/cifar10/mobilenetv2/shadow_train.p', '../pickle/cifar10/mobilenetv2/shadow_test.p', 10, 1],
             ['../pickle/tinyimagenet/resnet34/shadow.p', '../models/resnet34_tinyimagenet_shadow_1.pth', 
             '../pickle/tinyimagenet/resnet34/shadow_train.p', '../pickle/tinyimagenet/resnet34/shadow_test.p', 200, 0],
             ['../pickle/tinyimagenet/mobilenetv2/shadow.p', '../models/mobilenetv2_tinyimagenet_shadow_1.pth', 
             '../pickle/tinyimagenet/mobilenetv2/shadow_train.p', '../pickle/tinyimagenet/mobilenetv2/shadow_test.p', 200, 1],
             ['../pickle/cifar10/resnet34/shadow.p', '../models/resnet34_cifar10_shadow_2.pth', 
             '../pickle/cifar10/resnet34/shadow_test.p', '../pickle/cifar10/resnet34/shadow_train.p', 10, 0],
             ['../pickle/cifar10/mobilenetv2/shadow.p', '../models/mobilenetv2_cifar10_shadow_2.pth', 
             '../pickle/cifar10/mobilenetv2/shadow_test.p', '../pickle/cifar10/mobilenetv2/shadow_train.p', 10, 1],
             ['../pickle/tinyimagenet/resnet34/shadow.p', '../models/resnet34_tinyimagenet_shadow_2.pth', 
             '../pickle/tinyimagenet/resnet34/shadow_test.p', '../pickle/tinyimagenet/resnet34/shadow_train.p', 200, 0],
             ['../pickle/tinyimagenet/mobilenetv2/shadow.p', '../models/mobilenetv2_tinyimagenet_shadow_2.pth', 
             '../pickle/tinyimagenet/mobilenetv2/shadow_test.p', '../pickle/tinyimagenet/mobilenetv2/shadow_train.p', 200, 1]]

DATA_PATH = settings[idx][0]
MODEL_NAME = settings[idx][1]
TRAIN_DATA_PATH = settings[idx][2]
TEST_DATA_PATH = settings[idx][3]
num_c = settings[idx][4]
if settings[idx][5] == 0:
    model = resnet34(pretrained=False, num_classes=num_c).to(device)
else:
    model = mobilenet_v2(pretrained=False, num_classes=num_c).to(device)

if idx >= 0 and idx <= 3:
    with open(DATA_PATH, "rb") as f:
        all_dataset = pickle.load(f)

    train_dataset, test_dataset = train_test_split(all_dataset, test_size=0.5)

    with open(TRAIN_DATA_PATH, 'wb') as file:
        pickle.dump(train_dataset, file)

    with open(TEST_DATA_PATH, 'wb') as file:
        pickle.dump(test_dataset, file)

with open(TRAIN_DATA_PATH, "rb") as f: # Another Half
    train_dataset = pickle.load(f)

with open(TEST_DATA_PATH, "rb") as f: # Another Half
    test_dataset = pickle.load(f)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=True, num_workers=2)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for i, data in enumerate(train_loader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        model.train()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    epoch_loss = 128 * running_loss / len(train_dataset)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}') # Step [{i+1}/{len(train_loader)}],
    
    model.eval()  
    correct = 0
    total = 0
    with torch.no_grad():
        for data in train_loader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy_train = 100 * correct / total
    print(f'Accuracy on the train set: {accuracy_train:.2f}%', end=" ")

    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy_test = 100 * correct / total
    print(f'| on the test set: {accuracy_test:.2f}%')
    
    # if epoch_loss < 0.01 and accuracy_train > 99.99:
    #     break

# Save the shadow model
torch.save(model.state_dict(), MODEL_NAME)

