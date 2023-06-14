import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet34, mobilenet_v2
import pickle
from sklearn.model_selection import train_test_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# resnet34:     cifar10: 68.49% | tinyimagenet: 19.65%
# mobilenetv2:  cifar10: 72.25% | tinyimagenet: 24.19%

DATA_PATH = '../pickle/cifar10/mobilenetv2/shadow.p'
MODEL_NAME = '../models/mobilenetv2_cifar10_shadow.pth'
TRAIN_DATA_PATH = '../pickle/cifar10/mobilenetv2/shadow_train.p'
TEST_DATA_PATH = '../pickle/cifar10/mobilenetv2/shadow_test.p'
num_c = 10
model = mobilenet_v2(pretrained=False, num_classes=num_c).to(device)

with open(DATA_PATH, "rb") as f:
    all_dataset = pickle.load(f)

# Divide the dataset into training and evaluation sets
train_dataset, test_dataset = train_test_split(all_dataset, test_size=0.2, random_state=42)

with open(TRAIN_DATA_PATH, 'wb') as file:
    pickle.dump(train_dataset, file)

with open(TEST_DATA_PATH, 'wb') as file:
    pickle.dump(test_dataset, file)

with open(TRAIN_DATA_PATH, "rb") as f:
    train_dataset = pickle.load(f)

with open(TEST_DATA_PATH, "rb") as f:
    test_dataset = pickle.load(f)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=True, num_workers=2)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

# Training loop
num_epochs = 199 
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

        if i % 100 == 99:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {running_loss/100:.4f}')
            running_loss = 0.0
    
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

    accuracy = 100 * correct / total
    print(f'Accuracy on the train set: {accuracy:.2f}%', end=" ")

    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'| on the test set: {accuracy:.2f}%')

# Save the shadow model
torch.save(model.state_dict(), MODEL_NAME)

