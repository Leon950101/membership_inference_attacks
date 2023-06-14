import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet34
import multiprocessing
import pickle

# Define the shadow model architecture
class ShadowModel(nn.Module):
    def __init__(self):
        super(ShadowModel, self).__init__()
        self.resnet = resnet34(pretrained=False, num_classes=10) # Using ResNet34 as the base model
        # num_filters = self.model.fc.in_features
        # self.fc = nn.Linear(num_filters, 2)  # Output two classes: in-dataset or out-of-dataset

    def forward(self, x):
        x = self.resnet(x)
        return x

# Load CIFAR-10 dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

DATA_PATH = '../pickle/cifar10/resnet34/shadow.p'

with open(DATA_PATH, "rb") as f:
    train_dataset = pickle.load(f)

# train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=multiprocessing.cpu_count())

# Create shadow model instance
shadow_model = ShadowModel()

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(shadow_model.parameters(), lr=0.01, momentum=0.9)

# Train the shadow model
num_epochs = 10
if __name__ == '__main__':
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = shadow_model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if (i + 1) % 200 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, num_epochs, i + 1, len(train_loader), running_loss / 200))
                running_loss = 0.0

    # Save the shadow model
    torch.save(shadow_model.state_dict(), 'shadow_model.pth')
