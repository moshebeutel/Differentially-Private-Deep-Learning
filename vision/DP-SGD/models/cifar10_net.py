from torch import nn
import torch.nn.functional as F


class cifar10Net(nn.Module):
    def __init__(self):
        super(cifar10Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.fc1 = nn.Linear(32 * 4 * 4, 32 * 4 * 4)
        self.fc2 = nn.Linear(32 * 4 * 4, 32 * 2 * 2)
        self.fc3 = nn.Linear(32 * 2 * 2, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 32 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class TinyCifarNet(nn.Module):
    def __init__(self, num_filters = 4):
        super(TinyCifarNet, self).__init__()
        self.representation_size = num_filters * 7 * 7
        self.conv1 = nn.Conv2d(3, num_filters, 3, padding=1, stride=2)
        self.bn1 = nn.BatchNorm2d(num_filters, affine=False)
        self.relu = nn.ReLU(inplace=False)
        self.pool = nn.MaxPool2d(3, stride=2)
        self.fc1 = nn.Linear(self.representation_size, 10)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(-1, self.representation_size)
        x = self.fc1(x)
        assert x.size(0) == batch_size, f'output shape error. Expected {batch_size} but got {x.size(0)}'
        return x