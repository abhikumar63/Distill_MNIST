"""Teacher and student CNN architectures. No project imports."""

import mlx.nn as nn


class TeacherCNN1(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * 7 * 7, 256)
        self.fc2 = nn.Linear(256, 10)

    def __call__(self, x):
        x = nn.relu(self.conv1(x))
        x = nn.relu(self.conv2(x))
        x = self.pool(x)
        x = nn.relu(self.conv3(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = nn.relu(self.fc1(x))
        return self.fc2(x)


class TeacherCNN2(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(1, 48, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(48, 96, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(96, 160, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(160 * 7 * 7, 256)
        self.fc2 = nn.Linear(256, 10)

    def __call__(self, x):
        x = nn.relu(self.conv1(x))
        x = self.pool(x)
        x = nn.relu(self.conv2(x))
        x = nn.relu(self.conv3(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = nn.relu(self.fc1(x))
        return self.fc2(x)


class StudentCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 7 * 7, 64)
        self.fc2 = nn.Linear(64, 10)

    def __call__(self, x):
        x = nn.relu(self.conv1(x))
        x = self.pool(x)
        x = nn.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = nn.relu(self.fc1(x))
        return self.fc2(x)