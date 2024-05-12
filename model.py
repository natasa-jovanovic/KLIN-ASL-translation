import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim

# Formiraje modela
class Net(nn.Module):
    def __init__(self, input_shape=[3, 5, 5], num_classes=29, conv_layers=[16, 32], fc_layers=[120, 84], strides=[5, 5]):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], conv_layers[0], 5)
        self.conv2 = nn.Conv2d(conv_layers[0], conv_layers[1], 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.bn1 = nn.BatchNorm2d(conv_layers[0])
        self.bn2 = nn.BatchNorm2d(conv_layers[1])
        self.fc1 = nn.Linear(conv_layers[1] * 4 * 4, fc_layers[0])
        self.fc2 = nn.Linear(fc_layers[0], fc_layers[1])
        self.fc3 = nn.Linear(fc_layers[1], num_classes)

    def forward(self, x):
        x = self.pool(self.bn1(F.relu(self.conv1(x))))
        x = self.pool(self.bn2(F.relu(self.conv2(x))))
        x = x.view(-1, 32 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x