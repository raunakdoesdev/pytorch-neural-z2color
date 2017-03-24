import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as initialization
from torch.autograd import Variable


class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=12, out_channels=96, kernel_size=11, stride=3, groups=1)
        self.conv1_pool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv1_pool_drop = nn.Dropout2d(p=0.0)

        self.conv2 = nn.Conv2d(in_channels=102, out_channels=256, kernel_size=3, stride=2, groups=2)
        self.conv2_pool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2_pool_drop = nn.Dropout2d(p=0.0)

        self.ip1 = nn.Linear(in_features=256, out_features=512)
        self.ip1_drop = nn.Dropout(p=0.0)
        self.ip2 = nn.Linear(in_features=512, out_features=20)

        # Initialize weights
        nn.init.normal(self.conv1.weight, std=0.00001)
        nn.init.normal(self.conv2.weight, std=0.1)

        nn.init.xavier_normal(self.ip1.weight)
        nn.init.xavier_normal(self.ip2.weight)

    def forward(self, x, metadata):
        # Conv1
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv1_pool(x)
        x = self.conv1_pool_drop(x)
        
        x = self.conv2_pool_drop(self.conv2_pool(F.relu(self.conv2(x))))  # conv2

        x = self.ip1_drop(F.relu(self.ip1(x)))
        x = self.ip2(x)
        return x

def unit_test():
    test_net = SimpleNet()
    a = test_net( Variable(torch.randn(5, 12, 94, 128)), Variable(torch.randn(1, 6, 13, 26)) )

# unit_test()
