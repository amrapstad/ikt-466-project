import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.nn import init

class AudioClassifier(nn.Module):

    def __init__(self):
        super().__init__()
        conv_layers = []

        # First Convolution Block
        self.conv1 = nn.Conv2d(2, 32, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(32)
        init.kaiming_normal_(self.conv1.weight, nonlinearity='relu')
        self.conv1.bias.data.zero_()
        conv_layers += [self.conv1, self.relu1, self.bn1]

        # Second Convolution Block
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(64)
        init.kaiming_normal_(self.conv2.weight, nonlinearity='relu')
        self.conv2.bias.data.zero_()
        conv_layers += [self.conv2, self.relu2, self.bn2]

        # Third Convolution Block
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.relu3 = nn.ReLU()
        self.bn3 = nn.BatchNorm2d(128)
        init.kaiming_normal_(self.conv3.weight, nonlinearity='relu')
        self.conv3.bias.data.zero_()
        conv_layers += [self.conv3, self.relu3, self.bn3]

        # Fourth Convolution Block
        self.conv4 = nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.relu4 = nn.ReLU()
        self.bn4 = nn.BatchNorm2d(256)
        init.kaiming_normal_(self.conv4.weight, nonlinearity='relu')
        self.conv4.bias.data.zero_()
        conv_layers += [self.conv4, self.relu4, self.bn4]

        # Fifth Convolution Block
        self.conv5 = nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.relu5 = nn.ReLU()
        self.bn5 = nn.BatchNorm2d(512)
        init.kaiming_normal_(self.conv5.weight, nonlinearity='relu')
        self.conv5.bias.data.zero_()
        conv_layers += [self.conv5, self.relu5, self.bn5]

        # Convolution blocks wrapped in Sequential
        self.conv = nn.Sequential(*conv_layers)

        # Final global Adaptive Pooling
        self.ap = nn.AdaptiveAvgPool2d(output_size=1)

        self.dropout = nn.Dropout(p=0.5)
        self.lin = nn.Linear(512, 10)

    def forward(self, x):
        x = self.conv(x)     # Goes through conv + relu + bn 5 times
        x = self.ap(x)     
        x = x.view(x.shape[0], -1)
        x = self.dropout(x)
        x = self.lin(x)
        return x
