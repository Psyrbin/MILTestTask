import torch
from torch import nn

class ResNet20(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(ResNet20, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(16)

        self.convs = []
        self.bns = []
        layer_sizes = [6, 6, 6]
        channels = [16, 32, 64]

        for layer in range(len(layer_sizes)):
            if layer == 0:
                self.convs.append(nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False))
            else:
                self.convs.append(nn.Conv2d(in_channels=channels[layer-1], out_channels=channels[layer], kernel_size=3, stride=2, padding=1, bias=False))

            self.bns.append(nn.BatchNorm2d(channels[layer]))

            for _ in range(layer_sizes[layer] - 1):
                self.convs.append(nn.Conv2d(in_channels=channels[layer], out_channels=channels[layer], kernel_size=3, stride=1, padding=1, bias=False))
                self.bns.append(nn.BatchNorm2d(channels[layer]))
        self.convs = nn.ModuleList(self.convs)
        self.bns = nn.ModuleList(self.bns)

        self.shortcut_16_32 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=1, stride=2, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        self.shortcut_32_64 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1, stride=2, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(64)

        self.avgpool = nn.AvgPool2d(kernel_size=8)
        self.linear = nn.Linear(in_features=64, out_features=num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        shortcut = x

        for i in range(len(self.convs)):
            x = self.convs[i](x)
            x = self.bns[i](x)
            if i % 2 == 1:
                if i == 7:
                    x += self.bn2(self.shortcut_16_32(shortcut))
                elif i == 13:
                    x += self.bn3(self.shortcut_32_64(shortcut))
                else:
                    x += shortcut
            x = self.relu(x)
            if i % 2 == 1:
                shortcut = x

        x = self.avgpool(x)
        x = self.linear(x.view(-1, 64))
        return x
