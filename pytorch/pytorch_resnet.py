import numpy as np
import torch.nn as nn


class ResidualBlock3D(nn.Module):
    def __init__(self, inchannel, outchannel, stride):
        super(ResidualBlock3D, self).__init__()

        self.convs = nn.Sequential(
            nn.Conv3d(inchannel, outchannel, kernel_size=3,
                      stride=stride, padding=1, bias=False),
            nn.BatchNorm3d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv3d(outchannel, outchannel, kernel_size=3,
                      stride=1, padding=1, bias=False))

        self.shortcut = nn.Sequential()

        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Conv2d(inchannel, outchannel,
                                      kernel_size=1, stride=stride,
                                      padding=0, bias=False)
        else:
            self.shortcut = nn.Sequential()

        self.final = nn.Sequential(
            nn.BatchNorm3d(outchannel),
            nn.ReLU(inplace=True))

    def forward(self, x):
        out = self.convs(x) + self.shortcut(x)

        out = self.final(out)

        return out


class PytorchResNet3D(nn.Module):

    def __init__(self, input_shape, attention, dropout_prob, in_chan=1):
        super(PytorchResNet3D, self).__init__()

        self.use_attention = attention
        self.dropout_prob = dropout_prob
        self.in_chan = in_chan

        self.block_params = [
            [(64, 2), (64, 1), (64, 1)],
            [(128, 2), (128, 1), (128, 1)],
            [(256, 2), (256, 1), (256, 1)],
        ]

        self.input_shape = input_shape

        pool_shape = np.array(self.input_shape)

        for layer in self.block_params:
            for _, s in layer:
                # Assume padding 1, kernel 3 at all points.
                pool_shape = (pool_shape - 1) // s + 1

        self.pool_shape = pool_shape

        self.conv1 = self.make_layer(self.block_params[0])
        self.conv2 = self.make_layer(self.block_params[1])
        self.conv3 = self.make_layer(self.block_params[2])

        self.classify = nn.Sequential(
            nn.AvgPool3d(tuple(self.pool_shape)),
            nn.Flatten(),
            nn.Linear(self.in_chan, 2),
            nn.Dropout(self.dropout_prob))

    def make_layer(self, block_list):
        blocks = []
        for out_chan, stride in block_list:
            blocks.append(ResidualBlock3D(self.inchannels, out_chan, stride))
            self.in_chan = out_chan
        return nn.Sequential(*blocks)
