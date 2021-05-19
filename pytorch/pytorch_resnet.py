from typing import Optional, List

import numpy as np
import torch
import torch.nn as nn


class ResidualBlock3D(nn.Module):
    def __init__(self, inchannel: int, outchannel: int, stride: int):
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
            self.shortcut = nn.Conv3d(inchannel, outchannel,
                                      kernel_size=1, stride=stride,
                                      padding=0, bias=False)
        else:
            self.shortcut = nn.Sequential()

        self.final = nn.Sequential(
            nn.BatchNorm3d(outchannel),
            nn.ReLU(inplace=True))

    def forward(self, x: torch.Tensor):
        out = self.convs(x) + self.shortcut(x)

        out = self.final(out)

        return out


class GridAttentionBlock(nn.Module):

    def __init__(self, feat_chan: int, feat_shape: np.ndarray, gate_chan: int, gate_shape: np.ndarray,
                 inter_channels: Optional[int] = None):
        super(GridAttentionBlock, self).__init__()

        self.feat_chan = feat_chan
        self.feat_shape = feat_shape
        self.gate_chan = gate_chan
        self.gate_shape = gate_shape

        self.inter_channels = feat_chan // 2 if inter_channels is None else inter_channels
        self.scale = feat_shape // gate_shape

        # Set 1 bias=False, as will be added to mapped_g later
        self.map_f = nn.Conv3d(feat_chan, self.inter_channels, kernel_size=1)
        self.map_g = nn.Sequential(
            nn.Conv3d(gate_chan, self.inter_channels, kernel_size=1),
            nn.Upsample(scale_factor=tuple(self.scale), mode='trilinear', align_corners=True)
        )

        self.attend = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv3d(self.inter_channels, 1, 1)
        )

    def forward(self, f: torch.Tensor, g: torch.Tensor, return_attention_map: bool):

        mapped_f = self.map_f(f)
        mapped_g = self.map_g(g)

        attention = self.attend(mapped_f + mapped_g)
        # Shift each sample to have a minimum of zero
        shifted_att = attention - torch.amin(attention, dim=[1, 2, 3, 4], keepdim=True)
        # Scale samples so each sum's to 1
        scaled_att = shifted_att / torch.sum(shifted_att, dim=[1, 2, 3, 4], keepdim=True)

        if return_attention_map:
            return scaled_att * mapped_f, scaled_att
        else:
            return scaled_att * mapped_f


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

        self.layer_out_shapes = []
        out_shape = np.array(input_shape)

        for layer in self.block_params:
            for _, s in layer:
                # Assume padding 1, kernel 3 at all points.
                out_shape = (out_shape - 1) // s + 1
                self.layer_out_shapes.append(out_shape)

        self.pool_shape = tuple(self.layer_out_shapes[-1])

        self.conv1 = self.make_layer(self.block_params[0])
        self.conv2 = self.make_layer(self.block_params[1])
        self.conv3 = self.make_layer(self.block_params[2])

        self.classify = nn.Sequential(
            nn.AvgPool3d(self.pool_shape),
            nn.Flatten(),
            nn.Linear(self.in_chan, 2),
            nn.Dropout(self.dropout_prob))

        if self.use_attention:
            self.attention_block = GridAttentionBlock(
                64, self.layer_out_shapes[0],
                256, self.layer_out_shapes[-1])
            self.classify_attention = nn.Sequential(
                nn.Flatten(),
                nn.Linear(self.attention_block.inter_channels * self.layer_out_shapes[0].prod(), 2),
                nn.Dropout(self.dropout_prob)
            )
        else:
            self.attention_block = None
            self.classify_attention = None

    def make_layer(self, block_list):
        blocks = []
        for out_chan, stride in block_list:
            blocks.append(ResidualBlock3D(self.in_chan, out_chan, stride))
            self.in_chan = out_chan
        return nn.Sequential(*blocks)

    def forward(self, x, return_attention_map=False):

        conv1_out = self.conv1(x)
        conv2_out = self.conv2(conv1_out)
        conv3_out = self.conv3(conv2_out)

        out = self.classify(conv3_out)

        if self.use_attention:
            attended_conv1 = self.attention_block(conv1_out, conv3_out, return_attention_map)

            if return_attention_map:
                attended_conv1, att_map = attended_conv1

            out = (out + self.classify_attention(attended_conv1)) / 2

        if return_attention_map:
            return out, att_map
        else:
            return out
