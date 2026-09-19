"""OSNet — Omni-Scale Network for person re-identification
(Zhou et al., ICCV 2019), implemented in plain PyTorch so it can be used
for inference without the torchreid training framework and its dependency
chain.

Module / parameter names match KaiyangZhou's reference implementation so
the published pretrained state_dicts (osnet_x1_0, osnet_x0_75,
osnet_x0_5, osnet_x0_25 — trained on Market-1501 / MSMT17 / DukeMTMC) load
directly:

    net = osnet_x1_0(num_classes=1000)
    net.load_state_dict(torch.load("osnet_x1_0_market.pth", map_location="cpu"), strict=False)

Use vision.reid.backends.OSNetBackend rather than this module directly.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ConvLayer(nn.Module):
    def __init__(self, in_c, out_c, k, stride=1, padding=0, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, k, stride=stride, padding=padding, bias=False, groups=groups)
        self.bn = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class Conv1x1(nn.Module):
    def __init__(self, in_c, out_c, stride=1, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, 1, stride=stride, padding=0, bias=False, groups=groups)
        self.bn = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class Conv1x1Linear(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, 1, stride=stride, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(out_c)

    def forward(self, x):
        return self.bn(self.conv(x))


class LightConv3x3(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 1, stride=1, padding=0, bias=False)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, stride=1, padding=1, bias=False, groups=out_c)
        self.bn = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv2(self.conv1(x))))


class ChannelGate(nn.Module):
    def __init__(self, in_c, num_gates=None, reduction=16):
        super().__init__()
        num_gates = num_gates or in_c
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_c, in_c // reduction, 1, bias=True, padding=0)
        self.norm1 = nn.BatchNorm2d(in_c // reduction)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(in_c // reduction, num_gates, 1, bias=True, padding=0)
        self.gate_activation = nn.Sigmoid()

    def forward(self, x):
        inp = x
        x = self.global_avgpool(x)
        x = self.relu(self.norm1(self.fc1(x)))
        x = self.gate_activation(self.fc2(x))
        return inp * x


class OSBlock(nn.Module):
    def __init__(self, in_c, out_c, reduction=4, T=4):
        super().__init__()
        mid_c = out_c // reduction
        self.conv1 = Conv1x1(in_c, mid_c)
        self.conv2 = nn.ModuleList()
        for t in range(1, T + 1):
            layers = [LightConv3x3(mid_c, mid_c) for _ in range(t)]
            self.conv2.append(nn.Sequential(*layers))
        self.gate = ChannelGate(mid_c)
        self.conv3 = Conv1x1Linear(mid_c, out_c)
        self.downsample = None
        if in_c != out_c:
            self.downsample = Conv1x1Linear(in_c, out_c)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        x1 = self.conv1(x)
        x2 = 0
        for conv2_t in self.conv2:
            x2 = x2 + self.gate(conv2_t(x1))
        x3 = self.conv3(x2)
        if self.downsample is not None:
            identity = self.downsample(identity)
        return self.relu(x3 + identity)


class OSNet(nn.Module):
    def __init__(self, blocks_per_stage=(2, 2, 2), channels=(64, 256, 384, 512), feature_dim=512, num_classes=1000):
        super().__init__()
        self.conv1 = ConvLayer(3, channels[0], 7, stride=2, padding=3)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        self.conv2 = self._make_stage(channels[0], channels[1], blocks_per_stage[0], reduce_spatial=True)
        self.conv3 = self._make_stage(channels[1], channels[2], blocks_per_stage[1], reduce_spatial=True)
        self.conv4 = self._make_stage(channels[2], channels[3], blocks_per_stage[2], reduce_spatial=False)
        self.conv5 = Conv1x1(channels[3], channels[3])

        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = self._make_fc(channels[3], feature_dim)
        self.classifier = nn.Linear(feature_dim, num_classes)
        self.feature_dim = feature_dim

    def _make_stage(self, in_c, out_c, n_blocks, reduce_spatial):
        layers = [OSBlock(in_c, out_c)]
        layers += [OSBlock(out_c, out_c) for _ in range(n_blocks - 1)]
        if reduce_spatial:
            layers += [Conv1x1(out_c, out_c), nn.AvgPool2d(2, stride=2)]
        return nn.Sequential(*layers)

    def _make_fc(self, in_c, out_c):
        return nn.Sequential(
            nn.Linear(in_c, out_c),
            nn.BatchNorm1d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, return_features=True):
        x = self.maxpool(self.conv1(x))
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        v = self.global_avgpool(x).flatten(1)
        v = self.fc(v)
        if return_features:
            return v
        return self.classifier(v)


def osnet_x1_0(num_classes=1000, feature_dim=512):
    return OSNet(blocks_per_stage=(2, 2, 2), channels=(64, 256, 384, 512), feature_dim=feature_dim, num_classes=num_classes)


def osnet_x0_25(num_classes=1000, feature_dim=512):
    return OSNet(blocks_per_stage=(2, 2, 2), channels=(16, 64, 96, 128), feature_dim=feature_dim, num_classes=num_classes)
