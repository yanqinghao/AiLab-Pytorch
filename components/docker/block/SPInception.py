# coding=utf-8
from __future__ import absolute_import, print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import suanpan
from suanpan.app.arguments import Int
from suanpan.app import app
from args import PytorchLayersModel
from utils import getLayerName


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


class Inception(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
        super(Inception, self).__init__()

        self.branch1 = BasicConv2d(in_channels, ch1x1, kernel_size=1)

        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, ch3x3red, kernel_size=1),
            BasicConv2d(ch3x3red, ch3x3, kernel_size=3, padding=1),
        )

        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels, ch5x5red, kernel_size=1),
            BasicConv2d(ch5x5red, ch5x5, kernel_size=3, padding=1),
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=True),
            BasicConv2d(in_channels, pool_proj, kernel_size=1),
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        outputs = [branch1, branch2, branch3, branch4]
        return torch.cat(outputs, 1)


@app.input(PytorchLayersModel(key="inputModel"))
@app.param(Int(key="inChannels", default=192))
@app.param(Int(key="ch1x1", default=64))
@app.param(Int(key="ch3x3red", default=96))
@app.param(Int(key="ch3x3", default=128))
@app.param(Int(key="ch5x5red", default=16))
@app.param(Int(key="ch5x5", default=32))
@app.param(Int(key="poolProj", default=32))
@app.output(PytorchLayersModel(key="outputModel"))
def SPInception(context):
    args = context.args
    model = args.inputModel
    name = getLayerName(model.layers, "Inception")
    setattr(
        model,
        name,
        Inception(
            args.inChannels,
            args.ch1x1,
            args.ch3x3red,
            args.ch3x3,
            args.ch5x5red,
            args.ch5x5,
            args.poolProj,
        ),
    )
    model.layers[name] = getattr(model, name)
    return model


if __name__ == "__main__":
    suanpan.run(app)
