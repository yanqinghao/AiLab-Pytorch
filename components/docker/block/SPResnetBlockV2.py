# coding=utf-8
from __future__ import absolute_import, print_function

import torch.nn as nn
from suanpan.app.arguments import Int
from app import app
from arguments import PytorchLayersModel
from utils import getLayerName, plotLayers, calOutput, getScreenshotPath


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class ResnetBlockV2(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, groups=1, base_width=64, dilation=1):
        super(ResnetBlockV2, self).__init__()
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride
        downsample = None
        if stride != 1 or inplanes != planes * self.expansion:
            downsample = nn.Sequential(
                conv1x1(inplanes, planes * self.expansion, stride),
                nn.BatchNorm2d(planes * self.expansion),
            )
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)

        return out


@app.input(PytorchLayersModel(key="inputModel"))
@app.param(Int(key="inplanes", default=64))
@app.param(Int(key="planes", default=64))
@app.output(PytorchLayersModel(key="outputModel"))
def SPResnetBlockV2(context):
    # 从 Context 中获取相关数据
    args = context.args
    # 查看上一节点发送的 args.inputData 数据
    model = args.inputModel
    inputSize = calOutput(model)
    name = getLayerName(model.layers, "ResnetBlockV2")
    setattr(model, name, ResnetBlockV2(args.inplanes, args.planes))
    model.layers[name] = (getattr(model, name), getScreenshotPath())
    plotLayers(model, inputSize)

    return model


if __name__ == "__main__":
    SPResnetBlockV2()
