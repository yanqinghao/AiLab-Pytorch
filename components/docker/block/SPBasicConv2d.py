# coding=utf-8
from __future__ import absolute_import, print_function

import torch.nn as nn
import torch.nn.functional as F
from suanpan.app.arguments import Int
from app import app
from arguments import PytorchLayersModel
from utils import getLayerName, plotLayers, calOutput, getScreenshotPath


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


@app.input(PytorchLayersModel(key="inputModel"))
@app.param(Int(key="inChannels", default=64))
@app.param(Int(key="outChannels", default=64))
@app.output(PytorchLayersModel(key="outputModel"))
def SPBasicConv2d(context):
    # 从 Context 中获取相关数据
    args = context.args
    # 查看上一节点发送的 args.inputData 数据
    model = args.inputModel
    inputSize = calOutput(model)
    name = getLayerName(model.layers, "BasicConv2d")
    setattr(model, name, BasicConv2d(args.inChannels, args.outChannels))
    model.layers[name] = (getattr(model, name), getScreenshotPath())
    plotLayers(model, inputSize)

    return model


if __name__ == "__main__":
    SPBasicConv2d()
