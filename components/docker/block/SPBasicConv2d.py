# coding=utf-8
from __future__ import absolute_import, print_function

import torch.nn as nn
import torch.nn.functional as F
import suanpan
from suanpan.app.arguments import Int, Bool
from suanpan.app import app
from args import PytorchLayersModel
from utils import getLayerName


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


@app.input(PytorchLayersModel(key="inputModel"))
@app.param(Int(key="inChannels", default=64))
@app.param(Int(key="outChannels", default=64))
@app.param(Int(key="kernelSize", default=3))
@app.param(Int(key="stride", default=1))
@app.param(Int(key="padding", default=1))
@app.param(Bool(key="bias", default=True))
@app.output(PytorchLayersModel(key="outputModel"))
def SPBasicConv2d(context):
    args = context.args
    model = args.inputModel
    name = getLayerName(model.layers, "BasicConv2d")
    setattr(
        model,
        name,
        BasicConv2d(
            args.inChannels,
            args.outChannels,
            bias=args.bias,
            kernel_size=args.kernelSize,
            stride=args.stride,
            padding=args.padding,
        ),
    )
    model.layers[name] = getattr(model, name)
    return model


if __name__ == "__main__":
    suanpan.run(app)
