# coding=utf-8
from __future__ import absolute_import, print_function

import torch.nn as nn
from suanpan.app.arguments import Int
from app import app
from arguments import PytorchLayersModel
from utils import getLayerName, plotLayers, calOutput, getScreenshotPath


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size,
                stride,
                padding,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True),
        )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        layers.extend(
            [
                # dw
                ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            ]
        )
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


@app.input(PytorchLayersModel(key="inputModel"))
@app.param(Int(key="inChannels", default=64))
@app.param(Int(key="outChannels", default=64))
@app.param(Int(key="stride", default=1, help="1,2"))
@app.param(Int(key="expandRatio", default=1))
@app.output(PytorchLayersModel(key="outputModel"))
def SPInvertedResidual(context):
    # 从 Context 中获取相关数据
    args = context.args
    # 查看上一节点发送的 args.inputData 数据
    model = args.inputModel
    inputSize = calOutput(model)
    name = getLayerName(model.layers, "InvertedResidual")
    setattr(
        model,
        name,
        InvertedResidual(
            args.inChannels, args.outChannels, args.stride, args.expandRatio
        ),
    )
    model.layers[name] = (getattr(model, name), getScreenshotPath())
    plotLayers(model, inputSize)

    return model


if __name__ == "__main__":
    SPInvertedResidual()
