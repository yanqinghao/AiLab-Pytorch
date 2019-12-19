# coding=utf-8
from __future__ import absolute_import, print_function

import torch.nn as nn
from suanpan.app.arguments import Int, Bool
from app import app
from arguments import PytorchLayersModel
from utils import getLayerName, plotLayers, calOutput, getScreenshotPath


class VGGBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        conv2d_ks,
        padding,
        maxpool_ks,
        maxpool_stride,
        batch_norm=True,
        max_pool=True,
    ):
        super(VGGBlock, self).__init__()
        layers = []
        conv2d = nn.Conv2d(
            in_channels, out_channels, kernel_size=conv2d_ks, padding=padding
        )
        if batch_norm:
            layers += [conv2d, nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)]
        else:
            layers += [conv2d, nn.ReLU(inplace=True)]
        if max_pool:
            layers += [nn.MaxPool2d(kernel_size=maxpool_ks, stride=maxpool_stride)]
        self.layer = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer(x)
        return x


@app.input(PytorchLayersModel(key="inputModel"))
@app.param(Int(key="inChannel", default=1))
@app.param(Int(key="outChannel", default=16))
@app.param(Int(key="kernelSize", default=3))
@app.param(Int(key="padding", default=1))
@app.param(Int(key="maxpoolKS", default=2))
@app.param(Int(key="maxpoolStride", default=2))
@app.param(Bool(key="batchNorm", default=True))
@app.param(Bool(key="maxPool", default=True))
@app.output(PytorchLayersModel(key="outputModel"))
def SPVGGBlock(context):
    # 从 Context 中获取相关数据
    args = context.args
    # 查看上一节点发送的 args.inputData 数据
    model = args.inputModel
    inputSize = calOutput(model)
    name = getLayerName(model.layers, "VGGBlock")
    setattr(
        model,
        name,
        VGGBlock(
            args.inChannel,
            args.outChannel,
            args.kernelSize,
            args.padding,
            args.maxpoolKS,
            args.maxpoolStride,
            args.batchNorm,
            args.maxPool,
        ),
    )
    model.layers[name] = (getattr(model, name), getScreenshotPath())
    plotLayers(model, inputSize)

    return model


if __name__ == "__main__":
    SPVGGBlock()
