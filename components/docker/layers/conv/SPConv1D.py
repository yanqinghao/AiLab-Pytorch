"""
Created on Sun Aug 22 2019
@author: Yan Qinghao
layers
"""
# coding=utf-8
from __future__ import absolute_import, print_function

import torch.nn as nn
from suanpan.app.arguments import Int, String, Bool
from app import app
from arguments import PytorchLayersModel
from utils import getLayerName, plotLayers, calOutput, getScreenshotPath


@app.input(PytorchLayersModel(key="inputModel"))
@app.param(Int(key="inChannel", default=1))
@app.param(Int(key="outChannel", default=16))
@app.param(Int(key="kernelSize", default=5))
@app.param(Int(key="stride", default=1))
@app.param(Int(key="padding", default=2))
@app.param(String(key="paddingMode", default="zeros"))
@app.param(Int(key="dilation", default=1))
@app.param(Int(key="groups", default=1))
@app.param(Bool(key="bias", default=True))
@app.output(PytorchLayersModel(key="outputModel"))
def SPConv1D(context):
    """
    Applies a 1D convolution over an input signal composed of several input planes.
    """
    args = context.args
    # 查看上一节点发送的 args.inputData 数据
    model = args.inputModel
    inputSize = calOutput(model)
    name = getLayerName(model.layers, "Conv1D")
    setattr(
        model,
        name,
        nn.Conv1d(
            args.inChannel,
            args.outChannel,
            kernel_size=args.kernelSize,
            stride=args.stride,
            padding=args.padding,
            padding_mode=args.paddingMode,
            dilation=args.dilation,
            groups=args.groups,
            bias=args.bias,
        ),
    )
    model.layers[name] = (getattr(model, name), getScreenshotPath())
    plotLayers(model, inputSize)

    return model


if __name__ == "__main__":
    SPConv1D()
