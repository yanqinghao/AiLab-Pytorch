# coding=utf-8
from __future__ import absolute_import, print_function

import torch.nn as nn

from suanpan.app.arguments import Int
from app import app
from arguments import PytorchLayersModel
from utils import getLayerName, plotLayers, calOutput, getScreenshotPath


@app.input(PytorchLayersModel(key="inputModel"))
@app.param(Int(key="dim", default=None))
@app.output(PytorchLayersModel(key="outputModel"))
def SPLogSoftmax(context):
    # 从 Context 中获取相关数据
    args = context.args
    # 查看上一节点发送的 args.inputData 数据
    model = args.inputModel
    inputSize = calOutput(model)
    name = getLayerName(model.layers, "LogSoftmax")
    setattr(model, name, nn.LogSoftmax(dim=args.dim))
    model.layers[name] = (getattr(model, name), getScreenshotPath())
    plotLayers(model, inputSize)
    return model


if __name__ == "__main__":
    SPLogSoftmax()
