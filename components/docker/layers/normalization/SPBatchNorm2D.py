# coding=utf-8
from __future__ import absolute_import, print_function

import torch.nn as nn

from suanpan.app.arguments import Int
from app import app
from arguments import PytorchLayersModel
from utils import getLayerName, plotLayers, calOutput


@app.input(PytorchLayersModel(key="inputModel"))
@app.param(Int(key="numFeatures", default=32))
@app.output(PytorchLayersModel(key="outputModel"))
def SPBatchNorm2D(context):
    # 从 Context 中获取相关数据
    args = context.args
    # 查看上一节点发送的 args.inputData 数据
    model = args.inputModel
    inputSize = calOutput(model)
    name = getLayerName(model.layers, "BatchNorm2D")
    setattr(model, name, nn.BatchNorm2d(args.numFeatures))
    model.layers.append((name, getattr(model, name)))
    plotLayers(model, inputSize)

    return model


if __name__ == "__main__":
    SPBatchNorm2D()
