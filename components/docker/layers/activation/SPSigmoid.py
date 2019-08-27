# coding=utf-8
from __future__ import absolute_import, print_function

import torch.nn as nn

from app import app
from arguments import PytorchLayersModel
from utils import getLayerName, plotLayers, calOutput


@app.input(PytorchLayersModel(key="inputModel"))
@app.output(PytorchLayersModel(key="outputModel"))
def SPSigmoid(context):
    # 从 Context 中获取相关数据
    args = context.args
    # 查看上一节点发送的 args.inputData 数据
    model = args.inputModel
    inputSize = calOutput(model)
    name = getLayerName(model.layers, "Sigmoid")
    setattr(model, name, nn.Sigmoid())
    model.layers.append((name, getattr(model, name)))
    plotLayers(model, inputSize)
    return model


if __name__ == "__main__":
    SPSigmoid()
