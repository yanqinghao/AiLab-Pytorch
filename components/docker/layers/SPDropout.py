# coding=utf-8
from __future__ import absolute_import, print_function

import torch.nn as nn

from suanpan.app.arguments import Int, Bool
from app import app
from arguments import PytorchLayersModel
from utils import getLayerName, plotLayers, calOutput


@app.input(PytorchLayersModel(key="inputModel"))
@app.param(Int(key="p", default=0.5))
@app.param(Bool(key="inplace", default=False))
@app.output(PytorchLayersModel(key="outputModel"))
def SPDropout(context):
    # 从 Context 中获取相关数据
    args = context.args
    # 查看上一节点发送的 args.inputData 数据
    model = args.inputModel
    inputSize = calOutput(model)
    name = getLayerName(model.layers, "Dropout")
    setattr(model, name, nn.Dropout(p=args.p, inplace=args.inplace))
    model.layers.append((name, getattr(model, name)))
    plotLayers(model, inputSize)

    return model


if __name__ == "__main__":
    SPDropout()
