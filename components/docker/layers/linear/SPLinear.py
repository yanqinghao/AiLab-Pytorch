# coding=utf-8
from __future__ import absolute_import, print_function

import torch.nn as nn

from suanpan.app.arguments import Int, Bool
from app import app
from arguments import PytorchLayersModel
from utils import getLayerName, plotLayers, calOutput, getScreenshotPath


@app.input(PytorchLayersModel(key="inputModel"))
@app.param(Int(key="inFeature", default=7 * 7 * 32))
@app.param(Int(key="outFeature", default=10))
@app.param(Bool(key="bias", default=True))
@app.output(PytorchLayersModel(key="outputModel"))
def SPLinear(context):
    # 从 Context 中获取相关数据
    args = context.args
    # 查看上一节点发送的 args.inputData 数据
    model = args.inputModel
    inputSize = calOutput(model)
    name = getLayerName(model.layers, "Linear")
    setattr(
        model,
        name,
        nn.Linear(
            in_features=args.inFeature, out_features=args.outFeature, bias=args.bias
        ),
    )
    model.layers[name] = (getattr(model, name), getScreenshotPath())
    plotLayers(model, inputSize)

    return model


if __name__ == "__main__":
    SPLinear()
