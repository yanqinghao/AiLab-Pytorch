# coding=utf-8
from __future__ import absolute_import, print_function

import torch.nn as nn

from suanpan.app.arguments import Int, String, Float, Bool
from app import app
from arguments import PytorchLayersModel
from utils import getLayerName, plotLayers, calOutput, getScreenshotPath


@app.input(PytorchLayersModel(key="inputModel"))
@app.param(Int(key="numEmbeddings", default=10000))
@app.param(Int(key="embeddingDim", default=32))
@app.param(String(key="mode", default="mean"))
@app.param(Float(key="maxNorm", default=None))
@app.param(Float(key="normType", default=2.0))
@app.param(Bool(key="scaleGradByFreq", default=False))
@app.param(Bool(key="sparse", default=False))
@app.output(PytorchLayersModel(key="outputModel"))
def SPEmbeddingBag(context):
    # 从 Context 中获取相关数据
    args = context.args
    # 查看上一节点发送的 args.inputData 数据
    model = args.inputModel
    inputSize = calOutput(model)
    name = getLayerName(model.layers, "EmbeddingBag")
    setattr(
        model,
        name,
        nn.EmbeddingBag(
            args.numEmbeddings,
            args.embeddingDim,
            max_norm=args.maxNorm,
            norm_type=args.normType,
            scale_grad_by_freq=args.scaleGradByFreq,
            mode=args.mode,
            sparse=args.sparse,
            _weight=None,
        ),
    )
    model.layers[name] = (getattr(model, name), getScreenshotPath())
    plotLayers(model, inputSize)

    return model


if __name__ == "__main__":
    SPEmbeddingBag()
