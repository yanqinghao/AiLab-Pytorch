# coding=utf-8
from __future__ import absolute_import, print_function

import torch.nn as nn
import suanpan
from suanpan.app.arguments import Int, String, Float, Bool
from suanpan.app import app
from args import PytorchLayersModel
from utils import getLayerName


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
    args = context.args
    model = args.inputModel
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
    model.layers[name] = getattr(model, name)
    return model


if __name__ == "__main__":
    suanpan.run(app)
