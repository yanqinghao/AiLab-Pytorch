# coding=utf-8
from __future__ import absolute_import, print_function

import torch.nn as nn

import suanpan
from suanpan.app.arguments import Int
from suanpan.app import app
from args import PytorchLayersModel
from utils import getLayerName


@app.input(PytorchLayersModel(key="inputModel"))
@app.param(Int(key="dim", default=None))
@app.output(PytorchLayersModel(key="outputModel"))
def SPLogSoftmax(context):
    args = context.args
    model = args.inputModel
    name = getLayerName(model.layers, "LogSoftmax")
    setattr(model, name, nn.LogSoftmax(dim=args.dim))
    model.layers[name] = getattr(model, name)
    return model


if __name__ == "__main__":
    suanpan.run(app)
