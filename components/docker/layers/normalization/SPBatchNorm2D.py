# coding=utf-8
from __future__ import absolute_import, print_function

import torch.nn as nn

import suanpan
from suanpan.app.arguments import Int
from suanpan.app import app
from args import PytorchLayersModel
from utils import getLayerName


@app.input(PytorchLayersModel(key="inputModel"))
@app.param(Int(key="numFeatures", default=32))
@app.output(PytorchLayersModel(key="outputModel"))
def SPBatchNorm2D(context):
    args = context.args
    model = args.inputModel
    name = getLayerName(model.layers, "BatchNorm2D")
    setattr(model, name, nn.BatchNorm2d(args.numFeatures))
    model.layers[name] = getattr(model, name)
    return model


if __name__ == "__main__":
    suanpan.run(app)
