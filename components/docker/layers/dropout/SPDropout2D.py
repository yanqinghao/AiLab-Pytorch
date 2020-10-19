# coding=utf-8
from __future__ import absolute_import, print_function

import torch.nn as nn
import suanpan
from suanpan.app.arguments import Float, Bool
from suanpan.app import app
from args import PytorchLayersModel
from utils import getLayerName


@app.input(PytorchLayersModel(key="inputModel"))
@app.param(Float(key="p", default=0.5))
@app.param(Bool(key="inplace", default=False))
@app.output(PytorchLayersModel(key="outputModel"))
def SPDropout2D(context):
    args = context.args
    model = args.inputModel
    name = getLayerName(model.layers, "Dropout2D")
    setattr(model, name, nn.Dropout2d(p=args.p, inplace=args.inplace))
    model.layers[name] = getattr(model, name)
    return model


if __name__ == "__main__":
    suanpan.run(app)
