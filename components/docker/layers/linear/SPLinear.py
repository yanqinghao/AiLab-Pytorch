# coding=utf-8
from __future__ import absolute_import, print_function

import torch.nn as nn

import suanpan
from suanpan.app.arguments import Int, Bool
from suanpan.app import app
from args import PytorchLayersModel
from utils import getLayerName


@app.input(PytorchLayersModel(key="inputModel"))
@app.param(Int(key="inFeature", default=7 * 7 * 32))
@app.param(Int(key="outFeature", default=10))
@app.param(Bool(key="bias", default=True))
@app.output(PytorchLayersModel(key="outputModel"))
def SPLinear(context):
    args = context.args
    model = args.inputModel
    name = getLayerName(model.layers, "Linear")
    setattr(
        model,
        name,
        nn.Linear(in_features=args.inFeature, out_features=args.outFeature, bias=args.bias),
    )
    model.layers[name] = getattr(model, name)
    return model


if __name__ == "__main__":
    suanpan.run(app)
