# coding=utf-8
from __future__ import absolute_import, print_function

import suanpan
from suanpan.app.arguments import Int
from suanpan.app import app
from args import PytorchLayersModel
from utils import getLayerName, net


@app.input(PytorchLayersModel(key="inputModel"))
@app.param(Int(key="inChannels", default=64))
@app.param(Int(key="outChannels", default=64))
@app.param(Int(key="stride", default=1, help="1,2"))
@app.param(Int(key="expandRatio", default=1))
@app.output(PytorchLayersModel(key="outputModel"))
def SPInvertedResidual(context):
    args = context.args
    model = args.inputModel
    name = getLayerName(model.layers, "InvertedResidual")
    setattr(
        model,
        name,
        net.InvertedResidual(args.inChannels, args.outChannels, args.stride, args.expandRatio),
    )
    model.layers[name] = getattr(model, name)
    return model


if __name__ == "__main__":
    suanpan.run(app)
