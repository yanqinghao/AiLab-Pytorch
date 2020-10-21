# coding=utf-8
from __future__ import absolute_import, print_function

import suanpan
from suanpan.app.arguments import Int
from suanpan.app import app
from args import PytorchLayersModel
from utils import getLayerName, net


@app.input(PytorchLayersModel(key="inputModel"))
@app.param(Int(key="inplanes", default=64))
@app.param(Int(key="planes", default=64))
@app.output(PytorchLayersModel(key="outputModel"))
def SPResnetBlockV2(context):
    args = context.args
    model = args.inputModel
    name = getLayerName(model.layers, "ResnetBlockV2")
    setattr(model, name, net.ResnetBlockV2(args.inplanes, args.planes))
    model.layers[name] = getattr(model, name)
    return model


if __name__ == "__main__":
    suanpan.run(app)
