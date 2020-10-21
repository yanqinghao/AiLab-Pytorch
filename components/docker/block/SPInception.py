# coding=utf-8
from __future__ import absolute_import, print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import suanpan
from suanpan.app.arguments import Int
from suanpan.app import app
from args import PytorchLayersModel
from utils import getLayerName, net


@app.input(PytorchLayersModel(key="inputModel"))
@app.param(Int(key="inChannels", default=192))
@app.param(Int(key="ch1x1", default=64))
@app.param(Int(key="ch3x3red", default=96))
@app.param(Int(key="ch3x3", default=128))
@app.param(Int(key="ch5x5red", default=16))
@app.param(Int(key="ch5x5", default=32))
@app.param(Int(key="poolProj", default=32))
@app.output(PytorchLayersModel(key="outputModel"))
def SPInception(context):
    args = context.args
    model = args.inputModel
    name = getLayerName(model.layers, "Inception")
    setattr(
        model,
        name,
        net.Inception(
            args.inChannels,
            args.ch1x1,
            args.ch3x3red,
            args.ch3x3,
            args.ch5x5red,
            args.ch5x5,
            args.poolProj,
        ),
    )
    model.layers[name] = getattr(model, name)
    return model


if __name__ == "__main__":
    suanpan.run(app)
