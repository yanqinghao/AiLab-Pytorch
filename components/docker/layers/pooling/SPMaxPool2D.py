# coding=utf-8
from __future__ import absolute_import, print_function

import torch.nn as nn
import suanpan
from suanpan.app.arguments import Int, Bool
from suanpan.app import app
from args import PytorchLayersModel
from utils import getLayerName


@app.input(PytorchLayersModel(key="inputModel"))
@app.param(Int(key="kernelSize", default=2))
@app.param(Int(key="stride", default=2))
@app.param(Int(key="padding", default=0))
@app.param(Int(key="dilation", default=1))
@app.param(Bool(key="returnIndices", default=False))
@app.param(Bool(key="ceilMode", default=False))
@app.output(PytorchLayersModel(key="outputModel"))
def SPMaxPool2D(context):
    args = context.args
    model = args.inputModel
    name = getLayerName(model.layers, "MaxPool2D")
    setattr(
        model,
        name,
        nn.MaxPool2d(
            kernel_size=args.kernelSize,
            stride=args.stride,
            padding=args.padding,
            dilation=args.dilation,
            return_indices=args.returnIndices,
            ceil_mode=args.ceilMode,
        ),
    )
    model.layers[name] = getattr(model, name)
    return model


if __name__ == "__main__":
    suanpan.run(app)
