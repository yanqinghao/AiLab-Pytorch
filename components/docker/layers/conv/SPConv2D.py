# coding=utf-8
from __future__ import absolute_import, print_function

import torch.nn as nn
import suanpan
from suanpan.app.arguments import Int, String, Bool
from suanpan.app import app
from args import PytorchLayersModel
from utils import getLayerName


@app.input(PytorchLayersModel(key="inputModel"))
@app.param(Int(key="inChannel", default=1))
@app.param(Int(key="outChannel", default=16))
@app.param(Int(key="kernelSize", default=5))
@app.param(Int(key="stride", default=1))
@app.param(Int(key="padding", default=2))
@app.param(String(key="paddingMode", default="zeros"))
@app.param(Int(key="dilation", default=1))
@app.param(Int(key="groups", default=1))
@app.param(Bool(key="bias", default=True))
@app.output(PytorchLayersModel(key="outputModel"))
def SPConv2D(context):
    args = context.args
    model = args.inputModel
    name = getLayerName(model.layers, "Conv2D")
    setattr(
        model,
        name,
        nn.Conv2d(
            args.inChannel,
            args.outChannel,
            kernel_size=args.kernelSize,
            stride=args.stride,
            padding=args.padding,
            padding_mode=args.paddingMode,
            dilation=args.dilation,
            groups=args.groups,
            bias=args.bias,
        ),
    )
    model.layers[name] = getattr(model, name)
    return model


if __name__ == "__main__":
    suanpan.run(app)
