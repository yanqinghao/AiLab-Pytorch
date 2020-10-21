# coding=utf-8
from __future__ import absolute_import, print_function

import suanpan
from suanpan.app.arguments import Int, Bool
from suanpan.app import app
from args import PytorchLayersModel
from utils import getLayerName, net


@app.input(PytorchLayersModel(key="inputModel"))
@app.param(Int(key="inChannels", default=64))
@app.param(Int(key="outChannels", default=64))
@app.param(Int(key="kernelSize", default=3))
@app.param(Int(key="stride", default=1))
@app.param(Int(key="padding", default=1))
@app.param(Bool(key="bias", default=True))
@app.output(PytorchLayersModel(key="outputModel"))
def SPBasicConv2d(context):
    args = context.args
    model = args.inputModel
    name = getLayerName(model.layers, "BasicConv2d")
    setattr(
        model,
        name,
        net.BasicConv2d(
            args.inChannels,
            args.outChannels,
            bias=args.bias,
            kernel_size=args.kernelSize,
            stride=args.stride,
            padding=args.padding,
        ),
    )
    model.layers[name] = getattr(model, name)
    return model


if __name__ == "__main__":
    suanpan.run(app)
