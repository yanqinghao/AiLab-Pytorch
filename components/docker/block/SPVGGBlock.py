# coding=utf-8
from __future__ import absolute_import, print_function

import suanpan
from suanpan.app.arguments import Int, Bool
from suanpan.app import app
from args import PytorchLayersModel
from utils import getLayerName, net


@app.input(PytorchLayersModel(key="inputModel"))
@app.param(Int(key="inChannel", default=1))
@app.param(Int(key="outChannel", default=16))
@app.param(Int(key="kernelSize", default=3))
@app.param(Int(key="padding", default=1))
@app.param(Int(key="maxpoolKS", default=2))
@app.param(Int(key="maxpoolStride", default=2))
@app.param(Bool(key="batchNorm", default=True))
@app.param(Bool(key="maxPool", default=True))
@app.output(PytorchLayersModel(key="outputModel"))
def SPVGGBlock(context):
    args = context.args
    model = args.inputModel
    name = getLayerName(model.layers, "VGGBlock")
    setattr(
        model,
        name,
        net.VGGBlock(
            args.inChannel,
            args.outChannel,
            args.kernelSize,
            args.padding,
            args.maxpoolKS,
            args.maxpoolStride,
            args.batchNorm,
            args.maxPool,
        ),
    )
    model.layers[name] = getattr(model, name)
    return model


if __name__ == "__main__":
    suanpan.run(app)
