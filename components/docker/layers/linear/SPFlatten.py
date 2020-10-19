# coding=utf-8
from __future__ import absolute_import, print_function

import torch.nn as nn

import suanpan
from suanpan.app import app
from args import PytorchLayersModel
from utils import getLayerName


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size()[0], -1)


@app.input(PytorchLayersModel(key="inputModel"))
@app.output(PytorchLayersModel(key="outputModel"))
def SPFlatten(context):
    args = context.args
    model = args.inputModel
    name = getLayerName(model.layers, "Flatten")
    setattr(model, name, Flatten())
    model.layers[name] = getattr(model, name)
    return model


if __name__ == "__main__":
    suanpan.run(app)
