# coding=utf-8
from __future__ import absolute_import, print_function

import suanpan
from suanpan.app.arguments import Folder, String
from suanpan.app import app
from args import PytorchLayersModel, PytorchDataloader
from utils.visual import CNNLayerVisualization


@app.input(PytorchLayersModel(key="inputModel"))
@app.input(PytorchDataloader(key="inputLoader"))
@app.param(String(key="selectedLayer", default="Conv2D_0"))
@app.output(Folder(key="outputData"))
def SPCNNVisual(context):
    args = context.args
    model = args.inputModel
    selectedLayer = args.selectedLayer
    dataLoader = args.inputLoader
    cnnVisual = CNNLayerVisualization(model, selectedLayer)
    for data, _, paths in dataLoader:
        folder = cnnVisual.plot_layer(data, paths)
    return folder


if __name__ == "__main__":
    suanpan.run(app)
