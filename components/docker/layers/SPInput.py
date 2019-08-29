# coding=utf-8
from __future__ import absolute_import, print_function

from suanpan.app.arguments import Folder, ListOfInt
from suanpan.log import logger
from app import app
from arguments import PytorchLayersModel, SPNet
from utils import plotLayers, getScreenshotPath


@app.input(Folder(key="inputData"))
@app.param(ListOfInt(key="inputSize", default=[28, 28, 1]))
@app.output(PytorchLayersModel(key="outputModel"))
def SPInput(context):
    args = context.args
    if len(args.inputSize) > 3:
        logger.error("Wrong image size, pls input like (28,28,1).")
    model = SPNet(args.inputSize)
    model.layers["Input Layer"] = (None, getScreenshotPath())
    plotLayers(model)
    return model


if __name__ == "__main__":
    SPInput()
