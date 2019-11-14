# coding=utf-8
from __future__ import absolute_import, print_function

from suanpan.app.arguments import Folder, ListOfInt, String
from suanpan.log import logger
from app import app
from arguments import PytorchLayersModel, SPNet
from utils import plotLayers, getScreenshotPath


@app.input(Folder(key="inputData"))
@app.param(String(key="type", default="image", help="image, text"))
@app.param(ListOfInt(key="inputSize"))
@app.output(PytorchLayersModel(key="outputModel"))
def SPInput(context):
    args = context.args
    model = SPNet(args.inputSize, args.type)
    model.layers["Input"] = (None, getScreenshotPath())
    if len(args.inputSize) > 3:
        logger.error("Wrong image size, pls input like (28,28,1).")
    plotLayers(model)
    return model


if __name__ == "__main__":
    SPInput()
