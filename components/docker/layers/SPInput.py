# coding=utf-8
from __future__ import absolute_import, print_function

import suanpan
from suanpan.app.arguments import ListOfInt, String
from suanpan.log import logger
from suanpan.app import app
from args import PytorchLayersModel
from utils.net import SPNet


@app.param(String(key="type", default="image", help="image, text"))
@app.param(ListOfInt(key="inputSize"))
@app.output(PytorchLayersModel(key="outputModel"))
def SPInput(context):
    args = context.args
    model = SPNet(args.inputSize, args.type)
    if args.type == "image" and len(args.inputSize) > 3:
        logger.error("Wrong image size, pls input like (28,28,1).")
    return model


if __name__ == "__main__":
    suanpan.run(app)
