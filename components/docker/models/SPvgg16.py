# coding=utf-8
from __future__ import absolute_import, print_function

from suanpan.app.arguments import Bool
import torchvision.models as models
from app import app
from arguments import PytorchLayersModel
from utils import (
    getLayerName,
    plotLayers,
    calOutput,
    getScreenshotPath,
    downloadPretrained,
)


@app.input(PytorchLayersModel(key="inputModel"))
@app.param(Bool(key="pretrained", default=True))
@app.param(Bool(key="featureExtractor", default=True))
@app.output(PytorchLayersModel(key="outputModel"))
def SPvgg16(context):
    """VGG16"""
    args = context.args
    model = args.inputModel
    inputSize = calOutput(model)
    if args.pretrained:
        downloadPretrained("vgg16")
    name = getLayerName(model.layers, "VGG16")
    pretrainedModel = (
        models.vgg16(pretrained=args.pretrained).features
        if args.featureExtractor
        else models.vgg16(pretrained=args.pretrained)
    )
    setattr(model, name, pretrainedModel)
    model.layers[name] = (getattr(model, name), getScreenshotPath())
    plotLayers(model, inputSize)

    return model


if __name__ == "__main__":
    SPvgg16()
