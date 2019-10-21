# coding=utf-8
from __future__ import absolute_import, print_function

import pickle
from suanpan.app.arguments import Bool, ListOfString, String
from suanpan.log import logger
import torchvision.models as models
from app import app
from arguments import PytorchLayersModel, PytorchFinetuningModel
from utils import (
    getLayerName,
    plotLayers,
    calOutput,
    getScreenshotPath,
    downloadPretrained,
)


@app.input(PytorchLayersModel(key="inputModel"))
@app.param(String(key="modelName", default="vgg19"))
@app.param(Bool(key="pretrained", default=True))
@app.param(Bool(key="featureExtractor", default=False))
@app.param(Bool(key="requiresGrad", default=False))
@app.param(ListOfString(key="fineTuning", default=None))
@app.param(String(key="storageType", default="oss"))
@app.output(PytorchLayersModel(key="outputModel1"))
@app.output(PytorchFinetuningModel(key="outputModel2"))
def SPModels(context):
    """resnet18 alexnet vgg16 squeezenet1_0 densenet161 inception_v3 googlenet
    shufflenet_v2_x1_0 mobilenet_v2 resnext50_32x4d wide_resnet50_2 mnasnet1_0"""
    args = context.args
    model = args.inputModel
    inputSize = calOutput(model)
    if args.pretrained:
        downloadPretrained(args.modelName, args.storageType)
    layerName = getLayerName(model.layers, str(args.modelName).upper())
    pretrainedModel = (
        getattr(models, args.modelName)(pretrained=args.pretrained).features
        if args.featureExtractor
        else getattr(models, args.modelName)(pretrained=args.pretrained)
    )
    if not args.featureExtractor:
        with open("./utils/imagenet1000_clsid_to_human.pkl", "rb") as f:
            clsid_to_human = pickle.load(f)
        clsid_to_human[134] = "crane 1"
        clsid_to_human = dict(zip(clsid_to_human.values(), clsid_to_human.keys()))
        model.class_to_idx = clsid_to_human
    for name, param in pretrainedModel.named_parameters():
        isfreezed = "unfreezed" if args.requiresGrad else "freezed"
        logger.info("{} layer {}.".format(name, isfreezed))
        param.requires_grad = args.requiresGrad
    if args.fineTuning:
        fineTuning = (
            args.fineTuning
            if not args.featureExtractor
            else [i.split(".")[-1] for i in args.fineTuning]
        )
        for name, param in pretrainedModel.named_parameters():
            if ".".join(name.split(".")[:-1]) in fineTuning:
                logger.info("{} layer unfreezed.".format(name))
                param.requires_grad = True
    setattr(model, layerName, pretrainedModel)
    model.layers[layerName] = (getattr(model, layerName), getScreenshotPath())
    plotLayers(model, inputSize)
    finetuning = (
        {"name": layerName, "value": args.fineTuning, "type": "hastop"}
        if not args.featureExtractor
        else {"name": layerName, "value": args.fineTuning, "type": "notop"}
    )
    return model, finetuning


if __name__ == "__main__":
    SPModels()
