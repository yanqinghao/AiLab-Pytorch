# coding=utf-8
from __future__ import absolute_import, print_function

import torchvision.models as models
from suanpan.app.arguments import String, Json
from suanpan.app import app
from arguments import PytorchLayersModel


@app.input(PytorchLayersModel(key="inputModel"))
@app.param(String(key="modelName", default="all"))
@app.output(Json(key="outputData"))
def SPLookupParam(context):
    """resnet18 alexnet vgg16 squeezenet1_0 densenet161 inception_v3 googlenet
    shufflenet_v2_x1_0 mobilenet_v2 resnext50_32x4d wide_resnet50_2 mnasnet1_0"""
    args = context.args
    model = args.inputModel
    if args.modelName != "all":
        features = getattr(getattr(models, args.modelName)(), features, None)
    paramsName = [
        ".".join(i[0].split(".")[:-1]) for i in list(model.named_parameters())
    ]
    paramsSimple = []
    for i in paramsName:
        if i not in paramsSimple:
            paramsSimple.append(i)
    if args.modelName == "all":
        paramsOut = paramsSimple
    else:
        paramsOut = []
        for layer in paramsSimple:
            if str(args.modelName).upper() in layer:
                if features:
                    if "features" not in layer and "classifier" not in layer:
                        paramsOut.append(".".join(["features"] + layer.split(".")[1:]))
                    else:
                        paramsOut.append(".".join(layer.split(".")[1:]))
                else:
                    paramsOut.append(".".join(layer.split(".")[1:]))

    return paramsOut


if __name__ == "__main__":
    SPLookupParam()
