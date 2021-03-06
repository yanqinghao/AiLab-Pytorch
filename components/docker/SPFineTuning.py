# coding=utf-8
from __future__ import absolute_import, print_function

import suanpan
from suanpan.arguments import ListOfString
from suanpan.log import logger
from suanpan.app import app
from args import PytorchLayersModel, PytorchFinetuningModel


@app.input(PytorchFinetuningModel(key="inputModel1"))
@app.input(PytorchLayersModel(key="inputModel2"))
@app.param(ListOfString(key="fineTuning", default=None))
@app.param(ListOfString(key="freezeParam", default=None))
@app.output(PytorchLayersModel(key="outputModel"))
def SPFineTuning(context):
    args = context.args
    model = args.inputModel2
    pretrainedFineTuning = args.inputModel1
    if args.fineTuning:
        for name, param in model.named_parameters():
            if ".".join(name.split(".")[:-1]) in args.fineTuning:
                logger.info("{} layer unfreezed.".format(name))
                param.requires_grad = True
    if args.freezeParam:
        for name, param in model.named_parameters():
            if ".".join(name.split(".")[:-1]) in args.freezeParam:
                logger.info("{} layer freezed.".format(name))
                param.requires_grad = False
    if pretrainedFineTuning:
        if "value" in pretrainedFineTuning.keys():
            if pretrainedFineTuning["value"]:
                fineTuning = (
                    pretrainedFineTuning["value"]
                    if pretrainedFineTuning["type"] == "hastop"
                    else [i.split(".")[-1] for i in pretrainedFineTuning["value"]]
                )
                for name, param in model.named_parameters():
                    if ".".join(name.split(".")[1:-1]) in fineTuning:
                        logger.info("{} layer unfreezed.".format(name))
                        param.requires_grad = True
    return model


if __name__ == "__main__":
    suanpan.run(app)
