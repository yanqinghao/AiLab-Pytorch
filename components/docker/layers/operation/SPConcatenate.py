# coding=utf-8
from __future__ import absolute_import, print_function

from suanpan.app.arguments import Int
from suanpan.log import logger
from app import app
from arguments import PytorchLayersModel, SPMathOP


@app.input(PytorchLayersModel(key="inputModel1"))
@app.input(PytorchLayersModel(key="inputModel2"))
@app.input(PytorchLayersModel(key="inputModel3"))
@app.input(PytorchLayersModel(key="inputModel4"))
@app.input(PytorchLayersModel(key="inputModel5"))
@app.param(Int(key="dim"))
@app.output(PytorchLayersModel(key="outputModel"))
def SPConcatenate(context):
    args = context.args
    modelList = []
    for i in range(5):
        if getattr(args, "inputModel" + str(i + 1)):
            modelList.append(getattr(args, "inputModel" + str(i + 1)))
    if len(modelList) < 2:
        logger.error("input must have one more models.")
        raise ("expect more input")
    inputCheck = set([model.layers["Input"][1] for model in modelList])
    if len(inputCheck) > 1:
        logger.error("support one input node for now")
        raise ("expect one input node")
    param = {"dim": args.dim}
    model = SPMathOP(
        modelList[0].input_size, modelList[0].task, modelList, "cat", param=param
    )
    return model


if __name__ == "__main__":
    SPConcatenate()
