# coding=utf-8
from __future__ import absolute_import, print_function

import suanpan
from suanpan.error import NodeError
from suanpan.app import app
from args import PytorchLayersModel
from utils import net


@app.input(PytorchLayersModel(key="inputModel1"))
@app.input(PytorchLayersModel(key="inputModel2"))
@app.output(PytorchLayersModel(key="outputModel"))
def SPAdd(context):
    args = context.args
    modelList = []
    for i in range(2):
        if getattr(args, "inputModel" + str(i + 1)):
            modelList.append(getattr(args, "inputModel" + str(i + 1)))
    if len(modelList) < 2:
        NodeError("expect more input")
    inputCheck = set([model.layers["Input"][1] for model in modelList])
    if len(inputCheck) > 1:
        NodeError("expect one input node")
    model = net.SPMathOP(modelList[0].input_size, modelList[0].task, modelList, "add")
    model.layers["Input"] = modelList[0].layers["Input"]
    return model


if __name__ == "__main__":
    suanpan.run(app)
