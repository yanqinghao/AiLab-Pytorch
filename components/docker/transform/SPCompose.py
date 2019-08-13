# coding=utf-8
from __future__ import absolute_import, print_function

import torchvision.transforms as transforms

from suanpan.app import app
from arguments import PytorchTransModel


@app.input(PytorchTransModel(key="inputModel1"))
@app.input(PytorchTransModel(key="inputModel2"))
@app.input(PytorchTransModel(key="inputModel3"))
@app.input(PytorchTransModel(key="inputModel4"))
@app.input(PytorchTransModel(key="inputModel5"))
@app.output(PytorchTransModel(key="outputModel"))
def SPCompose(context):
    # 从 Context 中获取相关数据
    args = context.args
    transformLst = []
    for i in range(5):
        transform = getattr(args, "inputModel{}".format(i + 1))
        if transform:
            transformLst.append(transform)
    transformsAug = transforms.Compose(transformLst)

    return transformsAug


if __name__ == "__main__":
    SPCompose()
