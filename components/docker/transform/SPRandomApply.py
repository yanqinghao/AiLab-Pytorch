"""
Created on Sun Aug 19 2019
@author: Yan Qinghao
transforms
"""
# coding=utf-8
from __future__ import absolute_import, print_function

import torchvision.transforms as transforms
import suanpan
from suanpan.app.arguments import Float, Folder
from suanpan.app import app
from args import PytorchTransModel, PytorchDataset
from utils import transImgSave, mkFolder


@app.input(PytorchDataset(key="inputData"))
@app.input(PytorchTransModel(key="inputModel1"))
@app.input(PytorchTransModel(key="inputModel2"))
@app.input(PytorchTransModel(key="inputModel3"))
@app.input(PytorchTransModel(key="inputModel4"))
@app.input(PytorchTransModel(key="inputModel5"))
@app.param(Float(key="p", default=0.5))
@app.output(PytorchTransModel(key="outputModel"))
@app.output(Folder(key="outputData"))
def SPRandomApply(context):
    """
    Apply randomly a list of transformations with a given probability
    """
    args = context.args
    transformLst = []
    for i in range(5):
        transform = getattr(args, "inputModel{}".format(i + 1))
        if transform:
            transformLst.append(transform)
    transformsAug = transforms.RandomApply(transformLst, p=args.p)
    folder = transImgSave(args.inputData, transformsAug) if args.inputData else mkFolder()
    return transformsAug, folder


if __name__ == "__main__":
    suanpan.run(app)
