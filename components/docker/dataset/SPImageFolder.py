# coding=utf-8
from __future__ import absolute_import, print_function

from suanpan.app import app

from utils.folder import ImageFolder
import suanpan
from suanpan.arguments import String
from args import PytorchDataset, PytorchTransModel


@app.input(String(key="inputData"))
@app.input(PytorchTransModel(key="inputModel"))
@app.output(PytorchDataset(key="outputData"))
def SPImageFolder(context):
    args = context.args
    dataset = ImageFolder(oss_root=args.inputData, transform=args.inputModel)
    return dataset


if __name__ == "__main__":
    suanpan.run(app)
