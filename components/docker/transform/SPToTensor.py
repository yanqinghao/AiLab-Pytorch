# coding=utf-8
from __future__ import absolute_import, print_function

import torchvision.transforms as transforms
import suanpan
from suanpan.app import app
from suanpan.app.arguments import Folder
from args import PytorchTransModel


@app.input(Folder(key="inputData"))
@app.output(PytorchTransModel(key="outputModel"))
def SPToTensor(context):
    transform = transforms.ToTensor()
    return transform


if __name__ == "__main__":
    suanpan.run(app)
