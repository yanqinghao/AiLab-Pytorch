"""
Created on Sun Aug 19 2019
@author: Yan Qinghao
transforms
"""
# coding=utf-8
from __future__ import absolute_import, print_function

import torchvision.transforms as transforms

from suanpan.app.arguments import Int, Folder
from app import app
from arguments import PytorchTransModel, PytorchDataset
from utils import transImgSave, mkFolder


@app.input(PytorchDataset(key="inputData"))
@app.param(Int(key="size", default=28, help="Desired output size of the crop."))
@app.output(PytorchTransModel(key="outputModel"))
@app.output(Folder(key="outputData"))
def SPCenterCrop(context):
    """
    Crops the given PIL Image at the center.
    """
    args = context.args

    transform = transforms.CenterCrop(args.size)
    folder = transImgSave(args.inputData, transform) if args.inputData else mkFolder()

    return transform, folder


if __name__ == "__main__":
    SPCenterCrop()
