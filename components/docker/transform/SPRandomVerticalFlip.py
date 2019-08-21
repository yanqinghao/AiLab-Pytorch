"""
Created on Sun Aug 19 2019
@author: Yan Qinghao
transforms
"""
# coding=utf-8
from __future__ import absolute_import, print_function

import torchvision.transforms as transforms

from suanpan.app.arguments import Folder, Float
from app import app
from arguments import PytorchTransModel, PytorchDataset
from utils import transImgSave, mkFolder


@app.input(PytorchDataset(key="inputData"))
@app.param(Float(key="p", default=0.5, help=" probability of the image being flipped."))
@app.output(PytorchTransModel(key="outputModel"))
@app.output(Folder(key="outputData"))
def SPRandomVerticalFlip(context):
    """
    Vertically flip the given PIL Image randomly with a given probability.
    """
    args = context.args

    transform = transforms.RandomVerticalFlip(p=args.p)
    folder = transImgSave(args.inputData, transform) if args.inputData else mkFolder()

    return transform, folder


if __name__ == "__main__":
    SPRandomVerticalFlip()
