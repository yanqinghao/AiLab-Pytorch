"""
Created on Sun Aug 19 2019
@author: Yan Qinghao
transforms
"""
# coding=utf-8
from __future__ import absolute_import, print_function

import torchvision.transforms as transforms
import suanpan
from suanpan.app.arguments import Folder, Float
from suanpan.app import app
from args import PytorchTransModel, PytorchDataset
from utils import transImgSave, mkFolder


@app.input(PytorchDataset(key="inputData"))
@app.param(Float(key="brightness", default=0, help="How much to jitter brightness. "))
@app.param(Float(key="contrast", default=0, help="How much to jitter contrast. "))
@app.param(Float(key="saturation", default=0, help="How much to jitter saturation. "))
@app.param(Float(key="hue", default=0, help="How much to jitter hue. "))
@app.output(PytorchTransModel(key="outputModel"))
@app.output(Folder(key="outputData"))
def SPColorJitter(context):
    """
    Randomly change the brightness, contrast and saturation of an image.
    """
    args = context.args
    transform = transforms.ColorJitter(brightness=args.brightness,
                                       contrast=args.contrast,
                                       saturation=args.saturation,
                                       hue=args.hue)
    folder = transImgSave(args.inputData, transform) if args.inputData else mkFolder()
    return transform, folder


if __name__ == "__main__":
    suanpan.run(app)
