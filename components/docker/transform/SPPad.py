"""
Created on Sun Aug 19 2019
@author: Yan Qinghao
transforms
"""
# coding=utf-8
from __future__ import absolute_import, print_function

import torchvision.transforms as transforms
import suanpan
from suanpan.app.arguments import Int, Folder, String
from suanpan.app import app
from args import PytorchTransModel, PytorchDataset
from utils import transImgSave, mkFolder


@app.input(PytorchDataset(key="inputData"))
@app.param(Int(key="padding", default=1, help="Padding on each border."))
@app.param(Int(key="fill", default=0, help="Pixel fill value for constant fill."))
@app.param(
    String(
        key="paddingMode",
        default="constant",
        help="Type of padding.constant, edge, reflect, symmetric",
    ))
@app.output(PytorchTransModel(key="outputModel"))
@app.output(Folder(key="outputData"))
def SPPad(context):
    """
    Pad the given PIL Image on all sides with the given “pad” value.
    """
    args = context.args
    transform = transforms.Pad(args.padding, fill=args.fill, padding_mode=args.paddingMode)
    folder = transImgSave(args.inputData, transform) if args.inputData else mkFolder()
    return transform, folder


if __name__ == "__main__":
    suanpan.run(app)
