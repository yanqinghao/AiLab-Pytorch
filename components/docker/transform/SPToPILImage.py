"""
Created on Sun Aug 19 2019
@author: Yan Qinghao
transforms
"""
# coding=utf-8
from __future__ import absolute_import, print_function

import torchvision.transforms as transforms

from suanpan.app import app
from suanpan.app.arguments import Folder, String
from arguments import PytorchTransModel
from utils import transImgSave, mkFolder


@app.input(Folder(key="inputData"))
@app.param(
    String(
        key="mode",
        default=None,
        help="color space and pixel depth of input data (optional)."
        "1, L, P, RGB, RGBA, CMYK, YCbCr, LAB, HSV, I, F",
    )
)
@app.output(PytorchTransModel(key="outputModel"))
def SPToPILImage(context):
    """
    Convert a tensor or an ndarray to PIL Image.
    """
    args = context.args

    transform = transforms.ToPILImage(mode=args.mode)
    folder = transImgSave(args.inputData, transform) if args.inputData else mkFolder()

    return transform, folder


if __name__ == "__main__":
    SPToPILImage()
