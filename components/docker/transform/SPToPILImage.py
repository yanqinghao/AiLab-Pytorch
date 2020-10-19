"""
Created on Sun Aug 19 2019
@author: Yan Qinghao
transforms
"""
# coding=utf-8
from __future__ import absolute_import, print_function

import torchvision.transforms as transforms
import suanpan
from suanpan.app import app
from suanpan.app.arguments import String, Folder
from args import PytorchTransModel


@app.input(Folder(key="inputData"))
@app.param(
    String(
        key="mode",
        default=None,
        help="color space and pixel depth of input data (optional)."
        "1, L, P, RGB, RGBA, CMYK, YCbCr, LAB, HSV, I, F",
    ))
@app.output(PytorchTransModel(key="outputModel"))
def SPToPILImage(context):
    """
    Convert a tensor or an ndarray to PIL Image.
    """
    args = context.args
    transform = transforms.ToPILImage(mode=args.mode)
    return transform


if __name__ == "__main__":
    suanpan.run(app)
