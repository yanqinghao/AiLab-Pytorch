"""
Created on Sun Aug 19 2019
@author: Yan Qinghao
transforms
"""
# coding=utf-8
from __future__ import absolute_import, print_function

import torchvision.transforms as transforms
import suanpan
from suanpan.app.arguments import Int, Folder, ListOfFloat, Bool, Float
from suanpan.app import app
from args import PytorchTransModel, PytorchDataset
from utils import transImgSave, mkFolder


@app.input(PytorchDataset(key="inputData"))
@app.param(Float(key="degrees", default=28, help="Range of degrees to select from."))
@app.param(
    Int(
        key="resample",
        default=0,
        help="An optional resampling filter. BILINEAR 2, NEAREST 0, BICUBIC 3",
    ))
@app.param(Bool(key="expand", default=False, help="Optional expansion flag."))
@app.param(ListOfFloat(key="center", default=None, help="Optional center of rotation."))
@app.output(PytorchTransModel(key="outputModel"))
@app.output(Folder(key="outputData"))
def SPRandomRotation(context):
    """
    Rotate the image by angle.
    """
    args = context.args
    transform = transforms.RandomRotation(
        args.degrees,
        resample=args.resample,
        expand=args.resample,
        center=(*args.center, ) if args.center else args.center,
    )
    folder = transImgSave(args.inputData, transform) if args.inputData else mkFolder()
    return transform, folder


if __name__ == "__main__":
    suanpan.run(app)
