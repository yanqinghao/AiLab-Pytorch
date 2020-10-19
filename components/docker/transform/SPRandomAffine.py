"""
Created on Sun Aug 19 2019
@author: Yan Qinghao
transforms
"""
# coding=utf-8
from __future__ import absolute_import, print_function

import torchvision.transforms as transforms
import suanpan
from suanpan.app.arguments import Float, Folder, ListOfFloat, Int
from suanpan.app import app
from args import PytorchTransModel, PytorchDataset
from utils import transImgSave, mkFolder


@app.input(PytorchDataset(key="inputData"))
@app.param(Float(key="degrees", default=0, help=" Range of degrees to select from."))
@app.param(
    ListOfFloat(
        key="translate",
        default=None,
        help="tuple of maximum absolute fraction for horizontal and vertical translations.",
    ))
@app.param(ListOfFloat(key="scale", default=None, help="scaling factor interval"))
@app.param(Float(key="shear", default=None, help="Range of degrees to select from."))
@app.param(
    Int(
        key="resample",
        default=0,
        help="An optional resampling filter.PIL.Image.BILINEAR 2, PIL.Image.NEAREST 0,"
        " PIL.Image.BICUBIC 3",
    ))
@app.param(
    Int(
        key="fillcolor",
        default=0,
        help="Optional fill color (Tuple for RGB Image And int for grayscale) for the"
        " area outside the transform in the output image.",
    ))
@app.output(PytorchTransModel(key="outputModel"))
@app.output(Folder(key="outputData"))
def SPRandomAffine(context):
    """
    Random affine transformation of the image keeping center invariant
    """
    args = context.args
    transform = transforms.RandomAffine(
        args.degrees,
        translate=(*args.translate, ) if args.translate else args.translate,
        scale=(*args.scale, ) if args.scale else args.scale,
        shear=args.shear,
        resample=args.resample,
        fillcolor=args.fillcolor,
    )
    folder = transImgSave(args.inputData, transform) if args.inputData else mkFolder()
    return transform, folder


if __name__ == "__main__":
    suanpan.run(app)
