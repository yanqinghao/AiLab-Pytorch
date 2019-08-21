"""
Created on Sun Aug 19 2019
@author: Yan Qinghao
transforms
"""
# coding=utf-8
from __future__ import absolute_import, print_function

import torchvision.transforms as transforms

from suanpan.app.arguments import Int, Folder, ListOfFloat
from app import app
from arguments import PytorchTransModel, PytorchDataset
from utils import transImgSave, mkFolder


@app.input(PytorchDataset(key="inputData"))
@app.param(Int(key="size", default=28, help="expected output size of each edge"))
@app.param(
    ListOfFloat(
        key="scale",
        default=[0.08, 1.0],
        help="range of size of the origin size cropped",
    )
)
@app.param(
    ListOfFloat(
        key="ratio",
        default=[3.0 / 4.0, 4.0 / 3.0],
        help="range of aspect ratio of the origin aspect ratio cropped",
    )
)
@app.param(
    Int(
        key="interpolation",
        default=2,
        help="BILINEAR 2, NEAREST 0, BICUBIC 3, LANCZOS 1",
    )
)
@app.output(PytorchTransModel(key="outputModel"))
@app.output(Folder(key="outputData"))
def SPRandomResizedCrop(context):
    """
    Crop the given PIL Image to random size and aspect ratio.
    """
    args = context.args

    transform = transforms.RandomResizedCrop(
        args.size,
        scale=(*args.scale,),
        ratio=(*args.ratio,),
        interpolation=args.interpolation,
    )
    folder = transImgSave(args.inputData, transform) if args.inputData else mkFolder()

    return transform, folder


if __name__ == "__main__":
    SPRandomResizedCrop()
