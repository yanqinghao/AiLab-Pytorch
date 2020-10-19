"""
Created on Sun Aug 19 2019
@author: Yan Qinghao
transforms
"""
# coding=utf-8
from __future__ import absolute_import, print_function

import torchvision.transforms as transforms
import suanpan
from suanpan.app.arguments import Float, Int, Folder
from suanpan.app import app
from args import PytorchTransModel, PytorchDataset
from utils import transImgSave, mkFolder


@app.input(PytorchDataset(key="inputData"))
@app.param(
    Float(
        key="distortionScale",
        default=0.5,
        help="it controls the degree of distortion and ranges from 0 to 1. ",
    ))
@app.param(
    Float(
        key="p",
        default=0.5,
        help=" probability of the image being perspectively transformed.",
    ))
@app.param(
    Int(
        key="interpolation",
        default=3,
        help="BILINEAR 2, NEAREST 0, BICUBIC 3, LANCZOS 1",
    ))
@app.output(PytorchTransModel(key="outputModel"))
@app.output(Folder(key="outputData"))
def SPRandomPerspective(context):
    """
    Performs Perspective transformation of the given PIL Image randomly with a given probability.
    """
    args = context.args
    transform = transforms.RandomPerspective(
        distortion_scale=args.distortionScale,
        p=args.p,
        interpolation=args.interpolation,
    )
    folder = transImgSave(args.inputData, transform) if args.inputData else mkFolder()
    return transform, folder


if __name__ == "__main__":
    suanpan.run(app)
