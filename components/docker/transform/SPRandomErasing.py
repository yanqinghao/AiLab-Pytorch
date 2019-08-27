"""
Created on Sun Aug 19 2019
@author: Yan Qinghao
transforms
"""
# coding=utf-8
from __future__ import absolute_import, print_function

import torchvision.transforms as transforms

from suanpan.app.arguments import Int, Bool, Float, ListOfFloat
from app import app
from arguments import PytorchTransModel, PytorchDataset


@app.input(PytorchDataset(key="inputData"))
@app.param(
    Float(
        key="p",
        default=0.5,
        help="probability that the random erasing operation will be performed.",
    )
)
@app.param(
    ListOfFloat(
        key="scale",
        default=[0.02, 0.33],
        help="range of proportion of erased area against input image.",
    )
)
@app.param(
    ListOfFloat(
        key="ratio", default=[0.3, 3.3], help="range of aspect ratio of erased area."
    )
)
@app.param(Int(key="value", default=0, help="erasing value. "))
@app.param(
    Bool(key="inplace", default=False, help="boolean to make this transform inplace.")
)
@app.output(PytorchTransModel(key="outputModel"))
def SPRandomErasing(context):
    """
    Crop the given PIL Image at a random location.
    """
    args = context.args

    transform = transforms.RandomErasing(
        p=args.p,
        scale=(*args.scale,),
        ratio=(*args.ratio,),
        value=args.value,
        inplace=args.inplace,
    )

    return transform


if __name__ == "__main__":
    SPRandomErasing()
