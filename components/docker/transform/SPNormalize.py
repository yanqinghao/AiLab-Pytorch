# coding=utf-8
from __future__ import absolute_import, print_function

import torchvision.transforms as transforms

from suanpan.app.arguments import Bool, Folder, ListOfFloat
from app import app
from arguments import PytorchTransModel, PytorchDataset
from utils import transImgSave, mkFolder


@app.input(PytorchDataset(key="inputData"))
@app.param(
    ListOfFloat(
        key="mean",
        default=[0.485, 0.456, 0.406],
        help="Sequence of means for each channel.",
    )
)
@app.param(
    ListOfFloat(
        key="std",
        default=[0.229, 0.224, 0.225],
        help="Sequence of standard deviations for each channel.",
    )
)
@app.param(
    Bool(key="inplace", default=False, help="Bool to make this operation in-place.")
)
@app.output(PytorchTransModel(key="outputModel"))
@app.output(Folder(key="outputData"))
def SPNormalize(context):
    """
    Normalize a tensor image with mean and standard deviation. 
    """

    args = context.args
    transform = transforms.Normalize(args.mean, args.std, inplace=args.inplace)
    folder = transImgSave(args.inputData, transform) if args.inputData else mkFolder()

    return transform, folder


if __name__ == "__main__":
    SPNormalize()
