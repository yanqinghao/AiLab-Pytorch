# coding=utf-8
from __future__ import absolute_import, print_function

import torchvision.transforms as transforms

from suanpan.app.arguments import Int, Bool, String, Folder
from app import app
from arguments import PytorchTransModel, PytorchDataset
from utils import transImgSave


@app.input(PytorchDataset(key="inputData"))
@app.param(Int(key="size", default=28, help="Desired output size of the crop."))
@app.param(
    Int(
        key="padding",
        default=None,
        help="Optional padding on each border of the image.",
    )
)
@app.param(
    Bool(
        key="padIfNeeded",
        default=False,
        help="It will pad the image if smaller than the desired size to avoid "
        "raising an exception.",
    )
)
@app.param(Int(key="fill", default=0, help="Pixel fill value for constant fill."))
@app.param(
    String(
        key="paddingMode",
        default="constant",
        help="Type of padding. constant, edge, reflect, symmetric",
    )
)
@app.output(PytorchTransModel(key="outputModel"))
@app.output(Folder(key="outputData"))
def SPRandomCrop(context):
    # 从 Context 中获取相关数据
    args = context.args

    transform = transforms.RandomCrop(
        args.size,
        padding=args.padding,
        pad_if_needed=args.padIfNeeded,
        fill=args.fill,
        padding_mode=args.paddingMode,
    )
    folder = None
    if args.inputData:
        folder = transImgSave(args.inputData, transform)

    return transform, folder


if __name__ == "__main__":
    SPRandomCrop()
