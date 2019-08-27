# coding=utf-8
from __future__ import absolute_import, print_function

import torchvision.transforms as transforms

from suanpan.app import app
from suanpan.app.arguments import Folder, Int
from arguments import PytorchTransModel, PytorchDataset
from utils import transImgSave, mkFolder


@app.input(PytorchDataset(key="inputData"))
@app.param(Int(key="size", default=28, help=" Desired output size."))
@app.param(
    Int(
        key="interpolation",
        default=2,
        help="Desired interpolation.BILINEAR 2, NEAREST 0, BICUBIC 3, LANCZOS 1",
    )
)
@app.output(PytorchTransModel(key="outputModel"))
@app.output(Folder(key="outputData"))
def SPResize(context):
    # 从 Context 中获取相关数据
    args = context.args

    transform = transforms.Resize(args.size, interpolation=args.interpolation)
    folder = transImgSave(args.inputData, transform) if args.inputData else mkFolder()

    return transform, folder


if __name__ == "__main__":
    SPResize()
