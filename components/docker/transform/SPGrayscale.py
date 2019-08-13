# coding=utf-8
from __future__ import absolute_import, print_function

import torchvision.transforms as transforms

from suanpan.app.arguments import Int, Folder
from app import app
from arguments import PytorchTransModel, PytorchDataset
from utils import transImgSave


@app.input(PytorchDataset(key="inputData"))
@app.param(
    Int(
        key="numOutputChannels",
        default=1,
        help="number of channels desired for output image. 1 or 3",
    )
)
@app.output(PytorchTransModel(key="outputModel"))
@app.output(Folder(key="outputData"))
def SPGrayscale(context):
    # 从 Context 中获取相关数据

    args = context.args
    transform = transforms.Grayscale(num_output_channels=args.numOutputChannels)
    folder = None
    if args.inputData:
        folder = transImgSave(args.inputData, transform)

    return transform, folder


if __name__ == "__main__":
    SPGrayscale()
