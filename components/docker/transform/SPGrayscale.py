# coding=utf-8
from __future__ import absolute_import, print_function

import torchvision.transforms as transforms
import suanpan
from suanpan.app.arguments import Int, Folder
from suanpan.app import app
from args import PytorchTransModel, PytorchDataset
from utils import transImgSave, mkFolder


@app.input(PytorchDataset(key="inputData"))
@app.param(
    Int(
        key="numOutputChannels",
        default=1,
        help="number of channels desired for output image. 1 or 3",
    ))
@app.output(PytorchTransModel(key="outputModel"))
@app.output(Folder(key="outputData"))
def SPGrayscale(context):
    args = context.args
    transform = transforms.Grayscale(num_output_channels=args.numOutputChannels)
    folder = transImgSave(args.inputData, transform) if args.inputData else mkFolder()
    return transform, folder


if __name__ == "__main__":
    suanpan.run(app)
