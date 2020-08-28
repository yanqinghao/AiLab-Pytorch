# coding=utf-8
from __future__ import absolute_import, print_function

import os
import shutil

from suanpan.app import app
from suanpan.app.arguments import Folder
import utils
from arguments import PytorchDataset, PytorchTransModel


@app.input(Folder(key="inputData"))
@app.input(PytorchTransModel(key="inputModel"))
@app.output(PytorchDataset(key="outputData1"))
@app.output(PytorchDataset(key="outputData2"))
def SPMNIST(context):

    args = context.args

    filePath = os.path.join(args.inputData, "MNIST", "raw")
    if "MNIST" not in os.listdir(args.inputData):
        os.mkdir(os.path.join(args.inputData, "MNIST"))
    if "raw" not in os.listdir(os.path.join(args.inputData, "MNIST")):
        os.mkdir(filePath)
    filename = [
        "train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz",
    ]
    for i in filename:
        if not os.path.exists(os.path.join(filePath, i)):
            shutil.move(os.path.join(args.inputData, i), os.path.join(filePath, i))

    # MNIST dataset
    train_dataset = utils.mnist.MNIST(
        root=args.inputData, train=True, transform=args.inputModel, download=True
    )

    test_dataset = utils.mnist.MNIST(
        root=args.inputData, train=False, transform=args.inputModel
    )

    return train_dataset, test_dataset


if __name__ == "__main__":
    SPMNIST()
