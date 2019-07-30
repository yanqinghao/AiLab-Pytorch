# coding=utf-8
from __future__ import absolute_import, print_function

import os
import shutil
import torchvision.transforms as transforms

from suanpan.docker import DockerComponent as dc
from suanpan.docker.arguments import Int, Folder
import utils
from arguments import PytorchModel


@dc.input(Folder(key="inputData"))
@dc.param(Int(key="batchSize", default=100))
@dc.output(PytorchModel(key="outputModel1"))
@dc.output(PytorchModel(key="outputModel2"))
def SPMNIST(context):
    # 从 Context 中获取相关数据
    args = context.args
    # 查看上一节点发送的 args.inputData 数据
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
    # Hyper parameters
    batch_size = args.batchSize

    # MNIST dataset
    train_dataset = utils.mnist.MNIST(
        root=args.inputData, train=True, transform=transforms.ToTensor(), download=True
    )

    test_dataset = utils.mnist.MNIST(
        root=args.inputData, train=False, transform=transforms.ToTensor()
    )

    return train_dataset, test_dataset


if __name__ == "__main__":
    SPMNIST()
