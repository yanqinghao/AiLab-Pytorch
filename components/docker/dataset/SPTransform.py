# coding=utf-8
from __future__ import absolute_import, print_function

import torchvision.transforms as transforms

from suanpan.app import app
from suanpan.app.arguments import Folder
from arguments import PytorchTransModel


@app.input(Folder(key="inputData"))
@app.output(PytorchTransModel(key="outputModel"))
def SPTransform(context):
    # 从 Context 中获取相关数据
    args = context.args

    transformsAug = transforms.Compose(
        [
            transforms.RandomCrop(28),
            transforms.Grayscale(),
            transforms.Resize(28),
            transforms.ToTensor(),
        ]
    )

    return transformsAug


if __name__ == "__main__":
    SPTransform()
