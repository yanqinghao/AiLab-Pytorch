# coding=utf-8
from __future__ import absolute_import, print_function

import torchvision.transforms as transforms

from suanpan.docker import DockerComponent as dc
from suanpan.docker.arguments import Folder
from arguments import PytorchTransModel


@dc.input(Folder(key="inputData"))
@dc.output(PytorchTransModel(key="outputModel"))
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
