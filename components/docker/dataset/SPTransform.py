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
@dc.output(PytorchModel(key="outputModel"))
def SPTransform(context):
    # 从 Context 中获取相关数据
    args = context.args

    transformsAug = transforms.Compose([transforms.ToTensor()])

    return transformsAug


if __name__ == "__main__":
    SPTransform()
