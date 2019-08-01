# coding=utf-8
from __future__ import absolute_import, print_function

from suanpan.docker import DockerComponent as dc
from suanpan.docker.arguments import Folder
from utils.folder import ImageFolder
from arguments import PytorchDataset, PytorchTransModel


@dc.input(Folder(key="inputData"))
@dc.input(PytorchTransModel(key="inputModel"))
@dc.output(PytorchDataset(key="outputModel"))
def SPImageFolder(context):
    # 从 Context 中获取相关数据
    args = context.args

    dataset = ImageFolder(oss_root=args.inputData, transform=args.inputModel)

    return dataset


if __name__ == "__main__":
    SPImageFolder()
