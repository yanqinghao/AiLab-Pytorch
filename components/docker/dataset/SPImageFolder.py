# coding=utf-8
from __future__ import absolute_import, print_function

from suanpan.app import app

from utils.folder import ImageFolder
from arguments import PytorchDataset, PytorchTransModel, FolderPath


@app.input(FolderPath(key="inputData"))
@app.input(PytorchTransModel(key="inputModel"))
@app.output(PytorchDataset(key="outputData"))
def SPImageFolder(context):
    # 从 Context 中获取相关数据
    args = context.args

    dataset = ImageFolder(oss_root=args.inputData, transform=args.inputModel)

    return dataset


if __name__ == "__main__":
    SPImageFolder()
