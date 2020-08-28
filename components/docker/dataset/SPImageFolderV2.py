# coding=utf-8
from __future__ import absolute_import, print_function

from suanpan.app import app
from suanpan.storage import storage
from utils.folder import ImageFolder
from arguments import PytorchDataset, PytorchTransModel, FolderPath


@app.input(FolderPath(key="inputData"))
@app.input(PytorchTransModel(key="inputModel"))
@app.output(PytorchDataset(key="outputData"))
def SPImageFolderV2(context):
    # 从 Context 中获取相关数据
    args = context.args

    folder_list = [i for i in storage.listFolders(args.inputData)]
    dataset = ImageFolder(oss_root=folder_list[0], transform=args.inputModel)

    return dataset


if __name__ == "__main__":
    SPImageFolderV2()
