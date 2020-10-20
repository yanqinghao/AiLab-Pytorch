# coding=utf-8
from __future__ import absolute_import, print_function
import suanpan
from suanpan.app import app
from suanpan.storage import storage
from suanpan.app.arguments import String
from utils.folder import ImageFolder
from args import PytorchDataset, PytorchTransModel


@app.input(String(key="inputData"))
@app.input(PytorchTransModel(key="inputModel"))
@app.output(PytorchDataset(key="outputData"))
def SPImageFolderV2(context):
    args = context.args
    folder_list = [i for i in storage.listFolders(args.inputData)]
    dataset = ImageFolder(oss_root=folder_list[0], transform=args.inputModel)
    return dataset


if __name__ == "__main__":
    suanpan.run(app)
