# coding=utf-8
from __future__ import absolute_import, print_function

import os
import zipfile
from suanpan.app import app
from suanpan.docker.arguments import Folder, String
from suanpan.storage import StorageProxy
from suanpan import path


@app.param(String(key="storageType", default="oss"))
@app.output(Folder(key="trainDir"))
@app.output(Folder(key="valDir"))
@app.output(Folder(key="testDir"))
def SPCatsvsDogs(context):
    args = context.args

    storage = StorageProxy(None, None)
    storage.setBackend(type=args.storageType)

    storage.download("common/data/cats_and_dogs/cats_and_dogs.zip", "cats_and_dogs.zip")
    outpath = "./cats_and_dogs"
    with open("cats_and_dogs.zip", "rb") as f:
        z = zipfile.ZipFile(f)
        for name in z.namelist():
            z.extract(name, outpath)
    trainDir = os.path.join(outpath, "train")
    valDir = os.path.join(outpath, "validation")
    testDir = os.path.join(outpath, "test")
    path.copyFolder(trainDir, args.trainDir)
    path.copyFolder(valDir, args.valDir)
    path.copyFolder(testDir, args.testDir)
    return args.trainDir, args.valDir, args.testDir


if __name__ == "__main__":
    SPCatsvsDogs()  # pylint: disable=no-value-for-parameter
