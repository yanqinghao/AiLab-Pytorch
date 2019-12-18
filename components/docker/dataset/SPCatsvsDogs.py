# coding=utf-8
from __future__ import absolute_import, print_function

import os
import zipfile
from suanpan.app import app
from suanpan.docker.arguments import Folder, String
from suanpan.storage import StorageProxy


@app.param(String(key="storageType", default="oss"))
@app.output(Folder(key="trainDir"))
@app.output(Folder(key="valDir"))
@app.output(Folder(key="testDir"))
def SPCatsvsDogs(context):
    args = context.args

    storage = StorageProxy(None, None)
    storage.setBackend(type=args.storageType)

    storage.download("common/data/cats_and_dogs/cats_and_dogs.zip", "cats_and_dogs.zip")
    outpath = "./"
    with open("cats_and_dogs.zip", "rb") as f:
        z = zipfile.ZipFile(f)
        for name in z.namelist():
            z.extract(name, outpath)
    trainDir = os.path.join(outpath, "cats_and_dogs", "train")
    valDir = os.path.join(outpath, "cats_and_dogs", "validation")
    testDir = os.path.join(outpath, "cats_and_dogs", "test")
    return trainDir, valDir, testDir


if __name__ == "__main__":
    SPCatsvsDogs()  # pylint: disable=no-value-for-parameter
