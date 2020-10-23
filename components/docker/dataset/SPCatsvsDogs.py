# coding=utf-8
from __future__ import absolute_import, print_function

import os
import zipfile
import suanpan
from suanpan.app import app
from suanpan.docker.arguments import Folder
from suanpan.storage import storage


@app.output(Folder(key="trainDir"))
@app.output(Folder(key="valDir"))
@app.output(Folder(key="testDir"))
def SPCatsvsDogs(context):
    storage.download("common/data/cats_and_dogs/cats_and_dogs.zip", "cats_and_dogs.zip")
    outpath = "./cats_and_dogs"
    with open("cats_and_dogs.zip", "rb") as f:
        z = zipfile.ZipFile(f)
        for name in z.namelist():
            z.extract(name, outpath)
    trainDir = os.path.join(outpath, "train")
    valDir = os.path.join(outpath, "validation")
    testDir = os.path.join(outpath, "test")
    return trainDir, valDir, testDir


if __name__ == "__main__":
    suanpan.run(app)
