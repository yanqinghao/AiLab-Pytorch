# coding=utf-8
from __future__ import absolute_import, print_function

import suanpan
from suanpan.app import app
from suanpan.docker.arguments import Folder
from suanpan.storage import OssStorage


@app.output(Folder(key="trainDir"))
@app.output(Folder(key="valDir"))
@app.output(Folder(key="testDir"))
def SPCatsvsDogsOss(context):
    args = context.args
    oss = OssStorage(ossAccessId=None, ossAccessKey=None)
    oss.download("common/data/cats_and_dogs/train", args.trainDir)
    oss.download("common/data/cats_and_dogs/validation", args.valDir)
    oss.download("common/data/cats_and_dogs/test", args.testDir)
    return args.trainDir, args.valDir, args.testDir


if __name__ == "__main__":
   suanpan.run(app)
