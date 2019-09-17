# coding=utf-8
from __future__ import absolute_import, print_function

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

    storage.download("common/data/cats_and_dogs/train", args.trainDir)
    storage.download("common/data/cats_and_dogs/validation", args.valDir)
    storage.download("common/data/cats_and_dogs/test", args.testDir)

    return args.trainDir, args.valDir, args.testDir


if __name__ == "__main__":
    SPCatsvsDogs()  # pylint: disable=no-value-for-parameter
