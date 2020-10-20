# coding=utf-8
from __future__ import absolute_import, print_function
import suanpan
from suanpan.app import app
from suanpan.app.arguments import Folder, String
from suanpan.storage import StorageProxy


@app.param(String(key="storageType", default="oss"))
@app.output(Folder(key="outputDir"))
def SPMNISTRAW(context):
    args = context.args
    storage = StorageProxy(None, None)
    storage.setBackend(type=args.storageType)
    storage.download("common/data/mnist", "/tmp/mnist")
    return "/tmp/mnist"


if __name__ == "__main__":
    suanpan.run(app)
