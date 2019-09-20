# coding=utf-8
from __future__ import absolute_import, print_function

from suanpan.app import app
from suanpan.docker.arguments import Folder, String
from suanpan.storage import StorageProxy


@app.param(String(key="storageType", default="oss"))
@app.output(Folder(key="outputDir"))
def SPMNISTRAW(context):
    args = context.args

    storage = StorageProxy(None, None)
    storage.setBackend(type=args.storageType)

    storage.download("common/data/mnist", args.outputDir)

    return args.outputDir


if __name__ == "__main__":
    SPMNISTRAW()  # pylint: disable=no-value-for-parameter
