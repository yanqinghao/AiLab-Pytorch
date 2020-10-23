# coding=utf-8
from __future__ import absolute_import, print_function
import suanpan
from suanpan.app import app
from suanpan.app.arguments import Folder
from suanpan.storage import storage


@app.output(Folder(key="outputDir"))
def SPMNISTRAW(context):
    storage.download("common/data/mnist", "/tmp/mnist")
    return "/tmp/mnist"


if __name__ == "__main__":
    suanpan.run(app)
