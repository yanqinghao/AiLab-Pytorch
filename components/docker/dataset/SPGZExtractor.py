# coding=utf-8
from __future__ import absolute_import, print_function

import os
import suanpan
from suanpan.app import app
from suanpan.app.arguments import Folder
from torchtext.utils import extract_archive


def find_all_gz(folder):
    files_ = []
    list = [i for i in os.listdir(folder)]
    for i in range(0, len(list)):
        path = os.path.join(folder, list[i])
        if os.path.isdir(path):
            files_.extend(find_all_gz(path))
        if not os.path.isdir(path):
            if path.lower().endswith("gz"):
                files_.append(path)
    return files_


@app.input(Folder(key="inputData"))
@app.output(Folder(key="outputData"))
def SPGZExtractor(context):
    args = context.args
    outputPath = "/tmp/output"
    extract_archive(find_all_gz(args.inputData)[0], to_path=outputPath)
    return outputPath


if __name__ == "__main__":
    suanpan.run(app)
