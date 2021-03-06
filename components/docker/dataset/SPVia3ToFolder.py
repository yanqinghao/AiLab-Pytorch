# coding=utf-8
from __future__ import absolute_import, print_function

import os
import shutil
import json
import suanpan
from suanpan.app import app
from suanpan.app.arguments import Folder, Float


def getFilePath(path, classname, filename):
    dst = os.path.join(path, classname, filename)
    if os.path.exists(dst):
        filename = "{}.1.png".format(filename)
        dst = getFilePath(path, classname, filename)
    return dst


@app.input(Folder(key="inputData1"))
@app.input(Folder(key="inputData2"))
@app.param(Float(key="trainTestSplit", default=0.8))
@app.output(Folder(key="outputData1"))
@app.output(Folder(key="outputData2"))
def SPVia3ToFolder(context):
    args = context.args
    jsonFile = os.path.join(args.inputData2, "project.json")
    with open(jsonFile, "rb") as load_f:
        fileInfo = json.load(load_f)
    num2Label = fileInfo["attribute"]["1"]["options"]
    fileList = {}
    for i, j in fileInfo["metadata"].items():
        if j["av"]:
            if num2Label[j["av"]["1"]] in fileList.keys():
                fileList[num2Label[j["av"]["1"]]].append(j["vid"])
            else:
                fileList[num2Label[j["av"]["1"]]] = [j["vid"]]
    for className in fileList.keys():
        for n in range(len(fileList[className])):
            if n < int(len(fileList) * args.trainTestSplit):
                src = os.path.join(args.inputData1, fileList[className][n])
                dst = getFilePath(args.outputData1, className, os.path.split(src)[1])
                if not os.path.exists(os.path.split(dst)[0]):
                    os.makedirs(os.path.split(dst)[0])
                shutil.copy(src, dst)
            else:
                src = os.path.join(args.inputData1, fileList[className][n])
                dst = getFilePath(args.outputData2, className, os.path.split(src)[1])
                if not os.path.exists(os.path.split(dst)[0]):
                    os.makedirs(os.path.split(dst)[0])
                shutil.copy(src, dst)
    return args.outputData1, args.outputData2


if __name__ == "__main__":
    suanpan.run(app)
