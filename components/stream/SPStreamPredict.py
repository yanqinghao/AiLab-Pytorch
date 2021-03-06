# coding=utf-8
from __future__ import absolute_import, print_function

import os
import suanpan
from suanpan.app import app
from suanpan.app.arguments import Folder, Model, Int, Json
from suanpan.utils import npy, csv
from model import PytorchModel


def find_all_files(folder):
    files_ = []
    list = [i for i in os.listdir(folder)]
    for i in range(0, len(list)):
        path = os.path.join(folder, list[i])
        if os.path.isdir(path):
            files_.extend(find_all_files(path))
        if not os.path.isdir(path):
            files_.append(path)
    return files_


@app.input(Folder(key="inputData1"))
@app.input(Model(key="model1", model=PytorchModel))
@app.param(Int(key="duration", default=3600))
@app.output(Json(key="outputData1"))
def predict(context):
    args = context.args
    filePath = find_all_files(args.inputData1)[0]
    predictions = []
    if filePath.endswith("npy"):
        arr = npy.load(filePath)
        for data in arr:
            prediction = args.model1.predict_stream(data, "image")
            predictions += prediction
    elif filePath.endswith("csv"):
        dataframe = csv.load(filePath)
        for data in dataframe["text"].values:
            prediction = args.model1.predict_stream(data, "text")
            predictions += prediction

    return predictions


@app.afterCall
def modelHotReload(context):
    args = context.args
    if app.isStream:
        args.model1.reload(duration=args.duration)


if __name__ == "__main__":
    suanpan.run(app)
