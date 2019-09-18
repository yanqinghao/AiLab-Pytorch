# coding=utf-8
from __future__ import absolute_import, print_function

import os
import copy
import torch

import pandas as pd
import numpy as np
from PIL import Image, ImageDraw
from suanpan.app.arguments import Folder, Csv, Bool, ListOfInt
from suanpan import asyncio
from suanpan.screenshots import screenshots
from app import app
from arguments import PytorchLayersModel, PytorchDataloader
from utils.visual import CNNNNVisualization


@app.input(PytorchLayersModel(key="inputModel"))
@app.input(PytorchDataloader(key="inputTestLoader"))
@app.param(Bool(key="isLabeled", default=True))
@app.param(ListOfInt(key="fontColor", default=[255, 255, 255]))
@app.param(ListOfInt(key="fontXy", default=[5, 5]))
@app.output(Folder(key="outputData1"))
@app.output(Csv(key="outputData2"))
def SPTorchPredict(context):
    """
    model predict
    """
    args = context.args
    # 查看上一节点发送的 args.inputData 数据
    model = args.inputModel
    test_loader = args.inputTestLoader
    class_names = list(model.class_to_idx.keys())
    folder = "/pred_data/"
    pathtmp = ""
    # Device configuration
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Test the model
    # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    model.eval()
    with torch.no_grad():
        prediction = torch.tensor([], dtype=torch.long)
        filepath = []
        filelabel = []
        cnnVisual = CNNNNVisualization(model)
        cnnVisual.daemon = True
        cnnVisual.start()
        for images, labels, paths in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            prediction = torch.cat((prediction, predicted), 0)
            if isinstance(list(paths)[0], str):
                filepath = filepath + [os.path.join(*i[6:]) for i in list(paths)]
            else:
                filepath = filepath + list(paths.numpy())
            filelabel = filelabel + list(labels.numpy())
            if not pathtmp:
                pathtmp = list(paths)[0]

            for j in range(images.size()[0]):
                if isinstance(pathtmp, str):
                    save_path = os.path.join(folder, paths[j])
                else:
                    save_path = os.path.join(folder, "{}.png".format(paths[j]))
                img = Image.fromarray(
                    np.transpose(images[j].cpu().data.numpy(), (1, 2, 0))
                )
                draw = ImageDraw.Draw(img)
                draw.text(
                    (*args.fontXy,),
                    "predicted: {}".format(class_names[predicted[j]]),
                    (*args.fontColor,),
                )
                if not os.path.exists(os.path.split(save_path)[0]):
                    os.makedirs(os.path.split(save_path)[0])
                img.save(save_path)
            asyncio.run(screenshots.save(np.array(img)))
            cnnVisual.put(
                {
                    "status": "running",
                    "type": "layer",
                    "data": (copy.deepcopy(images), copy.deepcopy(paths)),
                }
            )
        cnnVisual.put({"status": "quit"})
        cnnVisual.tag = False
        cnnVisual.join()
        if args.isLabeled:
            df = pd.DataFrame(
                {
                    "file path or index": filepath,
                    "label": [class_names[i] for i in filelabel],
                    "predictions": [class_names[i] for i in prediction.tolist()],
                }
            )
        else:
            df = pd.DataFrame(
                {
                    "file path or index": filepath,
                    "predictions": [class_names[i] for i in prediction.tolist()],
                }
            )
        if isinstance(pathtmp, str):
            pathlist = pathtmp.split("/")
            folder = os.path.join(folder, *pathlist[:6])

    return folder, df


if __name__ == "__main__":
    SPTorchPredict()
