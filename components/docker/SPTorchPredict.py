# coding=utf-8
from __future__ import absolute_import, print_function

import os
import copy
import torch

import pandas as pd
from PIL import Image, ImageFont, ImageDraw
from suanpan.app.arguments import Folder, Csv
from app import app
from arguments import PytorchLayersModel, PytorchDataloader
from utils.visual import CNNNNVisualization


@app.input(PytorchLayersModel(key="inputModel"))
@app.input(PytorchDataloader(key="inputTestLoader"))
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
    folder = "/out_data/"
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
            filepath = filepath + list(paths)
            filelabel = filelabel + list(labels)
            if not pathtmp:
                pathtmp = list(paths)[0]

            for j in range(images.size()[0]):
                save_path = os.path.join(folder, paths[j])
                img = Image.open(os.path.join("/sp_data/", paths[j]))
                font = ImageFont.truetype("Ubuntu-B.ttf", size=20)
                draw = ImageDraw.Draw(img)
                draw.text(
                    (5, 5),
                    "predicted: {}".format(class_names[predicted[j]]),
                    (255, 255, 255),
                    font=font,
                )
                if not os.path.exists(os.path.split(save_path)[0]):
                    os.makedirs(os.path.split(save_path)[0])
                img.save(save_path)
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

        df = pd.DataFrame(
            {
                "file path or index": filepath,
                "label": filelabel,
                "predictions": [class_names[i] for i in prediction.tolist()],
            }
        )
        pathlist = pathtmp.split("/")
        folder = os.path.join(folder, *pathlist[:6])

    return folder, df


if __name__ == "__main__":
    SPTorchPredict()
