# coding=utf-8
from __future__ import absolute_import, print_function

import os
import copy
import torch

import pandas as pd
import numpy as np
from torchvision.transforms import functional as F
from PIL import ImageDraw, ImageFont, Image
from suanpan.storage import storage
from suanpan.app.arguments import Folder, Csv, Bool, ListOfInt, Int
from suanpan.screenshots import screenshots
from app import app
from arguments import PytorchLayersModel, PytorchDataloader
from utils.visual import CNNNNVisualization


@app.input(PytorchLayersModel(key="inputModel"))
@app.input(PytorchDataloader(key="inputTestLoader"))
@app.param(Bool(key="isLabeled", default=True))
@app.param(ListOfInt(key="fontColor", default=[255, 255, 255]))
@app.param(ListOfInt(key="fontXy", default=[5, 5]))
@app.param(Int(key="fontSize", default=8))
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
                filepath = filepath + [
                    os.path.join(*i.split(storage.delimiter)[6:]) for i in list(paths)
                ]
            else:
                filepath = filepath + list(paths.numpy())
            if not isinstance(labels[0], str):
                filelabel = filelabel + list(labels.numpy())
            else:
                filelabel = filelabel + list(labels)
            if not pathtmp:
                pathtmp = list(paths)[0]

            for j in range(images.size()[0]):
                if isinstance(pathtmp, str):
                    save_path = os.path.join(
                        folder, class_names[predicted[j]], paths[j]
                    )
                else:
                    save_path = os.path.join(
                        folder, class_names[predicted[j]], "{}.png".format(paths[j])
                    )
                if isinstance(pathtmp, str):
                    img = Image.open(os.path.join("/sp_data/", paths[j]))
                else:
                    img = F.to_pil_image(images[j].cpu())
                img.save(save_path)
                draw = ImageDraw.Draw(img)
                font = ImageFont.truetype("./utils/Ubuntu-B.ttf", args.fontSize)
                draw.text(
                    (*args.fontXy,),
                    "predicted: {}".format(class_names[predicted[j]]),
                    (*args.fontColor,),
                    font=font,
                )
                if not os.path.exists(os.path.split(save_path)[0]):
                    os.makedirs(os.path.split(save_path)[0])

            screenshots.save(np.array(img))
            cnnVisual.put(
                {
                    "status": "running",
                    "type": "layer",
                    "data": (copy.deepcopy(images), copy.deepcopy(paths)),
                }
            )
        cnnVisual.put({"status": "quit"})
        cnnVisual.tag = False
        cnnVisual.empty()
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
            pathlist = pathtmp.split(storage.delimiter)
            folder = os.path.join(folder, *pathlist[:6])

    return folder, df


if __name__ == "__main__":
    SPTorchPredict()
