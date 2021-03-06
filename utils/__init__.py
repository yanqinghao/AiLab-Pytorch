# coding=utf-8
from __future__ import absolute_import, print_function

import os
import zipfile
import torch
import numpy as np
import matplotlib.pyplot as plt
# import hiddenlayer as hl
# from graphviz import Digraph
from suanpan.screenshots import screenshots
from suanpan.storage import storage
from suanpan.utils import image
from suanpan.log import logger
from utils.mnist import MNIST
from utils.folder import ImageFolder
from utils.visual import CNNNNVisualization, CNNLayerVisualization
# from utils.visual import createScreenshots, getScreenshotPath
from utils.download import downloadPretrained, downloadTextDataset
from utils.collate import generate_batch
from utils.net import SPMathOP


def getLayerName(moduleList, match):
    layersName = []
    for name, module in moduleList.items():
        if match in name:
            layersName.append(int(name.replace(match + "_", "")))
    return f"{match}_{max(layersName) + 1}" if layersName else f"{match}_0"


def zipdir(path, ziph):
    for root, dirs, files in os.walk(path):
        for file in files:
            ziph.write(os.path.join(root, file))


def find_all_files(folder):
    files_ = []
    subpaths = [i for i in os.listdir(folder)]
    for i in range(0, len(subpaths)):
        path = os.path.join(folder, subpaths[i])
        if os.path.isdir(path):
            files_.extend(find_all_files(path))
        if not os.path.isdir(path):
            files_.append(path)
    return files_


def transImgSave(dataset, transform):
    folder = "/out_data/"
    pathtmp = ""
    for img, _, paths in dataset:
        if isinstance(paths, str):
            save_path = os.path.join(folder, os.path.join(*paths.split(storage.delimiter)[6:]))
        else:
            paths_img = "{}.png".format(paths)
            save_path = os.path.join(folder, paths_img)
        transformed_img = transform(img)
        if not os.path.exists(os.path.split(save_path)[0]):
            os.makedirs(os.path.split(save_path)[0])
        transformed_img.save(save_path)
        if not pathtmp:
            pathtmp = paths
    # if isinstance(pathtmp, str):
    #     pathlist = pathtmp.split("/")
    #     output = os.path.join(folder, *pathlist[:6])
    # else:
    #     output = "/out_data"
    if len(find_all_files(folder)) > 50:
        transform_images = "/tmp/zip_res"
        os.makedirs(transform_images, exist_ok=True)
        zipf = zipfile.ZipFile(os.path.join(transform_images, 'transform-result.zip'), 'w',
                               zipfile.ZIP_DEFLATED)
        zipdir(folder, zipf)
        zipf.close()
        return transform_images
    else:
        return folder


def mkFolder():
    folder = "/out_data"
    if not os.path.exists(folder):
        os.mkdir(folder)
    return folder


# def plotLayers(model, input_size=None):
#     try:
#         if model.task == "image":
#             file_name = "screenshots"
#             output_size = (calOutput(model) if len(model.layers) > 1
#                            or isinstance(model, SPMathOP) else [1] + [model.input_size[2]] +
#                            list(model.input_size[:2]) if len(model.input_size) == 3 else [1] +
#                            list(model.input_size))
#             name_list = [i[0] for i in model.layers.items()]
#             model_name = "{} Layer".format(name_list[-1])
#             input_name = ("IN (N {})".format("".join(["* {}".format(i) for i in input_size[1:]]))
#                           if len(model.layers) > 1 else None)
#             output_name = "OUT (N {})".format("".join(["* {}".format(i) for i in output_size[1:]]))
#             dot = Digraph(comment="Pytorch Model")
#             dot.attr("graph", size="12,12", dpi="300", bgcolor="#E8E8E8")
#             if input_name:
#                 dot.node(input_name)
#             dot.node(model_name, shape="box", style="filled", color="lightblue2")
#             dot.node(output_name)
#             if input_name:
#                 dot.edge(input_name, model_name)
#             dot.edge(model_name, output_name)
#             dot.render(file_name, format="png")
#             layer_image = image.read("{}.png".format(file_name))
#             screenshots.save(layer_image)
#     except:
#         logger.info("can not plot screeshot.")


def calOutput(model):
    try:
        model.to(torch.device("cpu"))
        input_size = ([1] + [model.input_size[-1]] +
                      list(model.input_size[:-1]) if len(model.input_size) == 3 else [1] +
                      list(model.input_size))
        return (model(torch.zeros(input_size)).shape
                if len(model.layers) > 1 or isinstance(model, SPMathOP) else input_size)
    except:
        return None


def datasetScreenshot(dataset, screenshot):
    for img, _, _ in dataset:
        screenshot.save(np.asarray(img))
        break
