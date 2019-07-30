# coding=utf-8
from __future__ import absolute_import, print_function
from .mnist import MNIST


def hello():
    print("Utils Hello!")


def getLayerName(moduleList, match):
    layersName = []
    for name, module in moduleList.named_children():
        if match in name:
            layersName.append(int(name.replace(match + "_", "")))
    return (
        "{}_{}".format(match, max(layersName) + 1)
        if layersName
        else "{}_{}".format(match, 0)
    )
