# coding=utf-8
from __future__ import absolute_import, print_function

import torch.nn as nn

from suanpan.docker import DockerComponent as dc
from suanpan.docker.arguments import Bool
from arguments import PytorchLayersModel
from utils import getLayerName


@dc.input(PytorchLayersModel(key="inputModel"))
@dc.param(Bool(key="inplace", default=False))
@dc.output(PytorchLayersModel(key="outputModel"))
def SPReLU(context):
    # 从 Context 中获取相关数据
    args = context.args
    # 查看上一节点发送的 args.inputData 数据
    model = args.inputModel
    name = getLayerName(model.layers, "ReLU")
    model.layers.add_module(name, nn.ReLU(inplace=args.inplace))
    print(model.layers)
    return model


if __name__ == "__main__":
    SPReLU()
