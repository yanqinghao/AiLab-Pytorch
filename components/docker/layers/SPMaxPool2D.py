# coding=utf-8
from __future__ import absolute_import, print_function

import torch.nn as nn

from suanpan.docker import DockerComponent as dc
from suanpan.docker.arguments import Int
from arguments import PytorchModel
from utils import getLayerName


@dc.input(PytorchModel(key="inputModel"))
@dc.param(Int(key="kernelSize", default=2))
@dc.param(Int(key="stride", default=2))
@dc.output(PytorchModel(key="outputModel"))
def SPMaxPool2D(context):
    # 从 Context 中获取相关数据
    args = context.args
    # 查看上一节点发送的 args.inputData 数据
    model = args.inputModel
    name = getLayerName(model.layers, "MaxPool2D")
    model.layers.add_module(
        name, nn.MaxPool2d(kernel_size=args.kernelSize, stride=args.stride)
    )
    print(model.layers)
    return model


if __name__ == "__main__":
    SPMaxPool2D()
