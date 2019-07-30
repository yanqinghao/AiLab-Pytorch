# coding=utf-8
from __future__ import absolute_import, print_function

import torch.nn as nn

from suanpan.docker import DockerComponent as dc
from suanpan.docker.arguments import Int, String
from arguments import PytorchModel
from utils import getLayerName


@dc.input(PytorchModel(key="inputModel"))
@dc.param(Int(key="inChannel", default=16))
@dc.param(Int(key="outChannel", default=32))
@dc.param(Int(key="kernelSize", default=5))
@dc.param(Int(key="stride", default=1))
@dc.param(Int(key="padding", default=2))
@dc.param(String(key="paddingMode", default="zeros"))
@dc.output(PytorchModel(key="outputModel"))
def SPConv2D(context):
    # 从 Context 中获取相关数据
    args = context.args
    # 查看上一节点发送的 args.inputData 数据
    model = args.inputModel
    name = getLayerName(model.layers, "Conv2D")
    model.layers.add_module(
        name,
        nn.Conv2d(
            args.inChannel,
            args.outChannel,
            kernel_size=args.kernelSize,
            stride=args.stride,
            padding=args.padding,
            padding_mode=args.paddingMode,
        ),
    )

    print(model.layers)

    return model


if __name__ == "__main__":
    SPConv2D()
