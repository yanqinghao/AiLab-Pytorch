# coding=utf-8
from __future__ import absolute_import, print_function

import torch.nn as nn

from suanpan.docker import DockerComponent as dc
from suanpan.docker.arguments import Int
from arguments import PytorchModel
from utils import getLayerName


@dc.input(PytorchModel(key="inputModel"))
@dc.param(Int(key="inFeature", default=7 * 7 * 32))
@dc.param(Int(key="outFeature", default=10))
@dc.output(PytorchModel(key="outputModel"))
def SPLinear(context):
    # 从 Context 中获取相关数据
    args = context.args
    # 查看上一节点发送的 args.inputData 数据
    model = args.inputModel
    name = getLayerName(model.layers, "Linear")
    model.layers.add_module(
        name, nn.Linear(in_features=args.inFeature, out_features=args.outFeature)
    )
    print(model.layers)
    return model


if __name__ == "__main__":
    SPLinear()
