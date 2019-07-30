# coding=utf-8
from __future__ import absolute_import, print_function

import torch
from suanpan.docker import DockerComponent as dc
from suanpan.docker.arguments import Int, Bool
from arguments import PytorchModel


@dc.input(PytorchModel(key="inputData"))
@dc.param(Int(key="batchSize", default=100))
@dc.param(Bool(key="shuffle", default=True))
@dc.output(PytorchModel(key="outputModel"))
def SPDataLoader(context):
    # 从 Context 中获取相关数据
    args = context.args
    # 查看上一节点发送的 args.inputData 数据
    dataLoader = torch.utils.data.DataLoader(
        dataset=args.inputData, batch_size=args.batchSize, shuffle=True
    )

    return dataLoader


if __name__ == "__main__":
    SPDataLoader()
