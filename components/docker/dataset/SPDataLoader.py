# coding=utf-8
from __future__ import absolute_import, print_function

import torch
from suanpan.app import app
from suanpan.app.arguments import Int, Bool
from arguments import PytorchDataloader, PytorchDataset


@app.input(PytorchDataset(key="inputData"))
@app.param(Int(key="batchSize", default=100))
@app.param(Bool(key="shuffle", default=True))
@app.output(PytorchDataloader(key="outputData"))
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
