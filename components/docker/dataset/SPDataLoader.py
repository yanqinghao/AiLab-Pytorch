# coding=utf-8
from __future__ import absolute_import, print_function

import importlib
import torch
import suanpan
from suanpan.app import app
from suanpan.app.arguments import Int, Bool
from args import PytorchDataloader, PytorchDataset


@app.input(PytorchDataset(key="inputData"))
@app.param(Int(key="batchSize", default=32))
@app.param(Bool(key="shuffle", default=True))
@app.output(PytorchDataloader(key="outputData"))
def SPDataLoader(context):
    args = context.args
    if getattr(args.inputData, "collate", None) is None:
        dataLoader = torch.utils.data.DataLoader(dataset=args.inputData,
                                                 batch_size=args.batchSize,
                                                 shuffle=True)
    else:
        collateFN = getattr(importlib.import_module(f"utils"),
                            getattr(args.inputData, "collate", None))
        dataLoader = torch.utils.data.DataLoader(
            dataset=args.inputData,
            batch_size=args.batchSize,
            shuffle=True,
            collate_fn=collateFN,
        )

    return dataLoader


if __name__ == "__main__":
    suanpan.run(app)
