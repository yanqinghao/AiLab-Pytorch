# coding=utf-8
from __future__ import absolute_import, print_function

import random
import torch
import suanpan
from suanpan.app import app
from suanpan.app.arguments import Float
from suanpan.log import logger
from utils.folder import ImageFolder
from utils.mnist import MNIST
from args import PytorchDataset


@app.input(PytorchDataset(key="inputData"))
@app.param(Float(key="percentage", default=0.2))
@app.output(PytorchDataset(key="outputData"))
def SPDatasetSampler(context):
    args = context.args
    dataset = args.inputData
    percent = (args.percentage if args.percentage >= 0 else 0 if args.percentage <= 1 else 1)
    if isinstance(dataset, ImageFolder):
        length = int(len(dataset.samples) * percent) or 1
        dataset.samples = random.sample(dataset.samples, length)
    elif isinstance(dataset, MNIST):
        length = int(len(dataset.data) * percent) or 1
        idx = torch.tensor(random.sample(range(len(dataset.data)), length))
        dataset.data = torch.index_select(dataset.data, 0, idx)
        dataset.targets = torch.index_select(dataset.targets, 0, idx)
    else:
        logger.info("Wrong type dataset, return whole dataset.")
    return dataset


if __name__ == "__main__":
    suanpan.run(app)
