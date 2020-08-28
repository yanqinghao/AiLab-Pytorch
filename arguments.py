# coding=utf-8
from __future__ import absolute_import, print_function

import torch
import pickle
import torch.nn as nn

from suanpan.components import Result
from suanpan.storage.arguments import Model
from suanpan.storage.arguments import Folder


class SPNet(nn.Module):
    def __init__(self, input_size, task):
        super(SPNet, self).__init__()
        self.layers = {}
        self.class_to_idx = None
        self.input_size = tuple(input_size) if input_size else None
        self.task = task

    def forward(self, x, offsets=None):
        out = x
        for _, j in self.layers.items():
            if j[0] is not None:
                if isinstance(j[0], nn.EmbeddingBag):
                    out = j[0](out, offsets)
                else:
                    out = j[0](out)
        return out


class SPMathOP(nn.Module):
    def __init__(self, input_size, task, model_list, op, param=None):
        super(SPMathOP, self).__init__()
        self.layers = {}
        self.class_to_idx = None
        self.input_size = tuple(input_size) if input_size else None
        self.task = task
        self.model_list = model_list
        self.op = op
        self.param = param

    def forward(self, x, offsets=None):
        out = x
        mid = []
        for model in self.model_list:
            mid.append(model(out))
        if self.op == "add":
            out = getattr(torch, self.op)(*mid)
        elif self.op == "cat":
            out = getattr(torch, self.op)((*mid,), **self.param)
        for _, j in self.layers.items():
            if j[0] is not None:
                if isinstance(j[0], nn.EmbeddingBag):
                    out = j[0](out, offsets)
                else:
                    out = j[0](out)
        return out


class PytorchLayersModel(Model):
    FILETYPE = "layers"

    def format(self, context):
        super(PytorchLayersModel, self).format(context)
        if self.filePath:
            with open(self.filePath, "rb") as f:
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                self.value = torch.load(f, map_location=device)

        return self.value

    def save(self, context, result):
        with open(self.filePath, "wb") as f:
            torch.save(result.value, f)

        return super(PytorchLayersModel, self).save(
            context, Result.froms(value=self.filePath)
        )


class PytorchDataset(PytorchLayersModel):
    FILENAME = "dataset"
    FILETYPE = None


class PytorchDataloader(PytorchLayersModel):
    FILENAME = "dataloader"
    FILETYPE = None


class PytorchTransModel(PytorchLayersModel):
    FILETYPE = "transaug"


class FolderPath(Folder):
    def load(self, args):
        self.value = getattr(args, self.key)
        self.logLoaded(self.value)
        return self.value

    def format(self, context):
        return self.value


class PytorchOptimModel(Model):
    FILETYPE = "optim"

    def format(self, context):
        super(PytorchOptimModel, self).format(context)
        if self.filePath:
            with open(self.filePath, "rb") as f:
                self.value = pickle.load(f)

        return self.value

    def save(self, context, result):
        with open(self.filePath, "wb") as f:
            pickle.dump(result.value, f)

        return super(PytorchOptimModel, self).save(
            context, Result.froms(value=self.filePath)
        )


class PytorchSchedulerModel(PytorchOptimModel):
    FILETYPE = "scheduler"


class PytorchFinetuningModel(PytorchOptimModel):
    FILETYPE = "finetuning"


class PytorchLayersStreamModel(PytorchLayersModel):
    def format(self, context):
        if self.filePath:
            with open(self.filePath, "rb") as f:
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                self.value = torch.load(f, map_location=device)

        return self.value
