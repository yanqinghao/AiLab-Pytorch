# coding=utf-8
from __future__ import absolute_import, print_function

import torch
import torch.nn as nn

from suanpan.components import Result
from suanpan.storage.arguments import Model
from suanpan.storage.arguments import Folder


class SPNet(nn.Module):
    def __init__(self):
        super(SPNet, self).__init__()
        self.layers = nn.ModuleList()

    def forward(self, x):
        out = x
        for i in self.layers:
            if isinstance(i, nn.Linear):
                out = out.reshape(out.size(0), -1)
            out = i(out)
        return out


class PytorchLayersModel(Model):
    FILETYPE = "layers"

    def format(self, context):
        super(PytorchLayersModel, self).format(context)
        if self.filePath:
            with open(self.filePath, "rb") as f:
                self.value = torch.load(f)

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
