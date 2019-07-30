# coding=utf-8
from __future__ import absolute_import, print_function

import torch
import torch.nn as nn

from suanpan.components import Result
from suanpan.storage.arguments import Model


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


class PytorchModel(Model):
    FILETYPE = "torch"

    def format(self, context):
        super(PytorchModel, self).format(context)
        if self.filePath:
            with open(self.filePath, "rb") as f:
                self.value = torch.load(f)

        return self.value

    def save(self, context, result):
        with open(self.filePath, "wb") as f:
            torch.save(result.value, f)

        return super(PytorchModel, self).save(
            context, Result.froms(value=self.filePath)
        )
