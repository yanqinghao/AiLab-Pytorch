import torch
import torch.nn as nn


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
            if j is not None:
                if isinstance(j, nn.EmbeddingBag):
                    out = j(out, offsets)
                else:
                    out = j(out)
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
            out = getattr(torch, self.op)((*mid, ), **self.param)
        for _, j in self.layers.items():
            if j[0] is not None:
                if isinstance(j[0], nn.EmbeddingBag):
                    out = j[0](out, offsets)
                else:
                    out = j[0](out)
        return out
