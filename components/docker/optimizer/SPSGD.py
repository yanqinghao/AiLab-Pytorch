"""
Created on Sun Aug 21 2019
@author: Yan Qinghao
transforms
"""
# coding=utf-8
from __future__ import absolute_import, print_function


from suanpan.app import app
from suanpan.app.arguments import Folder, Float, Bool
from arguments import PytorchOptimModel


@app.input(Folder(key="inputData"))
@app.param(Float(key="lr", default=0.001))
@app.param(Float(key="momentum", default=0))
@app.param(Float(key="dampening", default=0))
@app.param(Float(key="weightDecay", default=0))
@app.param(Bool(key="nesterov", default=False))
@app.output(PytorchOptimModel(key="outputModel"))
def SPSGD(context):
    """
    Implements stochastic gradient descent (optionally with momentum).
    """
    args = context.args

    data = {
        "name": "SGD",
        "param": {
            "lr": args.lr,
            "momentum": args.momentum,
            "dampening": args.dampening,
            "weight_decay": args.weightDecay,
            "nesterov": args.nesterov,
        },
    }

    return data


if __name__ == "__main__":
    SPSGD()
