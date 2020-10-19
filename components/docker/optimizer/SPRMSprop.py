"""
Created on Sun Aug 27 2019
@author: Yan Qinghao
transforms
"""
# coding=utf-8
from __future__ import absolute_import, print_function

import suanpan
from suanpan.app import app
from suanpan.app.arguments import Float, Bool
from args import PytorchOptimModel


@app.param(Float(key="lr", default=0.01))
@app.param(Float(key="alpha", default=0.99))
@app.param(Float(key="eps", default=1e-08))
@app.param(Float(key="weightDecay", default=0))
@app.param(Float(key="momentum", default=0))
@app.param(Bool(key="centered", default=False))
@app.output(PytorchOptimModel(key="outputModel"))
def SPRMSprop(context):
    """
    Implements stochastic gradient descent (optionally with momentum).
    """
    args = context.args
    data = {
        "name": "RMSprop",
        "param": {
            "lr": args.lr,
            "alpha": args.alpha,
            "eps": args.eps,
            "weight_decay": args.weightDecay,
            "momentum": args.momentum,
            "centered": args.centered,
        },
    }
    return data


if __name__ == "__main__":
    suanpan.run(app)
