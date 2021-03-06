"""
Created on Sun Aug 21 2019
@author: Yan Qinghao
transforms
"""
# coding=utf-8
from __future__ import absolute_import, print_function

import suanpan
from suanpan.app import app
from suanpan.app.arguments import Float, ListOfFloat, Bool
from args import PytorchOptimModel


@app.param(Float(key="lr", default=0.001))
@app.param(ListOfFloat(key="betas", default=[0.9, 0.999]))
@app.param(Float(key="eps", default=1e-08))
@app.param(Float(key="weightDecay", default=0))
@app.param(Bool(key="amsgrad", default=False))
@app.output(PytorchOptimModel(key="outputModel"))
def SPAdam(context):
    """
    Implements Adam algorithm.
    """
    args = context.args
    data = {
        "name": "Adam",
        "param": {
            "lr": args.lr,
            "betas": (*args.betas,),
            "eps": args.eps,
            "weight_decay": args.weightDecay,
            "amsgrad": args.amsgrad,
        },
    }
    return data


if __name__ == "__main__":
    suanpan.run(app)
