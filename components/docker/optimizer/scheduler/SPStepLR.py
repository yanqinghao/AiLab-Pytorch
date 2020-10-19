# coding=utf-8
from __future__ import absolute_import, print_function

import suanpan
from suanpan.app import app
from suanpan.app.arguments import Float, Int
from args import PytorchSchedulerModel


@app.param(Int(key="stepSize", default=10))
@app.param(Float(key="gamma", default=0.1))
@app.param(Int(key="lastEpoch", default=-1))
@app.output(PytorchSchedulerModel(key="outputModel"))
def SPStepLR(context):
    args = context.args
    data = {
        "name": "StepLR",
        "param": {
            "step_size": args.stepSize,
            "gamma": args.gamma,
            "last_epoch": args.lastEpoch,
        },
    }
    return data


if __name__ == "__main__":
    suanpan.run(app)
