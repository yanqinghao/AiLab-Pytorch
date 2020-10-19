"""
Created on Sun Aug 21 2019
@author: Yan Qinghao
transforms
"""
# coding=utf-8
from __future__ import absolute_import, print_function

import suanpan
from suanpan.app import app
from suanpan.app.arguments import Float, Int, String, Bool
from args import PytorchSchedulerModel


@app.param(String(key="mode", default="min", help="One of min, max."))
@app.param(
    Float(
        key="factor",
        default=0.1,
        help="Factor by which the learning rate will be reduced.",
    ))
@app.param(
    Int(
        key="patience",
        default=10,
        help="Number of epochs with no improvement after which learning rate will be reduced.",
    ))
@app.param(
    Bool(
        key="verbose",
        default=False,
        help="If True, prints a message to stdout for each update.",
    ))
@app.param(
    Float(
        key="threshold",
        default=1e-4,
        help="Threshold for measuring the new optimum, to only focus on significant changes.",
    ))
@app.param(String(key="thresholdMode", default="rel", help="One of rel, abs."))
@app.param(
    Int(
        key="cooldown",
        default=0,
        help="Number of epochs to wait before resuming normal operation after lr has been reduced.",
    ))
@app.param(Float(key="minLr", default=0, help="A scalar or a list of scalars."))
@app.param(Float(key="eps", default=1e-8, help="Minimal decay applied to lr."))
@app.output(PytorchSchedulerModel(key="outputModel"))
def SPReduceLROnPlateau(context):
    """
    Reduce learning rate when a metric has stopped improving.
    """
    args = context.args
    data = {
        "name": "ReduceLROnPlateau",
        "param": {
            "mode": args.mode,
            "factor": args.factor,
            "patience": args.patience,
            "verbose": args.verbose,
            "threshold": args.threshold,
            "threshold_mode": args.thresholdMode,
            "cooldown": args.cooldown,
            "min_lr": args.minLr,
            "eps": args.eps,
        },
    }
    return data


if __name__ == "__main__":
    suanpan.run(app)
