# coding=utf-8
from __future__ import absolute_import, print_function


from suanpan.app import app
from suanpan.app.arguments import Folder, Float, Int
from arguments import PytorchSchedulerModel


@app.input(Folder(key="inputData"))
@app.param(Int(key="stepSize", default=10))
@app.param(Float(key="gamma", default=0.1))
@app.param(Int(key="lastEpoch", default=-1))
@app.output(PytorchSchedulerModel(key="outputModel"))
def SPStepLR(context):
    # 从 Context 中获取相关数据
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
    SPStepLR()
