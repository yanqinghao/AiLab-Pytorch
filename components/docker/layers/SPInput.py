# coding=utf-8
from __future__ import absolute_import, print_function

from suanpan.docker import DockerComponent as dc
from suanpan.docker.arguments import Folder
from arguments import PytorchModel, SPNet

@dc.input(Folder(key="inputData"))
@dc.output(PytorchModel(key="outputModel"))
def SPInput(context):
    # 从 Context 中获取相关数据
    args = context.args
    # 查看上一节点发送的 args.inputData 数据
    model = SPNet()

    print(model.layers)

    return model


if __name__ == "__main__":
    SPInput()
