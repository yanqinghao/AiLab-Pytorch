# coding=utf-8
from __future__ import absolute_import, print_function

import torch

from suanpan.docker import DockerComponent as dc
from suanpan.docker.arguments import Json
from arguments import PytorchLayersModel


@dc.input(PytorchLayersModel(key="inputModel"))
@dc.output(Json(key="outputData"))
def SPTorchPredict(context):
    # 从 Context 中获取相关数据
    args = context.args
    # 查看上一节点发送的 args.inputData 数据
    model = args.inputModel
    test_loader = args.inputLoader

    # Device configuration
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Test the model
    model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(
            "Test Accuracy of the model on the 10000 test images: {} %".format(
                100 * correct / total
            )
        )

    return {"result": 100 * correct / total}


if __name__ == "__main__":
    SPTorchPredict()
