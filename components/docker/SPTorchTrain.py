# coding=utf-8
from __future__ import absolute_import, print_function

import torch
import torch.nn as nn

from suanpan.docker import DockerComponent as dc
from suanpan.docker.arguments import Int, Float
from arguments import PytorchLayersModel, PytorchDataloader

@dc.input(PytorchLayersModel(key="inputModel"))
@dc.input(PytorchDataloader(key="inputLoader"))
@dc.param(Int(key="epochs", default=5))
@dc.param(Float(key="learningRate", default=0.001))
@dc.output(PytorchLayersModel(key="outputModel"))
def SPTrain(context):
    # 从 Context 中获取相关数据
    args = context.args
    # 查看上一节点发送的 args.inputData 数据
    model = args.inputModel
    train_loader = args.inputLoader

    # Device configuration
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Hyper parameters
    num_epochs = args.epochs
    learning_rate = args.learningRate

    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    total_step = len(train_loader)
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print(
                    "Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}".format(
                        epoch + 1, num_epochs, i + 1, total_step, loss.item()
                    )
                )

    return model


if __name__ == "__main__":
    SPTrain()
