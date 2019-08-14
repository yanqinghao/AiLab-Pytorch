# coding=utf-8
from __future__ import absolute_import, print_function

import copy
import time
import torch
import torch.nn as nn

from suanpan.app.arguments import Int, String
from suanpan.log import logger
from app import app
from arguments import (
    PytorchLayersModel,
    PytorchDataloader,
    PytorchOptimModel,
    PytorchSchedulerModel,
)
from utils import trainingLog
from utils.visual import CNNNNVisualization


@app.input(PytorchLayersModel(key="inputModel"))
@app.input(PytorchDataloader(key="inputTrainLoader"))
@app.input(PytorchDataloader(key="inputValLoader"))
@app.output(PytorchOptimModel(key="inputOptimModel"))
@app.output(PytorchSchedulerModel(key="inputSchedulerModel"))
@app.param(Int(key="epochs", default=5))
@app.param(
    String(
        key="lossFunction",
        default="CrossEntropyLoss",
        help="L1Loss, NLLLoss, KLDivLoss, MSELoss, BCELoss, BCEWithLogitsLoss, NLLLoss2d, "
        "CosineEmbeddingLoss, CTCLoss, HingeEmbeddingLoss, MarginRankingLoss, "
        "MultiLabelMarginLoss, MultiLabelSoftMarginLoss, MultiMarginLoss, SmoothL1Loss,"
        "SoftMarginLoss, CrossEntropyLoss, TripletMarginLoss, PoissonNLLLoss",
    )
)
@app.output(PytorchLayersModel(key="outputModel"))
def SPTorchTrain(context):
    # 从 Context 中获取相关数据
    args = context.args
    # 查看上一节点发送的 args.inputData 数据
    model = args.inputModel
    trainLoader = args.inputTrainLoader
    valLoader = args.inputValLoader
    optimModel = args.inputOptimModel
    schedulerModel = args.inputSchedulerModel
    if valLoader:
        loader = {"train": trainLoader, "val": valLoader}
    else:
        logger.info("Use train_dataset as default val_dataset in training.")
        loader = {"train": trainLoader, "val": trainLoader}
    model.class_to_idx = trainLoader.dataset.class_to_idx
    log = {
        "epoch": [],
        "train_acc": [],
        "train_loss": [],
        "val_acc": [],
        "val_loss": [],
    }

    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    # Device configuration
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Hyper parameters
    num_epochs = args.epochs

    # Loss and optimizer
    criterion = getattr(nn, args.lossFunction)()
    if optimModel:
        logger.info("Use {} as optim function in training.".format(optimModel["name"]))
        optimizer = getattr(torch.optim, optimModel["name"])(
            model.parameters(), **optimModel["param"]
        )
    else:
        logger.info("Use Adam as default optim function in training.")
        optimizer = torch.optim.Adam(model.parameters())
    # Decay LR by scheduler
    if schedulerModel:
        logger.info(
            "Use {} as lr_scheduler in training.".format(schedulerModel["name"])
        )
        scheduler = getattr(torch.optim.lr_scheduler, schedulerModel["name"])(
            optimizer, **schedulerModel["param"]
        )
    else:
        logger.info("No model lr_scheduler is used in training.")

    for epoch in range(num_epochs):
        logger.info("Epoch {}/{}".format(epoch + 1, num_epochs))
        log["epoch"].append(epoch)
        for phase in ["train", "val"]:
            if phase == "train":
                if schedulerModel:
                    scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode
                cnnVisual = CNNNNVisualization(model)

            running_loss = 0.0
            running_corrects = 0
            running_steps = len(loader[phase])
            for i, (images, labels, paths) in enumerate(loader[phase]):
                images = images.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(images)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item()
                running_corrects += torch.sum(
                    preds == labels.data
                ).double() / images.size(0)
                if phase == "val":
                    cnnVisual.plot_each_layer(images, paths)

            epoch_loss = running_loss / running_steps
            epoch_acc = running_corrects.double() / running_steps
            log["{}_loss".format(phase)].append(epoch_loss)
            log["{}_acc".format(phase)].append(epoch_acc)
            logger.info(
                "{} Loss: {:.4f} Acc: {:.4f}".format(phase, epoch_loss, epoch_acc)
            )
            # deep copy the model
            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        trainingLog(log)

    time_elapsed = time.time() - since
    logger.info(
        "Training complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )
    logger.info("Best val Acc: {:4f}".format(best_acc))
    # load best model weights
    model.load_state_dict(best_model_wts)

    return model


if __name__ == "__main__":
    SPTorchTrain()
