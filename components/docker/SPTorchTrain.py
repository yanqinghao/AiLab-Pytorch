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
from utils.visual import CNNNNVisualization


@app.input(PytorchLayersModel(key="inputModel"))
@app.input(PytorchDataloader(key="inputTrainLoader"))
@app.input(PytorchDataloader(key="inputValLoader"))
@app.input(PytorchOptimModel(key="inputOptimModel"))
@app.input(PytorchSchedulerModel(key="inputSchedulerModel"))
@app.param(Int(key="__gpu", default=0))
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
    gpu = args.__gpu

    if valLoader:
        loader = {"train": trainLoader, "val": valLoader}
    else:
        logger.info("Use train_dataset as default val_dataset in training.")
        loader = {"train": trainLoader, "val": trainLoader}
    model.class_to_idx = trainLoader.dataset.class_to_idx
    model.vocab = (
        getattr(trainLoader.dataset, "get_vocab", None)()
        if getattr(trainLoader.dataset, "get_vocab", None)
        else None
    )
    model.NGRAMS = getattr(trainLoader.dataset, "NGRAMS", None)
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
    device = torch.device("cuda:0" if gpu > 0 else "cpu")

    logger.info("Use {} as device in training.".format("cuda:0" if gpu > 0 else "cpu"))

    model = model.to(device)

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
    cnnThreads = []
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
                cnnThreads.append(cnnVisual)
                cnnVisual.daemon = True
                cnnVisual.start()

            running_loss = 0.0
            running_corrects = 0
            running_steps = len(loader[phase])
            for i, (data, labels, paths) in enumerate(loader[phase]):
                if isinstance(data, torch.Tensor):
                    data = data.to(device)
                elif isinstance(data, dict):
                    for name, value in data.items():
                        data[name] = value.to(device)
                else:
                    raise ("Wrong input type")
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == "train"):
                    if isinstance(data, torch.Tensor):
                        outputs = model(data)
                    elif isinstance(data, dict):
                        x = data["input"]
                        params = data.pop("input")
                        outputs = model(x, **params)
                        data["input"] = x
                    else:
                        raise ("Wrong model input")
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item()
                running_corrects += torch.sum(
                    preds == labels.data
                ).double() / labels.size(0)
                if phase == "val" and i == 0 and isinstance(data, torch.Tensor):
                    cnnVisual.put(
                        {
                            "status": "running",
                            "type": "layer",
                            "data": (copy.deepcopy(data), copy.deepcopy(paths)),
                        }
                    )

            epoch_loss = running_loss / running_steps
            epoch_acc = running_corrects.double().item() / running_steps
            log["{}_loss".format(phase)].append(epoch_loss)
            log["{}_acc".format(phase)].append(epoch_acc)
            logger.info(
                "{} Loss: {:.4f} Acc: {:.4f}".format(phase, epoch_loss, epoch_acc)
            )
            # deep copy the model
            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        cnnVisual.put(
            {"status": "running", "type": "log", "data": (copy.deepcopy(log),)}
        )
        cnnVisual.put({"status": "quit"})

    for i in cnnThreads:
        i.tag = False

    for i in cnnThreads:
        i.join()

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
