# coding=utf-8

import os
import copy
import time
import zipfile
import torch
import numpy as np
import pandas as pd
from torch import nn
from PIL import Image
from torchvision.transforms import functional as F
from torchtext.data.utils import get_tokenizer, ngrams_iterator
from suanpan.error import AppError
from suanpan.storage import storage
from suanpan.log import logger
from suanpan.model import Model
from utils import io

MODEL_FILE = "model.layers"


class PytorchModel(Model):
    def __init__(self):
        super().__init__()
        self.needTrain = None

    def load(self, path):
        self.model = io.load_model(os.path.join(path, MODEL_FILE))
        return self

    def save(self, path):
        io.dump_model(self.model, os.path.join(path, MODEL_FILE))
        return path

    def prepare(self, modelClass, **kwargs):
        self.needTrain = kwargs.pop("needTrain", True)
        self.model = modelClass(**kwargs)

    # def train(self, X, y=None, **kwargs):
    #     if self.needTrain:
    #         self.model.fit(X, y, **kwargs)

    # def predict(self, X):
    #     logger.info("model is an estimator, use predict()")
    #     predictions = self.model.predict(X)
    #     labelCount = 1 if len(predictions.shape) == 1 else predictions.shape[1]
    #     predictions = (predictions.T if labelCount > 1 else predictions.reshape(
    #         1, len(predictions)))

    #     return predictions

    def transform(self, X):
        logger.info("model is an transformer, use transform()")
        return self.model.transform(X)

    def evaluate(self, X, y):
        return self.model.score(X, y)

    def trainPredict(self, X, y=None):
        return self.model.fit_predict(X, y)

    def trainTransform(self, X, y=None):  # pylint: disable=unused-argument
        return (self.model.fit_transform(X) if y is None else self.model.fit_transform(X, y))

    def set_model(self, model):
        self.model = model

    def image_preprocess(self, X):
        image = (Image.fromarray(np.uint8(X[:, :, ::-1]))
                 if len(X.shape) == 3 else Image.fromarray(np.uint8(X)))
        image = F.resize(image, (*self.model.input_size[:2], ))
        image = F.center_crop(image, (*self.model.input_size[:2], ))
        data = F.to_tensor(image)
        data = data.unsqueeze(0)

        return data

    def text_preprocess(self, X):
        vocab = self.model.vocab
        ngrams = self.model.NGRAMS
        tokenizer = get_tokenizer("basic_english")
        text = torch.tensor([vocab[token] for token in ngrams_iterator(tokenizer(X), ngrams)])
        offset_flag = False
        for name, layer in self.model.layers.items():
            if isinstance(layer[0], nn.EmbeddingBag):
                offset_flag = True
                break
        if offset_flag:
            return {"input": text, "offsets": torch.tensor([0])}
        else:
            return text

    def predict_stream(self, data, preprocess):
        if preprocess == "text":
            processedData = self.text_preprocess(data)
        elif preprocess == "image":
            processedData = self.image_preprocess(data)
        else:
            raise ("The input data is not recognized")
        class_names = list(self.model.class_to_idx.keys())
        # Device configuration
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Test the model
        # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
        self.model.eval()
        with torch.no_grad():
            prediction = torch.tensor([], dtype=torch.long)

            if isinstance(processedData, torch.Tensor):
                processedData = processedData.to(device)
            elif isinstance(processedData, dict):
                for name, value in processedData.items():
                    processedData[name] = value.to(device)
            else:
                raise ("Wrong input type")
            if isinstance(processedData, torch.Tensor):
                outputs = self.model(processedData)
            elif isinstance(processedData, dict):
                x = processedData.pop("input")
                outputs = self.model(x, **processedData)
                processedData["input"] = x
            else:
                raise ("Wrong model input")
            _, predicted = torch.max(outputs.data, 1)
            prediction = torch.cat((prediction, predicted), 0)

        return [class_names[i] for i in prediction.tolist()]

    def set_epoch(self, num_epochs):
        self.num_epochs = num_epochs

    def set_scheduler(self, scheduler):
        self.scheduler = scheduler

    def set_device(self, device):
        self.device = device
        self.model = self.model.to(device)

    def set_dataloader(self, dataloader):
        self.dataloader = dataloader

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def set_criterion(self, criterion):
        self.criterion = criterion

    def train(self):
        best_acc = 0.0
        since = time.time()
        best_model_wts = copy.deepcopy(self.model.state_dict())
        for epoch in range(self.num_epochs):
            logger.info("Epoch {}/{}".format(epoch + 1, self.num_epochs))
            for phase in ["train", "val"]:
                if phase == "train":
                    if self.scheduler:
                        self.scheduler.step()
                    self.model.train()
                else:
                    self.model.eval()
                running_loss = 0.0
                running_corrects = 0
                running_steps = len(self.dataloader[phase])
                for i, (data, labels, paths) in enumerate(self.dataloader[phase]):
                    if i % 100 == 0:
                        logger.info("train {} batch".format(i))
                    if isinstance(data, torch.Tensor):
                        data = data.to(self.device)
                    elif isinstance(data, dict):
                        for name, value in data.items():
                            data[name] = value.to(self.device)
                    else:
                        raise ("Wrong input type")
                    labels = labels.to(self.device)
                    self.optimizer.zero_grad()
                    with torch.set_grad_enabled(phase == "train"):
                        if isinstance(data, torch.Tensor):
                            outputs = self.model(data)
                        elif isinstance(data, dict):
                            x = data.pop("input")
                            outputs = self.model(x, **data)
                            data["input"] = x
                        else:
                            raise ("Wrong model input")
                        _, preds = torch.max(outputs, 1)
                        loss = self.criterion(outputs, labels)
                        if phase == "train":
                            loss.backward()
                            self.optimizer.step()
                    running_loss += loss.item()
                    running_corrects += torch.sum(preds == labels.data).double() / labels.size(0)
                epoch_loss = running_loss / running_steps
                epoch_acc = running_corrects.double().item() / running_steps
                logger.info("{} Loss: {:.4f} Acc: {:.4f}".format(phase, epoch_loss, epoch_acc))
                if phase == "val" and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(self.model.state_dict())
        time_elapsed = time.time() - since
        logger.info("Training complete in {:.0f}m {:.0f}s".format(time_elapsed // 60,
                                                                  time_elapsed % 60))
        logger.info("Best val Acc: {:4f}".format(best_acc))
        self.model.load_state_dict(best_model_wts)
        return self

    def predict(self, labeled=False):
        folder = "/tmp/pred_data/"
        pathtmp = ""
        class_names = list(self.model.class_to_idx.keys())
        self.model.eval()
        with torch.no_grad():
            prediction = torch.tensor([], dtype=torch.long)
            filepath = []
            filelabel = []
            for data, labels, paths in self.dataloader["predict"]:
                if isinstance(data, torch.Tensor):
                    data = data.to(self.device)
                elif isinstance(data, dict):
                    for name, value in data.items():
                        data[name] = value.to(self.device)
                else:
                    raise ("Wrong input type")
                if isinstance(data, torch.Tensor):
                    outputs = self.model(data)
                elif isinstance(data, dict):
                    x = data.pop("input")
                    outputs = self.model(x, **data)
                    data["input"] = x
                else:
                    raise ("Wrong model input")
                _, predicted = torch.max(outputs.data, 1)
                prediction = torch.cat((prediction, predicted), 0)
                if isinstance(list(paths)[0], str):
                    filepath = filepath + [
                        os.path.join(*i.split(storage.delimiter)[6:]) for i in list(paths)
                    ]
                else:
                    filepath = filepath + list(paths.numpy())
                if labels[0] is None:
                    filelabel = filelabel + list(labels)
                elif not isinstance(labels[0], str):
                    filelabel = filelabel + list(labels.numpy())
                else:
                    filelabel = filelabel + list(labels)
                if not pathtmp:
                    pathtmp = list(paths)[0]
                if isinstance(data, torch.Tensor) and len(data.size()) == 4:
                    for j in range(data.size()[0]):
                        if isinstance(pathtmp, str):
                            save_path = os.path.join(
                                folder,
                                os.path.join(
                                    *os.path.split(paths[j])[0].split(storage.delimiter)[6:]),
                                class_names[predicted[j]],
                                os.path.split(paths[j])[1],
                            )
                        else:
                            save_path = os.path.join(folder, class_names[predicted[j]],
                                                     "{}.png".format(paths[j]))
                        if isinstance(pathtmp, str):
                            img = Image.open(os.path.join("/sp_data/", paths[j]))
                        else:
                            img = F.to_pil_image(data[j].cpu())

                        if not os.path.exists(os.path.split(save_path)[0]):
                            os.makedirs(os.path.split(save_path)[0])
                        img.save(save_path)

            if labeled:
                df = pd.DataFrame({
                    "file path or index": filepath,
                    "label": [class_names[i] for i in filelabel],
                    "predictions": [class_names[i] for i in prediction.tolist()],
                })
            else:
                df = pd.DataFrame({
                    "file path or index": filepath,
                    "predictions": [class_names[i] for i in prediction.tolist()],
                })
            # if isinstance(pathtmp, str):
            #     pathlist = pathtmp.split(storage.delimiter)
            #     folder = os.path.join(folder, *pathlist[:6])
        os.makedirs(folder, exist_ok=True)
        prediction_images = "/tmp/zip_res"
        os.makedirs(prediction_images, exist_ok=True)
        zipf = zipfile.ZipFile(os.path.join(prediction_images, 'prediction-result.zip'), 'w',
                               zipfile.ZIP_DEFLATED)
        self.zipdir(folder, zipf)
        zipf.close()
        return prediction_images, df

    def zipdir(self, path, ziph):
        for root, dirs, files in os.walk(path):
            for file in files:
                ziph.write(os.path.join(root, file))
