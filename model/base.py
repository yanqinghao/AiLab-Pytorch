# coding=utf-8

import os

import torch
import numpy as np
from torch import nn
from PIL import Image
from torchvision.transforms import functional as F
from torchtext.data.utils import get_tokenizer, ngrams_iterator
from suanpan.error import AppError
from suanpan.log import logger
from suanpan.model import Model
from utils import io

MODEL_FILE = "model.model"


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

    def train(self, X, y=None, **kwargs):
        if self.needTrain:
            self.model.fit(X, y, **kwargs)

    def predict(self, X):
        logger.info("model is an estimator, use predict()")
        predictions = self.model.predict(X)
        labelCount = 1 if len(predictions.shape) == 1 else predictions.shape[1]
        predictions = (predictions.T if labelCount > 1 else predictions.reshape(
            1, len(predictions)))

        return predictions

    def transform(self, X):
        logger.info("model is an transformer, use transform()")
        return self.model.transform(X)

    def evaluate(self, X, y):
        return self.model.score(X, y)

    def trainPredict(self, X, y=None):
        return self.model.fit_predict(X, y)

    def trainTransform(self, X, y=None):  # pylint: disable=unused-argument
        return (self.model.fit_transform(X) if y is None else self.model.fit_transform(X, y))

    def setModel(self, model):
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

    def predict(self, data, preprocess):
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
