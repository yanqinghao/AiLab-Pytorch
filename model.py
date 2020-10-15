# coding=utf-8
from __future__ import absolute_import, print_function

import os
import torch
import numpy as np
from torch import nn
from PIL import Image
from torchvision.transforms import functional as F
from torchtext.data.utils import get_tokenizer, ngrams_iterator
from suanpan.model import Model as BaseModel
from arguments import PytorchLayersStreamModel


class PytorchModel(BaseModel):
    def load(self, path):
        model = PytorchLayersStreamModel("")
        model.filePath = os.path.join(path, "model.layers")
        self.model = model.format(None)
        return self.model

    def image_preprocess(self, X):
        image = (
            Image.fromarray(np.uint8(X[:, :, ::-1]))
            if len(X.shape) == 3
            else Image.fromarray(np.uint8(X))
        )
        image = F.resize(image, (*self.model.input_size[:2],))
        image = F.center_crop(image, (*self.model.input_size[:2],))
        data = F.to_tensor(image)
        data = data.unsqueeze(0)

        return data

    def text_preprocess(self, X):
        vocab = self.model.vocab
        ngrams = self.model.NGRAMS
        tokenizer = get_tokenizer("basic_english")
        text = torch.tensor(
            [vocab[token] for token in ngrams_iterator(tokenizer(X), ngrams)]
        )
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