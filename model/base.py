# coding=utf-8

import math
import os
from functools import partial

import joblib
from hyperopt import fmin, tpe
from sklearn.model_selection import train_test_split
from suanpan.error import AppError
from suanpan.log import logger
from suanpan.model import Model

MODEL_FILE = "model.model"


class SklearnModel(Model):
    def __init__(self):
        super().__init__()
        self.needTrain = None

    def prepare(self, modelClass, **kwargs):
        self.needTrain = kwargs.pop("needTrain", True)
        self.model = modelClass(**kwargs)

    def load(self, path):
        self.model = joblib.load(os.path.join(path, MODEL_FILE))
        return self

    def save(self, path):
        joblib.dump(self.model, os.path.join(path, MODEL_FILE))
        return path

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
