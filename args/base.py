# coding=utf-8

from suanpan import path, runtime
from suanpan.app.arguments import File
from suanpan.components import Result
from suanpan.utils import json

from utils import io


class PytorchLayersModel(File):
    FILENAME = "model"
    FILETYPE = "layers"

    def transform(self, value):
        filePath = super().transform(value)
        if not filePath:
            return None
        _load = io.load_model if self.required else runtime.saferun(io.load_model)
        return _load(filePath)

    def save(self, result):
        path.mkdirs(self.filePath, parent=True)
        io.dump_model(result.value, self.filePath)
        return super().save(Result.froms(value=self.filePath))


class PytorchDataset(PytorchLayersModel):
    FILENAME = "dataset"
    FILETYPE = None


class PytorchDataloader(PytorchLayersModel):
    FILENAME = "dataloader"
    FILETYPE = None


class PytorchTransModel(PytorchLayersModel):
    FILETYPE = "transaug"


class PytorchOptimModel(File):
    FILENAME = "model"
    FILETYPE = "optim"

    def transform(self, value):
        filePath = super().transform(value)
        if not filePath:
            return None
        _load = json.load if self.required else runtime.saferun(json.load)
        return _load(filePath)

    def save(self, result):
        path.mkdirs(self.filePath, parent=True)
        json.dump(result.value, self.filePath)
        return super().save(Result.froms(value=self.filePath))


class PytorchSchedulerModel(PytorchOptimModel):
    FILETYPE = "scheduler"


class PytorchFinetuningModel(PytorchOptimModel):
    FILETYPE = "finetuning"
