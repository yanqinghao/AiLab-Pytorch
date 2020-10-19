# coding=utf-8

from suanpan import app
from suanpan.imports import imports

PytorchLayersModel = imports(f"args.{app.TYPE}.PytorchLayersModel")
PytorchDataset = imports(f"args.{app.TYPE}.PytorchDataset")
PytorchDataloader = imports(f"args.{app.TYPE}.PytorchDataloader")
PytorchTransModel = imports(f"args.{app.TYPE}.PytorchTransModel")
PytorchOptimModel = imports(f"args.{app.TYPE}.PytorchOptimModel")
PytorchSchedulerModel = imports(f"args.{app.TYPE}.PytorchSchedulerModel")
PytorchFinetuningModel = imports(f"args.{app.TYPE}.PytorchFinetuningModel")
