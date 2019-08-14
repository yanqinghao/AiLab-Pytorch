# coding=utf-8
from __future__ import absolute_import, print_function


from suanpan.app.arguments import Folder, String
from app import app
from arguments import PytorchLayersModel, PytorchDataloader
from utils.visual import CNNLayerVisualization


@app.input(PytorchLayersModel(key="inputModel"))
@app.input(PytorchDataloader(key="inputLoader"))
@app.param(String(key="selectedLayer", default="Conv2D_0"))
@app.output(Folder(key="outputData"))
def SPCNNVisual(context):
    # 从 Context 中获取相关数据
    args = context.args
    # 查看上一节点发送的 args.inputData 数据
    model = args.inputModel
    selectedLayer = args.selectedLayer
    dataLoader = args.inputLoader
    cnnVisual = CNNLayerVisualization(model, selectedLayer)
    for data, _, paths in dataLoader:
        folder = cnnVisual.plot_layer(data, paths)
    return folder


if __name__ == "__main__":
    SPCNNVisual()
