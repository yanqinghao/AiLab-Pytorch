"""
Created on Sun Aug 11 00:00:00 2019
@author: Yan Qinghao
"""
import os
import time
import threading
import queue
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import softmax
from suanpan.utils import image
from suanpan.screenshots import screenshots
from suanpan import asyncio


class ScreenshotsThread(threading.Thread):
    def __init__(self, target=None, tag=True, last_time=None):
        super(ScreenshotsThread, self).__init__()
        self.q = queue.Queue()
        self.tag = tag
        self.last_time = last_time

    def put(self, data):
        self.q.put(data)

    def run(self):
        while True:
            if self.tag or self.q.qsize():
                data = self.q.get()
                if data["status"] == "quit":
                    break
                if self._target:
                    self._target[data["type"]](*data["data"])
            else:
                break


class Visualization(ScreenshotsThread):
    """
        Base Visualization Class
    """

    def __init__(self, model, target=None, tag=True, last_time=None):
        super(Visualization, self).__init__()
        self.model = model
        self.layers = model.layers
        self.model.eval()
        self.outputs = {}

    def hook_layer(self, selected_layer):
        def hook_function(module, input, output):
            # Gets the conv output of the selected filter (from selected layer)
            self.outputs[selected_layer] = output.detach()

        # Hook the selected layer
        layer = getattr(self.model, selected_layer)
        return layer.register_forward_hook(hook_function)

    def plot_cnn_layer(self, data, file_name):
        fig = plt.figure()
        row_col = int(np.ceil(np.sqrt(len(data))))
        ax = fig.subplots(row_col, row_col)
        for i, axi in enumerate(ax.reshape(-1)):
            if i >= len(data):
                break
            axi.axis("off")
            axi.imshow(np.round(data[i].cpu().data.numpy() * 225), cmap="gray")
        fig.savefig(file_name)
        plt.close(fig)
        return file_name

    def plot_linear_layer(self, data, file_name):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.pie(
            softmax(data.cpu().data.numpy()),
            labels=list(self.model.class_to_idx.keys()),
            autopct="%1.1f%%",
            shadow=True,
            startangle=140,
        )
        fig.savefig(file_name)
        plt.close(fig)
        return file_name


class CNNLayerVisualization(Visualization):
    """
        Produces a set of images at a specific layer, each
        image represents a filter output. use screenshot
        function to display.
    """

    def __init__(self, model, selected_layer, target=None, tag=True, last_time=None):
        super(CNNLayerVisualization, self).__init__(model)
        self.selected_layer = selected_layer

    def plot_layer(self, data, paths):
        folder = "/out_data/"
        handle = self.hook_layer(self.selected_layer)
        out = self.model(data)
        handle.remove()
        cnn_out = self.outputs[self.selected_layer]
        pathtmp = None
        for cnn_fillter, path in zip(cnn_out, paths):
            if not pathtmp:
                pathtmp = path
            if isinstance(path, str):
                file_name = os.path.join(folder, path)
            else:
                paths_img = "{}.png".format(path)
                file_name = os.path.join(folder, paths_img)
            if not os.path.exists(os.path.split(file_name)[0]):
                os.makedirs(os.path.split(file_name)[0])
            self.plot_cnn_layer(cnn_fillter, file_name)
            img = image.read(file_name)
            if not self.last_time:
                screenshots.save(img)
            elif (time.time() - self.last_time) > 1:
                screenshots.save(img)
            else:
                time.sleep(1)
                screenshots.save(img)
            self.last_time = time.time()
        if isinstance(pathtmp, str):
            pathlist = pathtmp.split("/")
            output = os.path.join(folder, *pathlist[:6])
        else:
            output = "/out_data"
        return output


class CNNNNVisualization(Visualization):
    """
        Produces a set of images at each layer, each
        image represents a filter output. use screenshot
        function to display.
    """

    def __init__(self, model, target=None, tag=True, last_time=None):
        super(CNNNNVisualization, self).__init__(model)
        self._target = {"layer": self.plot_each_layer, "log": self.training_log}

    def plot_each_layer(self, data, paths):
        folder = "/out_data/"
        handles = {}
        for layer in self.layers:
            handles[layer[0]] = self.hook_layer(layer[0])
        out = self.model(data[0].unsqueeze_(0))
        for handle in handles.items():
            handle[1].remove()
        outputs = self.outputs
        for layer_name, layer_outputs in outputs.items():
            if len(layer_outputs.size()) == 4:
                for layer_output, path in zip(layer_outputs, [paths[0]]):
                    if isinstance(path, str):
                        file_name = os.path.join(folder, path)
                    else:
                        paths_img = "{}.png".format(path)
                        file_name = os.path.join(folder, paths_img)
                    if not os.path.exists(os.path.split(file_name)[0]):
                        os.makedirs(os.path.split(file_name)[0])
                    self.plot_cnn_layer(layer_output, file_name)
                    img = image.read(file_name)
                    if not self.last_time:
                        screenshots.save(img)
                    elif (time.time() - self.last_time) > 1:
                        screenshots.save(img)
                    else:
                        time.sleep(1)
                        screenshots.save(img)
                    self.last_time = time.time()
            elif len(layer_outputs.size()) == 2:
                for layer_output, path in zip(layer_outputs, [paths[0]]):
                    if isinstance(path, str):
                        file_name = os.path.join(folder, path)
                    else:
                        paths_img = "{}.png".format(path)
                        file_name = os.path.join(folder, paths_img)
                    if not os.path.exists(os.path.split(file_name)[0]):
                        os.makedirs(os.path.split(file_name)[0])
                    self.plot_linear_layer(layer_output, file_name)
                    img = image.read(file_name)
                    if not self.last_time:
                        screenshots.save(img)
                    elif (time.time() - self.last_time) > 1:
                        screenshots.save(img)
                    else:
                        time.sleep(1)
                        screenshots.save(img)
                    self.last_time = time.time()
            else:
                pass
        return time.time()

    def training_log(self, log):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        epochs = log["epoch"]
        acc = log["train_acc"]
        val_acc = log["val_acc"]
        ax.plot(epochs, acc, "b", label="Training acc")
        ax.plot(epochs, val_acc, "r", label="Validation acc")
        ax.text(0, 0.95, "Training acc", fontsize=10, color="b")
        ax.text(0, 0.9, "Validation acc", fontsize=10, color="r")
        length = ((len(epochs) - 1) // 5 + 1) * 5
        ax.axis([0, length, 0, 1])
        fig.canvas.draw()

        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        plt.close(fig)
        if not self.last_time:
            screenshots.save(data)
        elif (time.time() - self.last_time) > 1:
            screenshots.save(data)
        else:
            time.sleep(1)
            screenshots.save(data)
        self.last_time = time.time()

        return time.time()

