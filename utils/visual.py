"""
Created on Sun Aug 11 00:00:00 2019
@author: Yan Qinghao
"""
import os
import string
import random
import time
import threading
import queue
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import softmax
from suanpan.utils import image
from suanpan.screenshots import screenshots, Screenshots
from suanpan.storage import storage
from suanpan import g


def random_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return "".join(random.choice(chars) for x in range(size))


class ScreenshotsThread(threading.Thread):
    def __init__(self, target=None, tag=True, last_time=None):
        super(ScreenshotsThread, self).__init__()
        self.q = queue.Queue()
        self.tag = tag
        self.last_time = last_time

    def put(self, data):
        self.q.put(data)

    def empty(self):
        if self.q.qsize():
            self.q.queue.clear()
            self.put({"status": "quit"})

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
        if "." not in selected_layer:
            layer = getattr(self.model, selected_layer)
        else:
            layer_parent = getattr(self.model, selected_layer.split(".")[0])
            layer_dict = dict(layer_parent.named_modules())
            if ".".join(selected_layer.split(".")[1:]) in layer_dict.keys():
                layer = layer_dict[".".join(selected_layer.split(".")[1:])]
            else:
                layer = layer_dict[".".join(selected_layer.split(".")[2:])]
        return layer.register_forward_hook(hook_function)

    def plot_cnn_layer(self, data, file_name):
        if not os.path.exists(os.path.split(file_name)[0]):
            os.makedirs(os.path.split(file_name)[0])
        if len(data) == 3:
            image.save(
                file_name,
                np.transpose(np.round(data.cpu().data.numpy() * 225), (1, 2, 0)),
            )
        else:
            fig = plt.figure()
            row_col = int(np.ceil(np.sqrt(len(data))))
            ax = fig.subplots(row_col, row_col)
            axArr = ax.reshape(-1) if isinstance(ax, np.ndarray) else [ax]
            for i, axi in enumerate(axArr):
                if i >= len(data):
                    break
                axi.axis("off")
                axi.imshow(np.round(data[i].cpu().data.numpy() * 225), cmap="gray")
            fig.savefig(file_name)
            plt.close(fig)

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
        elif (time.time() - self.last_time) >= 1:
            screenshots.save(img)
        else:
            time.sleep(1 - (time.time() - self.last_time))
            screenshots.save(img)
        self.last_time = time.time()
        if isinstance(pathtmp, str):
            pathlist = pathtmp.split(storage.delimiter)
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

    screenshots_dict = {}

    def __init__(self, model, target=None, tag=True, last_time=None):
        super(CNNNNVisualization, self).__init__(model)
        self._target = {"layer": self.plot_each_layer, "log": self.training_log}

    def plot_each_layer(self, data, paths):
        folder = "/out_data/"
        handles = {}
        self.outputs["Input"] = data[0].unsqueeze_(0)
        for layer_name, layer in self.layers.items():
            if layer_name != "Input":
                handles[layer_name] = self.hook_layer(layer_name)
        out = self.model(data[0].unsqueeze_(0))
        for handle in handles.items():
            handle[1].remove()
        outputs = self.outputs
        for layer_name, layer_outputs in outputs.items():
            if layer_name not in list(self.screenshots_dict.keys()):
                storage_name = self.layers[layer_name][1]
                self.screenshots_dict[layer_name] = createScreenshots(*storage_name)
            screenshots_node = self.screenshots_dict[layer_name]
            if len(layer_outputs.size()) == 4:
                for layer_output, path in zip(layer_outputs, [paths[0]]):
                    if isinstance(path, str):
                        file_name = os.path.join(
                            folder,
                            os.path.splitext(path)[0]
                            + "_"
                            + random_generator()
                            + os.path.splitext(path)[1],
                        )
                    else:
                        paths_img = "{}_{}.png".format(path, random_generator())
                        file_name = os.path.join(folder, paths_img)
                    if not os.path.exists(os.path.split(file_name)[0]):
                        os.makedirs(os.path.split(file_name)[0])
                    self.plot_cnn_layer(layer_output, file_name)
                    img = image.read(file_name)
                    if not self.last_time:
                        screenshots_node.save(img)
                    elif (time.time() - self.last_time) >= 1:
                        screenshots_node.save(img)
                    else:
                        time.sleep(1 - (time.time() - self.last_time))
                        screenshots_node.save(img)
            elif (
                len(layer_outputs.size()) == 2
                and layer_name == list(outputs.keys())[-1]
            ):
                for layer_output, path in zip(layer_outputs, [paths[0]]):
                    if isinstance(path, str):
                        file_name = os.path.join(
                            folder,
                            os.path.splitext(path)[0]
                            + "_"
                            + random_generator()
                            + os.path.splitext(path)[1],
                        )
                    else:
                        paths_img = "{}_{}.png".format(path, random_generator())
                        file_name = os.path.join(folder, paths_img)
                    if not os.path.exists(os.path.split(file_name)[0]):
                        os.makedirs(os.path.split(file_name)[0])
                    self.plot_linear_layer(layer_output, file_name)
                    img = image.read(file_name)
                    if not self.last_time:
                        screenshots_node.save(img)
                    elif (time.time() - self.last_time) >= 1:
                        screenshots_node.save(img)
                    else:
                        time.sleep(1 - (time.time() - self.last_time))
                        screenshots_node.save(img)
            else:
                pass
        self.last_time = time.time()
        return time.time()

    def training_log(self, log):
        screenshots.current.storageName, screenshots.current.thumbnail = (
            getScreenshotPath()
        )
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
        elif (time.time() - self.last_time) >= 1:
            screenshots.save(data)
        else:
            time.sleep(1 - (time.time() - self.last_time))
            screenshots.save(data)
        self.last_time = time.time()

        return time.time()

    def plot_each_layer_onenode(self, data, paths):
        folder = "/out_data/"
        handles = {}
        self.outputs["Input"] = data[0].unsqueeze_(0)
        for layer_name, layer in self.layers.items():
            if layer_name != "Input":
                handles[layer_name] = self.hook_layer(layer_name)
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
                    elif (time.time() - self.last_time) >= 1:
                        screenshots.save(img)
                    else:
                        time.sleep(1 - (time.time() - self.last_time))
                        screenshots.save(img)
                    self.last_time = time.time()
            elif (
                len(layer_outputs.size()) == 2
                and layer_name == list(outputs.keys())[-1]
            ):
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
                    elif (time.time() - self.last_time) >= 1:
                        screenshots.save(img)
                    else:
                        time.sleep(1 - (time.time() - self.last_time))
                        screenshots.save(img)
                    self.last_time = time.time()
            else:
                pass
        return time.time()

