"""
Created on Sun Aug 11 00:00:00 2019
@author: Yan Qinghao
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import softmax
from suanpan.utils import image
from suanpan.screenshots import screenshots


class Visualization(object):
    """
        Base Visualization Class
    """

    def __init__(self, model):
        self.model = model
        self.layers = model.layers
        self.model.eval()
        # Create the folder to export images if not exists
        self.outputs = {}

    def hook_layer(self, selected_layer):
        def hook_function(module, input, output):
            # Gets the conv output of the selected filter (from selected layer)
            self.outputs[selected_layer] = output.detach()

        # Hook the selected layer
        layer = getattr(self.model, selected_layer)
        layer.register_forward_hook(hook_function)

    def plot_cnn_layer(self, data, file_name):
        fig = plt.figure()
        row_col = int(np.ceil(np.sqrt(len(data))))
        ax = fig.subplots(row_col, row_col)
        for i, axi in enumerate(ax.reshape(-1)):
            if i >= len(data):
                break
            axi.axis("off")
            axi.imshow(np.round(data[i].data.numpy() * 225), cmap="gray")
        fig.savefig(file_name)
        plt.close(fig)
        return file_name

    def plot_linear_layer(self, data, file_name):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.pie(
            softmax(data.data.numpy()),
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

    def __init__(self, model, selected_layer):
        super(CNNLayerVisualization, self).__init__(model)
        self.selected_layer = selected_layer
        # Create the folder to export images if not exists

    def plot_layer(self, data, paths):
        folder = "/out_data/"
        self.hook_layer(self.selected_layer)
        out = self.model(data)
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
        screenshots.save(img)
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

    def __init__(self, model):
        super(CNNNNVisualization, self).__init__(model)

    def plot_each_layer(self, data, paths):
        folder = "/out_data/"
        for layer in self.layers:
            self.hook_layer(layer[0])
        out = self.model(data[0].unsqueeze_(0))
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
                    screenshots.save(img)
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
                    screenshots.save(img)
            else:
                pass
