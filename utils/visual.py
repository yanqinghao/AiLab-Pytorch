"""
Created on Sun Aug 11 00:00:00 2019
@author: Yan Qinghao
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from suanpan.utils import image
from suanpan.screenshots import screenshots
from suanpan.log import logger


class CNNLayerVisualization(object):
    """
        Produces an image that minimizes the loss of a convolution
        operation for a specific layer and filter
    """

    def __init__(self, model, selected_layer):
        self.model = model
        self.model.eval()
        self.selected_layer = selected_layer
        self.conv_output = None
        # Create the folder to export images if not exists

    def hook_layer(self):
        def hook_function(module, input, output):
            # Gets the conv output of the selected filter (from selected layer)
            self.conv_output = output.detach()

        # Hook the selected layer
        layer = getattr(self.model, self.selected_layer)
        layer.register_forward_hook(hook_function)

    def plotLayer(self, data, paths):
        folder = "/out_data/"
        self.hook_layer()
        out = self.model(data)
        cnn_out = self.conv_output
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
            fig = plt.figure()
            ax = fig.subplots(4, 4)
            for i, axi in enumerate(ax.reshape(-1)):
                axi.imshow(np.round(cnn_fillter[i].data.numpy() * 225), cmap="gray")
            fig.savefig(file_name)
            plt.close(fig)
        img = image.read(file_name)
        screenshots.save(img)
        if isinstance(pathtmp, str):
            pathlist = pathtmp.split("/")
            output = os.path.join(folder, *pathlist[:6])
        else:
            output = "/out_data"
        logger.info(output)
        return output
