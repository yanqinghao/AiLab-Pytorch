# coding=utf-8
from __future__ import absolute_import, print_function

import os

from suanpan.proxy import Proxy


class App(Proxy):
    MAPPING = {
        "spark": "suanpan.app.spark.SparkApp",
        "docker": "app.docker.DockerApp",
        "stream": "suanpan.app.stream.StreamApp",
    }

    def __init__(self, *args, **kwargs):
        super(App, self).__init__()
        self.setBackend(*args, **kwargs)

    @property
    def isSpark(self):
        return self.isType("spark")

    @property
    def isDocker(self):
        return self.isType("docker")

    @property
    def isStream(self):
        return self.isType("stream")

    def isType(self, appType):
        return getattr(self, self.TYPE_KEY, None) == appType


TYPE = os.environ.get("SP_APP_TYPE")
app = App(type=TYPE)
