# coding=utf-8
from __future__ import absolute_import, print_function

from suanpan.docker import DockerComponent as dc
from suanpan.docker.arguments import Folder
from suanpan.storage import storage


@dc.output(Folder(key="trainDir"))
@dc.output(Folder(key="valDir"))
@dc.output(Folder(key="testDir"))
def SPDogvsCat(context):
    args = context.args

    storage.download("common/data/cats_and_dogs/train", args.trainDir)
    storage.download("common/data/cats_and_dogs/validation", args.valDir)
    storage.download("common/data/cats_and_dogs/test", args.testDir)

    return args.trainDir, args.valDir, args.testDir


if __name__ == "__main__":
    SPDogvsCat()  # pylint: disable=no-value-for-parameter
