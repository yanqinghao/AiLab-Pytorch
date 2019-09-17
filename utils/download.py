from suanpan.storage import StorageProxy

name_to_file = {"vgg16": "vgg16-397923af.pth"}


def downloadPretrained(name, storageType):
    storage = StorageProxy(None, None)
    storage.setBackend(type=storageType)
    file_path = "common/model/pytorch/{}".format(name_to_file[name])
    local_path = "/root/.cache/torch/checkpoints/{}".format(name_to_file[name])
    return storage.download(file_path, local_path)
