from suanpan.storage import storage

name_to_file = {"vgg16": "vgg16-397923af.pth"}


def downloadPretrained(name):
    file_path = "common/model/pytorch/{}".format(name_to_file[name])
    local_path = "/root/.cache/torch/checkpoints/{}".format(name_to_file[name])
    return storage.download(file_path, local_path)
