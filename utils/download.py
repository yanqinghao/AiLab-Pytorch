from suanpan.storage import StorageProxy

name_to_file = {
    "vgg11": "vgg11-bbd30ac9.pth",
    "vgg11_bn": "vgg11_bn-6002323d.pth",
    "vgg13": "vgg13-c768596a.pth",
    "vgg13_bn": "vgg13_bn-abd245e5.pth",
    "vgg16": "vgg16-397923af.pth",
    "vgg16_bn": "vgg16_bn-6c64b313.pth",
    "vgg19": "vgg19-dcbb9e9d.pth",
    "vgg19_bn": "vgg19_bn-c79401a0.pth",
    "alexnet": "alexnet-owt-4df8aa71.pth",
    "densenet161": "densenet161-8d451a50.pth",
    "googlenet": "googlenet-1378be20.pth",
    "inception_v3": "inception_v3_google-1a9a5a14.pth",
    "mnasnet1_0": "mnasnet1.0_top1_73.512-f206786ef8.pth",
    "mobilenet_v2": "mobilenet_v2-b0353104.pth",
    "resnet18": "resnet18-5c106cde.pth",
    "resnext50_32x4d": "resnext50_32x4d-7cdf4587.pth",
    "shufflenet_v2_x1_0": "shufflenetv2_x1-5666bf0f80.pth",
    "squeezenet1_0": "squeezenet1_0-a815701f.pth",
    "wide_resnet50_2": "wide_resnet50_2-95faca4d.pth",
}


def downloadPretrained(name, storageType):
    storage = StorageProxy(None, None)
    storage.setBackend(type=storageType)
    file_path = "common/model/pytorch/{}".format(name_to_file[name])
    local_path = "/root/.cache/torch/checkpoints/{}".format(name_to_file[name])
    return storage.download(file_path, local_path)
