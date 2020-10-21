import torch
import torch.nn as nn
import torch.nn.functional as F


class SPNet(nn.Module):
    def __init__(self, input_size, task):
        super(SPNet, self).__init__()
        self.layers = {}
        self.class_to_idx = None
        self.input_size = tuple(input_size) if input_size else None
        self.task = task

    def forward(self, x, offsets=None):
        out = x
        for _, j in self.layers.items():
            if j is not None:
                if isinstance(j, nn.EmbeddingBag):
                    out = j(out, offsets)
                else:
                    out = j(out)
        return out


class SPMathOP(nn.Module):
    def __init__(self, input_size, task, model_list, op, param=None):
        super(SPMathOP, self).__init__()
        self.layers = {}
        self.class_to_idx = None
        self.input_size = tuple(input_size) if input_size else None
        self.task = task
        self.model_list = model_list
        self.op = op
        self.param = param

    def forward(self, x, offsets=None):
        out = x
        mid = []
        for model in self.model_list:
            mid.append(model(out))
        if self.op == "add":
            out = getattr(torch, self.op)(*mid)
        elif self.op == "cat":
            out = getattr(torch, self.op)((*mid, ), **self.param)
        for _, j in self.layers.items():
            if j[0] is not None:
                if isinstance(j[0], nn.EmbeddingBag):
                    out = j[0](out, offsets)
                else:
                    out = j[0](out)
        return out


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size()[0], -1)


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


class Inception(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
        super(Inception, self).__init__()

        self.branch1 = BasicConv2d(in_channels, ch1x1, kernel_size=1)

        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, ch3x3red, kernel_size=1),
            BasicConv2d(ch3x3red, ch3x3, kernel_size=3, padding=1),
        )

        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels, ch5x5red, kernel_size=1),
            BasicConv2d(ch5x5red, ch5x5, kernel_size=3, padding=1),
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=True),
            BasicConv2d(in_channels, pool_proj, kernel_size=1),
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        outputs = [branch1, branch2, branch3, branch4]
        return torch.cat(outputs, 1)


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size,
                stride,
                padding,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True),
        )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class ResnetBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, groups=1, base_width=64, dilation=1):
        super(ResnetBlock, self).__init__()
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(conv1x1(inplanes, planes, stride), nn.BatchNorm2d(planes))
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)

        return out


class ResnetBlockV2(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, groups=1, base_width=64, dilation=1):
        super(ResnetBlockV2, self).__init__()
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride
        downsample = None
        if stride != 1 or inplanes != planes * self.expansion:
            downsample = nn.Sequential(
                conv1x1(inplanes, planes * self.expansion, stride),
                nn.BatchNorm2d(planes * self.expansion),
            )
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)

        return out


class VGGBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        conv2d_ks,
        padding,
        maxpool_ks,
        maxpool_stride,
        batch_norm=True,
        max_pool=True,
    ):
        super(VGGBlock, self).__init__()
        layers = []
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=conv2d_ks, padding=padding)
        if batch_norm:
            layers += [conv2d, nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)]
        else:
            layers += [conv2d, nn.ReLU(inplace=True)]
        if max_pool:
            layers += [nn.MaxPool2d(kernel_size=maxpool_ks, stride=maxpool_stride)]
        self.layer = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer(x)
        return x
