from __future__ import division

import torch
import torch.nn as nn
import torch.functional as F
from torch.autograd import Variable
import numpy as np


def parse_cfg(cfg_file):
    """
    Parse a cfg file that defines the layers for the network.
    Return a list of blocks that are represented with dictionaries.
    """

    file = open(cfg_file, "r")
    lines = file.read().split("\n")
    lines = [x.rstrip().lstrip() for x in lines if len(x) > 0 and x[0] != "#"]

    block = {}
    blocks = []

    for line in lines:
        if line[0] == "[":  # we have started a new block
            if len(block) != 0:  # the previous block is still in block
                blocks.append(block)
                block = {}
            block["type"] = line[1:-1].rstrip()
        else:
            key, value = line.split("=")
            block[key.rstrip()] = value.lstrip()

    blocks.append(block)
    return blocks


class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()


class DetectionLayer(nn.Module):
    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors


def create_layers(blocks):
    net_hypers = blocks[0]
    layers = nn.ModuleList()
    prev_kernel_depth = 3  # initialized to 3 for 3 channel rgb input
    output_kernel_depths = []

    for i, b in enumerate(blocks[1:]):
        module = nn.Sequential()

        if b["type"] == "convolutional":
            activation = b["activation"]
            batch_normalize = b.get("batch_normalize", 0)
            bias = True if batch_normalize == 0 else False

            filters = int(b["filters"])
            size = int(b["size"])
            stride = int(b["stride"])
            padding = int(b["pad"])

            pad = (size - 1) // 2 if padding else 0

            conv_layer = nn.Conv2d(
                prev_kernel_depth, filters, size, stride, pad, bias=bias
            )
            module.add_module(f"conv_{i}", conv_layer)

            if batch_normalize:
                bn = nn.BatchNorm2d(filters)
                module.add_module(f"batch_norm_{i}", bn)

            if activation == "leaky":
                activation_layer = nn.LeakyReLU(0.1, inplace=True)
                module.add_module(f"leaky_{i}", activation_layer)

        elif b["type"] == "upsample":
            stride = int(b["stride"])
            up = nn.Upsample(scale_factor=stride, mode="bilinear")
            module.add_module(f"upsample_{i}", up)

        elif b["type"] == "route":
            b["layers"] = b["layers"].split(", ")
            start = int(b["layers"][0])
            end = int(b["layers"][1]) if 1 < len(b["layers"]) else 0

            if start > 0:
                start -= i
            if end > 0:
                end -= i
            route = EmptyLayer()
            module.add_module(f"route_{i}", route)

            if end < 0:
                filters = (
                    output_kernel_depths[i + start] + output_kernel_depths[i + end]
                )
            else:
                filters = output_kernel_depths[i + start]

        elif b["type"] == "shortcut":
            shortcut = EmptyLayer()
            module.add_module(f"shortcut_{i}", shortcut)

        elif b["type"] == "yolo":
            mask = b["mask"].split(",")
            mask = [int(x) for x in mask]

            anchors = b["anchors"].split(",")
            anchors = [int(x) for x in anchors]
            anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in mask]

            detection = DetectionLayer(anchors)
            module.add_module(f"Detection_{i}", detection)

        layers.append(module)
        prev_kernel_depth = filters
        output_kernel_depths.append(filters)

    return net_hypers, layers


class Darknet(nn.Module):
    def __init__(self, cfgfile):
        super(Darknet, self).__init__()
        self.blocks = parse_cfg(cfgfile)
        self.net_hypers, self.layers = create_layers(self.blocks)

    def forward(self, x, CUDA):
        modules = self.blocks[1:]
        outputs = {}

        write = 0
        for i, module in enumerate(modules):
            module_type = module["type"]

            if module_type == "convolutional" or module_type == "upsample":
                x = self.layers[i](x)
            elif module_type == 'route':
                
