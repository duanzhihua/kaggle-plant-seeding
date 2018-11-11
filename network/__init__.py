# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from torch import nn
from torchvision import models


def resnet50(classes, pretrain=True):
    if pretrain:
        net = models.resnet50(pretrained=True)
    else:
        net = models.resnet50()
# =============================================================================
#     resnet50 default:
#     (avgpool): AvgPool2d(kernel_size=7, stride=1, padding=0)
#     (fc): Linear(in_features=2048, out_features=1000, bias=True)
#     net.avgpool = nn.AdaptiveAvgPool2d(1)
#     class Linear(Module)-->in_features =2048
# =============================================================================
    net.fc = nn.Linear(net.fc.in_features, classes)
    return net
