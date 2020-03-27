# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
#   Licensed under the Apache License, Version 2.0 (the "License").
#   You may not use this file except in compliance with the License.
#   A copy of the License is located at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   or in the "license" file accompanying this file. This file is distributed
#   on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
#   express or implied. See the License for the specific language governing
#   permissions and limitations under the License.
# ==============================================================================

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ConvBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(ConvBlock, self).__init__()
        self.layers = nn.Sequential()
        self.layers.add_module('Conv', nn.Conv2d(in_planes, out_planes,
            kernel_size=3, stride=1, padding=1, bias=False))
        self.layers.add_module('BatchNorm', nn.BatchNorm2d(out_planes))

        self.layers.add_module('ReLU', nn.ReLU(inplace=True))

        self.layers.add_module(
            'MaxPool', nn.MaxPool2d(kernel_size=2, stride=2, padding=0))

    def forward(self, x):
        out = self.layers(x)
        return out

class ConvNet_4_64(nn.Module):
    def __init__(self, inputW=80, inputH=80):
        super(ConvNet_4_64, self).__init__()

        conv_blocks = []
        ## 4 blocks, each block conv + bn + relu + maxpool, with filter 64
        conv_blocks.append(ConvBlock(3, 64))
        for i in range(3):
            conv_blocks.append(ConvBlock(64, 64))

        self.conv_blocks = nn.Sequential(*conv_blocks)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv_blocks(x)
        out = out.view(out.size(0),-1)
        return out


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)

        self.droprate = dropRate
        if self.droprate > 0:
            self.dropoutLayer = nn.Dropout(p=self.droprate)
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                               padding=0, bias=False) or None
    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))

        out = out if self.equalInOut else x
        out = self.conv1(out)
        if self.droprate > 0:
            out = self.dropoutLayer(out)
            #out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(self.relu2(self.bn2(out)))

        if not self.equalInOut:
            return self.convShortcut(x) + out
        else:
            return x + out


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(nb_layers):
            in_plances_arg = i == 0 and in_planes or out_planes
            stride_arg = i == 0 and stride or 1
            layers.append(block(in_plances_arg, out_planes, stride_arg, dropRate))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):
    def __init__(self, depth=28, widen_factor=10, dropRate=0.0, userelu=True, isCifar=False):
        super(WideResNet, self).__init__()
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert((depth - 4) % 6 == 0)
        n = (depth - 4) // 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1, padding=1, bias=False)

        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate) if isCifar \
                else  NetworkBlock(n, nChannels[0], nChannels[1], block, 2, dropRate)
        # 2nd block

        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)

        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True) if userelu else None
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.bn1(out)

        if self.relu is not None:
            out = self.relu(out)

        out = F.avg_pool2d(out, out.size(3))
        out = out.view(-1, self.nChannels)

        return out


def label_to_1hot(label, K):
    B, N = label.size()
    labels = []
    for i in range(K):
        labels.append((label == i).unsqueeze(2))
    return torch.cat(labels, -1).float()


class dni_linear(nn.Module):
    def __init__(self, input_dims, dni_hidden_size=1024):
        super(dni_linear, self).__init__()
        self.layer1 = nn.Sequential(
                      nn.Linear(input_dims, dni_hidden_size),
                      nn.ReLU(),
                      nn.BatchNorm1d(dni_hidden_size)
                      )
        self.layer2 = nn.Sequential(
                      nn.Linear(dni_hidden_size, dni_hidden_size),
                      nn.ReLU(),
                      nn.BatchNorm1d(dni_hidden_size)
                      )
        self.layer3 = nn.Linear(dni_hidden_size, input_dims)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        return out


class LinearDiag(nn.Module):
    def __init__(self, num_features, bias=False):
        super(LinearDiag, self).__init__()
        weight = torch.FloatTensor(num_features).fill_(1) # initialize to the identity transform
        self.weight = nn.Parameter(weight, requires_grad=True)

        if bias:
            bias = torch.FloatTensor(num_features).fill_(0)
            self.bias = nn.Parameter(bias, requires_grad=True)
        else:
            self.register_parameter('bias', None)

    def forward(self, X):
        assert(X.dim()==2 and X.size(1)==self.weight.size(0))
        out = X * self.weight.expand_as(X)
        if self.bias is not None:
            out = out + self.bias.expand_as(out)
        return out


class FeatExemplarAvgBlock(nn.Module):
    def __init__(self, nFeat):
        super(FeatExemplarAvgBlock, self).__init__()

    def forward(self, features_train, labels_train):
        labels_train_transposed = labels_train.transpose(1,2)
        # B x nK x nT @ B x nT x nC = B x nK x nC
        weight_novel = torch.bmm(labels_train_transposed, features_train)
        weight_novel = weight_novel.div(
            labels_train_transposed.sum(dim=2, keepdim=True).expand_as(weight_novel))
        return weight_novel


def get_featnet(architecture, inputW=80, inputH=80):
    # if cifar dataset, the last 2 blocks of WRN should be without stride
    isCifar = (inputW == 32) or (inputH == 32)
    if architecture == 'WRN_28_10':
        net = WideResNet(28, 10, isCifar=isCifar)
        return net, net.nChannels

    elif architecture == 'ConvNet_4_64':
        return eval(architecture)(inputW, inputH), 64 * (inputH/2**4) * (inputW/2**4)

    else:
        raise ValueError('No such feature net available!')
