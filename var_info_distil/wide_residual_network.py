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
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import kaiming_normal_

# This file describes the architecture of the wide residual network (Zagoruyko et al., https://arxiv.org/abs/1605.07146)
# used for the experiment. We note that this implementation of wide residual network was inspired by the author's
# original code (https://github.com/szagoruyko/wide-residual-networks).

# Hyper-parameters used for training the wide residual network in its official implementation. See
# https://github.com/szagoruyko/wide-residual-networks.

BATCH_SIZE = 128
LEARNING_RATE = 1e-1
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005
LEARNING_RATE_DECAY_MILESTONES = [60, 120, 160]
LEARNING_RATE_DECAY_FACTOR = 0.9


def conv1x1(in_channels, out_channels, stride=1):
    # convenient shortcuts for constructing the network
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=False, stride=stride)


def conv3x3(in_channels, out_channels, stride=1):
    # convenient shortcuts for constructing the network
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False, stride=stride)


class BasicBlock(nn.Module):
    # This class represents a basic residual block that is used repeteadely for constructing the wide residual network.
    # It consists of two of batch normalization, two of recified linear unit, and two of convolutional layers.
    def __init__(self, in_channels, out_channels, stride, drop_rate):
        super(BasicBlock, self).__init__()

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = conv3x3(in_channels, out_channels, stride)

        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels, 1)

        # If the number of channels corresponding to the input and the output is different, we use an additional 1x1
        # convolutional layer as a residual connection between the input and the output.
        self.is_equal_in_out = in_channels == out_channels
        if not self.is_equal_in_out:
            self.conv_shortcut = conv1x1(in_channels, out_channels, stride)

        self.drop_rate = drop_rate

    def forward(self, x):
        if self.is_equal_in_out:
            out = self.conv1(self.relu1(self.bn1(x)))
            out = self.relu2(self.bn2(out))
            if self.drop_rate > 0:
                out = F.dropout(out, p=self.drop_rate, training=self.training)

            out = self.conv2(out)
            out = torch.add(out, x)

        else:
            x = self.relu1(self.bn1(x))
            out = self.conv1(x)
            out = self.relu2(self.bn2(out))
            if self.drop_rate > 0:
                out = F.dropout(out, p=self.drop_rate, training=self.training)

            out = self.conv2(out)
            out = torch.add(out, self.conv_shortcut(x))

        return out


class GroupBlock(nn.Module):
    # This class represents the group of BasicBlock modules for building wide residual networks. It is a sequence of
    # BasicBlocks, where the first block modifies the number of channel for the input.
    def __init__(self, num_blocks, in_channels, out_channels, stride, drop_rate):
        super(GroupBlock, self).__init__()

        self.layer = self._make_layer(in_channels, out_channels, num_blocks, stride, drop_rate)

    def _make_layer(self, in_channels, out_channels, num_blocks, stride, drop_rate):
        layers = []
        for i in range(int(num_blocks)):
            if i == 0:
                _in_channels, _out_channels, _stride = (in_channels, out_channels, stride)
            else:
                _in_channels, _out_channels, _stride = (out_channels, out_channels, 1)

            layers.append(BasicBlock(_in_channels, _out_channels, _stride, drop_rate))

        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResidualNetwork(nn.Module):
    """
    Wide residual network for classification of CIFAR-10 dataset (Zagoruyko et al., https://arxiv.org/abs/1605.07146).

    :param int depth: depth parameter for constructing the neural network.
    :param int width: width parameter for constructing the neural network.
    :param int num_classes: number of classes in the classification dataset.
    :param float drop_rate: dropout ratio for the dropout layers.
    """

    def __init__(self, depth, width, num_classes=10, drop_rate=0.0):
        super(WideResidualNetwork, self).__init__()
        self.depth = depth
        self.width = width
        self.layerwise_num_channels = [int(16 * width), int(32 * width), int(64 * width)]
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        # The wide residual network assigns 4 modules outside the GroupBlock modules.
        # There exist three GroupBlock modules and each one has depth of twice the number of its blocks.
        # Hence, the number of blocks per group can be inversely calculated from the depth factor.
        self.num_blocks_per_group = (self.depth - 4) / 6

        # Convolutional layer before any GroupBlock module with fixed number of channels.
        self.conv1 = conv3x3(3, 16, 1)

        # 1st GroupBlock
        self.layer1 = GroupBlock(
            num_blocks=self.num_blocks_per_group,
            in_channels=16,
            out_channels=self.layerwise_num_channels[0],
            stride=1,
            drop_rate=self.drop_rate,
        )

        # 2nd GroupBlock
        self.layer2 = GroupBlock(
            num_blocks=self.num_blocks_per_group,
            in_channels=self.layerwise_num_channels[0],
            out_channels=self.layerwise_num_channels[1],
            stride=2,
            drop_rate=self.drop_rate,
        )

        # 3rd GroupBlock
        self.layer3 = GroupBlock(
            num_blocks=self.num_blocks_per_group,
            in_channels=self.layerwise_num_channels[1],
            out_channels=self.layerwise_num_channels[2],
            stride=2,
            drop_rate=self.drop_rate,
        )

        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(self.layerwise_num_channels[2])
        self.relu = nn.ReLU(inplace=True)
        self.avg_pool2d = nn.AvgPool2d(kernel_size=8)
        self.fc = nn.Linear(self.layerwise_num_channels[2], self.num_classes)
        self.view_dim = self.layerwise_num_channels[2]

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.relu(self.bn1(out))
        out = self.avg_pool2d(out)
        out = out.view(-1, self.view_dim)
        out = self.fc(out)

        return (out,)


class TeacherWideResidualNetwork(WideResidualNetwork):
    """
    WideResidualNetwork modified as a teacher network for transfer learning with variational information distillation.
    In order to provide target for the student network, we modify the original wide residual network by allowing to 
    extract the features from intermediate groups.
    """

    def __init__(self, depth, width, num_classes=10, drop_rate=0.0, load_path=None):
        super(TeacherWideResidualNetwork, self).__init__(depth, width, num_classes, drop_rate)
        # Load the pre-trained weights for the teacher model when the path is provided.
        if load_path is not None:
            state_dict = torch.load(load_path, map_location=next(self.parameters()).device)
            self.load_state_dict(state_dict)

    def forward(self, x):
        out = self.conv1(x)
        
        # teacher network provides its intermediate representation (z1, z2, z3) as the knowledge to "teach" for a 
        # student network.
        z1 = out = self.layer1(out)
        z2 = out = self.layer2(out)
        z3 = out = self.layer3(out)

        out = self.relu(self.bn1(out))
        out = self.avg_pool2d(out)
        out = out.view(-1, self.view_dim)
        out = self.fc(out)

        return out, (z1, z2, z3)


class StudentWideResidualNetwork(WideResidualNetwork):
    """
    WideResidualNetwork modified as a student network for transfer learning with variational information distillation.
    Branches and variance parameters are added for variational approximation of features from the teacher network.

    :param int depth: depth parameter for constructing the neural network
    :param int width: width parameter for constructing the neural network
    :param int teacher_width: width parameter for constructing the branchs of the neural network (corresponds to dimension
    of features from the teacher network).
    :param int num_classes: number of classes in the classification dataset.
    :param float drop_rate: dropout ratio for the dropout layers.
    """

    def __init__(self, depth, width, teacher_width, num_classes=10, drop_rate=0.0):
        super(StudentWideResidualNetwork, self).__init__(depth, width, num_classes, drop_rate)
        self.teacher_layerwise_num_channels = [
            int(16 * teacher_width),
            int(32 * teacher_width),
            int(64 * teacher_width),
        ]

        self.branch1 = nn.Sequential(
            conv1x1(self.layerwise_num_channels[0], self.teacher_layerwise_num_channels[0]),
            conv1x1(self.teacher_layerwise_num_channels[0], self.teacher_layerwise_num_channels[0]),
        )
        self.branch2 = nn.Sequential(
            conv1x1(self.layerwise_num_channels[1], self.teacher_layerwise_num_channels[1]),
            conv1x1(self.teacher_layerwise_num_channels[1], self.teacher_layerwise_num_channels[1]),
        )
        self.branch3 = nn.Sequential(
            conv1x1(self.layerwise_num_channels[2], self.teacher_layerwise_num_channels[2]),
            conv1x1(self.teacher_layerwise_num_channels[2], self.teacher_layerwise_num_channels[2]),
        )

        # variance are represented as a softplus function applied to "variance parameters".
        init_variance_param_value = self._variance_param_to_variance(torch.tensor(5.0))
        self.variance_param1 = nn.Parameter(
            torch.full((self.teacher_layerwise_num_channels[0], 1, 1), init_variance_param_value)
        )
        self.variance_param2 = nn.Parameter(
            torch.full((self.teacher_layerwise_num_channels[1], 1, 1), init_variance_param_value)
        )
        self.variance_param3 = nn.Parameter(
            torch.full((self.teacher_layerwise_num_channels[2], 1, 1), init_variance_param_value)
        )

    def _variance_to_variance_param(self, variance):
        """
        Convert variance to corresponding variance parameter by inverse of the softplus function.

        :param torch.FloatTensor variance: the target variance for obtaining the variance parameter
        """
        return torch.log(torch.exp(variance) - 1.0)

    def _variance_param_to_variance(self, variance_param):
        """
        Convert the variance parameter to corresponding variance by the softplus function.

        :param torch.FloatTensor variance_param: the target variance parameter for obtaining the variance
        """
        return torch.log(torch.exp(variance_param) + 1.0)

    def forward(self, x):
        out = self.conv1(x)
        z1 = out = self.layer1(out)
        z2 = out = self.layer2(out)
        z3 = out = self.layer3(out)
        out = self.relu(self.bn1(out))
        out = self.avg_pool2d(out)
        out = out.view(-1, self.view_dim)
        out = self.fc(out)

        branch1_out = self.branch1(z1)
        branch2_out = self.branch2(z2)
        branch3_out = self.branch3(z3)

        variance1 = self._variance_param_to_variance(self.variance_param1)
        variance2 = self._variance_param_to_variance(self.variance_param2)
        variance3 = self._variance_param_to_variance(self.variance_param3)

        # If the input has an additional dimension for mini-batch, resize the variance to match its dimension
        if x.dim() == 4:
            variance1 = variance1.unsqueeze(0)
            variance2 = variance2.unsqueeze(0)
            variance3 = variance3.unsqueeze(0)

        return out, ((branch1_out, variance1), (branch2_out, variance2), (branch3_out, variance3))
