# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
"""
Model Handler constants
"""

from enum import Enum

OPERATION = 'op'
NO_OP = 'null'
LAYER_NAME = 'name'
NODES = 'nodes'
ELEMENTS_OFFSET = 'elements_offset'
PREV_SYMBOLS = 'prev_symbols'
ID2NAME = 'id2name'
LABEL_IDX = -2
DATA = 'data'
N = 'n'
SELF = 'self'
LAYER = 'layer_factory_list'
NAME = 'name'
ATTRIBUTES = 'attrs'
INPUTS = 'inputs'


class LayerType(Enum):
    FULLYCONNECTED = 'FullyConnected'
    CONVOLUTION = 'Convolution'
    ACTIVATION = 'Activation'
    POOLING = 'Pooling'
    FLATTEN = 'Flatten'
    SOFTMAXOUTPUT = 'SoftmaxOutput'
    DROPOUT = 'Dropout'
    CONCAT = 'Concat'
    BATCHNORM = 'BatchNorm'
    PLUS = '_Plus'
    SVMOUTPUT = 'SVMOutput'
    LRN = 'LRN'
    DATA = 'Data'
    MEAN = 'mean'
    EMBEDDING = 'Embedding'
