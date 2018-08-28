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
Layer classes for adding new layers to models
"""

import ast
import mxnet as mx

from . import consts


def _parse_bool_string(string):
    return string.lower() in ['true', '1']


def _get_input_symbol(layer_dict):
    input_id = layer_dict[consts.INPUTS][0][0] + layer_dict[consts.ELEMENTS_OFFSET]
    input_name = layer_dict[consts.ID2NAME][input_id]
    input_symbol = layer_dict[consts.PREV_SYMBOLS][input_name]
    return input_symbol


class LayerFactory(object):
    """
    Base class for neural network layer.

    :param str name: Name of the resulting layer.
    """
    def __init__(self, name):
        self.attributes = {consts.NAME: name}
        self.output = False
        self.SymbolClass = None
        self.layer_type = None

    def create_layer(self, layer_input):
        """
        Add layer to symbol graph provided in layer_input.

        :param layer_input: Symbol to add layer to.
        :type layer_input: :class:`mx.symbol.Symbol`
        :rtype: :class:`mx.symbol.Symbol`
        """
        return self.SymbolClass(data=layer_input, **self.attributes)

    @staticmethod
    def _from_dict(layer_dict):
        """
        Function to return layer object from dictionary.

        :rtype: :class:`LayerFactory`
        :raises ValueError: If layer type is not supported by package.
        """
        # Table of layer names to layer classes
        class_dict = {
            'FullyConnected': FullyConnected,
            'Convolution': Convolution,
            'Activation': Activation,
            'Pooling': Pooling,
            'Flatten': Flatten,
            'SoftmaxOutput': SoftmaxOutput,
            'Dropout': Dropout,
            'Concat': Concat,
            'BatchNorm': BatchNorm,
            '_Plus': Add,
            'SVMOutput': SVMOutput,
            'Embedding': Embedding,
            'mean': Mean,
            'null': Data
        }
        try:
            constructor_function = class_dict[layer_dict[consts.OPERATION]]._from_dict
        except KeyError:
            raise ValueError('Unsupported layer type found in dict: ' + layer_dict[consts.OPERATION])
        return constructor_function(layer_dict)


class FullyConnected(LayerFactory):
    """
    Class for fully connected layers.

    :param str name: Name of the resulting layer.
    :param int num_hidden: Number of hidden nodes of the output.
    :param boolean no_bias: Whether to disable bias parameter.
    """
    def __init__(self, name, num_hidden, no_bias=False):
        super(FullyConnected, self).__init__(name=name)
        self.attributes['num_hidden'] = num_hidden
        self.attributes['no_bias'] = no_bias
        self.layer_type = consts.LayerType.FULLYCONNECTED.value
        self.SymbolClass = mx.sym.FullyConnected

    def _from_dict(layer_dict):
        name = layer_dict[consts.NAME]
        num_hidden = int(layer_dict[consts.ATTRIBUTES]['num_hidden'])
        no_bias = layer_dict[consts.ATTRIBUTES].get('no_bias', 'False') == 'True'
        input_symbol = _get_input_symbol(layer_dict)
        return FullyConnected(name=name, num_hidden=num_hidden, no_bias=no_bias), input_symbol


class SoftmaxOutput(LayerFactory):
    """Class for softmax output layers.

    :param str name: Name of the resulting layer.
    """
    def __init__(self, name):
        super(SoftmaxOutput, self).__init__(name=name)
        self.layer_type = consts.LayerType.SOFTMAXOUTPUT.value
        self.SymbolClass = mx.sym.SoftmaxOutput
        self.output = True

    def _from_dict(layer_dict):
        name = layer_dict[consts.NAME]
        input_symbol = _get_input_symbol(layer_dict)
        return SoftmaxOutput(name=name), input_symbol


class Activation(LayerFactory):
    """Class for activation layers.

    :param name: Name of the resulting layer.
    :param str act_type: Activation function to be applied. Can be selected among:

        - 'relu'
        - 'sigmoid'
        - 'softrelu'
        - 'tanh'
    """
    def __init__(self, name, act_type='relu'):
        super(Activation, self).__init__(name=name)
        self.attributes['act_type'] = act_type
        self.layer_type = consts.LayerType.ACTIVATION.value
        self.SymbolClass = mx.sym.Activation

    def _from_dict(layer_dict):
        name = layer_dict[consts.NAME]
        act_type = layer_dict[consts.ATTRIBUTES]['act_type']
        input_symbol = _get_input_symbol(layer_dict)
        return Activation(name=name, act_type=act_type), input_symbol


class Convolution(LayerFactory):
    """Class for convolution layers.

    :param str name: Name of the resulting layer.
    :param tuple kernel: Convolution kernel size.
    :param int num_filter: Convolution filter (channel) number.
    :param tuple dilate: Convolution dilate.
    :param tuple pad: Pad for convolution.
    :param str cudnn_tune: Whether to pick convolution algorithm by running performance test. Can be selected among:

        - 'None'
        - 'fastest'
        - 'limited_workspace'
        - 'off'
    :param tuple stride: Convolution stride.
    :param boolean no_bias: Whether to disable bias parameter.
    :param boolean cudnn_off: Turn off cudnn for this layer.
    :param layout: Set layout for input, output and weight. Empty for default layout: NCW for 1d, NCHW for 2d and NCDHW
                   for 3d. Can be selected among:

                   - 'None'
                   - 'NCDHW'
                   - 'NCHW'
                   - 'NCW'
                   - 'NDHWC'
                   - 'NHWC'
    :param int num_group: Number of group partitions (Non-negative).
    """
    def __init__(self, name, kernel, num_filter, dilate=(), pad=(), cudnn_tune=None, stride=(), no_bias=False,
                 cudnn_off=False, layout=None, num_group=1):
        super(Convolution, self).__init__(name=name)
        self.attributes['kernel'] = kernel
        self.attributes['num_filter'] = num_filter
        self.attributes['dilate'] = dilate
        self.attributes['pad'] = pad
        self.attributes['cudnn_tune'] = cudnn_tune
        self.attributes['stride'] = stride
        self.attributes['no_bias'] = no_bias
        self.attributes['cudnn_off'] = cudnn_off
        self.attributes['layout'] = layout
        self.attributes['num_group'] = num_group
        self.layer_type = consts.LayerType.CONVOLUTION.value
        self.SymbolClass = mx.sym.Convolution

    def _from_dict(layer_dict):
        name = layer_dict[consts.NAME]
        kernel = ast.literal_eval(layer_dict[consts.ATTRIBUTES]['kernel'])
        pad = ast.literal_eval(layer_dict[consts.ATTRIBUTES].get('pad', '()'))
        stride = ast.literal_eval(layer_dict[consts.ATTRIBUTES].get('stride', '()'))
        dilate = ast.literal_eval(layer_dict[consts.ATTRIBUTES].get('dilate', '()'))
        num_filter = int(layer_dict[consts.ATTRIBUTES]['num_filter'])
        cudnn_tune = layer_dict[consts.ATTRIBUTES].get('cudnn_tune', 'None')
        no_bias = layer_dict[consts.ATTRIBUTES].get('no_bias', 'False') == 'True'
        cudnn_off = _parse_bool_string(layer_dict[consts.ATTRIBUTES].get('cudnn_off', 'False'))
        layout = layer_dict[consts.ATTRIBUTES].get('layout', None)
        num_group = int(layer_dict[consts.ATTRIBUTES].get('num_group', 1))
        input_symbol = _get_input_symbol(layer_dict)
        return Convolution(name=name, kernel=kernel, num_group=num_group, num_filter=num_filter, dilate=dilate,
                           pad=pad, cudnn_tune=cudnn_tune, stride=stride, no_bias=no_bias,
                           cudnn_off=cudnn_off, layout=layout), input_symbol


class Data(LayerFactory):
    """Class for data layer.

    :param name: Name of the resulting layer.
    """
    def __init__(self, name):
        super(Data, self).__init__(name=name)
        self.layer_type = consts.LayerType.DATA.value

    def create_layer(self, layer_input=None):
        """
        Returns variable symbol with name 'data'.

        :param layer_input: Symbol to add layer to.
        :type layer_input: :class:`mx.symbol.Symbol`
        """
        return mx.sym.var(**self.attributes)

    def _from_dict(layer_dict):
        name = layer_dict[consts.NAME]
        data_name = layer_dict[consts.DATA]
        if name == data_name:
            return Data(name=name), None
        else:
            # nodes that have 'op' as 'null' will return None, None
            return None, None


class Pooling(LayerFactory):
    """Class for pooling layer.

    :param str name: Name of the resulting layer.
    :param str pool_type: Pooling type to be applied. Can be selected among:

        - 'avg'
        - 'max'
        - 'sum'
    :param tuple kernel: Pooling kernel size.
    :param tuple pad: Pad for pooling.
    :param str pooling_convention: Pooling convention to be applied. Can be selected among:

        - 'valid' (default)
        - 'full'
    :param tuple stride: Stride for pooling.
    :param boolean global_pool: Ignore kernel size, do global pooling based on current input feature map.
    """
    def __init__(self, pool_type, name, kernel=(), pad=(), pooling_convention='valid', stride=(),
                 global_pool=False):
        super(Pooling, self).__init__(name=name)
        self.attributes['kernel'] = kernel
        self.attributes['pool_type'] = pool_type
        self.attributes['pad'] = pad
        self.attributes['pooling_convention'] = pooling_convention
        self.attributes['stride'] = stride
        self.attributes['global_pool'] = global_pool
        self.layer_type = consts.LayerType.POOLING.value
        self.SymbolClass = mx.sym.Pooling

    def _from_dict(layer_dict):
        name = layer_dict[consts.NAME]
        kernel = ast.literal_eval(layer_dict[consts.ATTRIBUTES]['kernel'])
        pad = ast.literal_eval(layer_dict[consts.ATTRIBUTES].get('pad', '()'))
        pool_type = layer_dict[consts.ATTRIBUTES]['pool_type']
        pooling_convention = layer_dict[consts.ATTRIBUTES].get('pooling_convention', 'valid')
        stride = ast.literal_eval(layer_dict[consts.ATTRIBUTES].get('stride', '()'))
        global_pool = layer_dict[consts.ATTRIBUTES].get('global_pool', 'False') == 'True'
        input_symbol = _get_input_symbol(layer_dict)
        return Pooling(name=name, kernel=kernel, pad=pad, pool_type=pool_type,
                       pooling_convention=pooling_convention, stride=stride, global_pool=global_pool), input_symbol


class Flatten(LayerFactory):
    """Class for flatten layer.

    :param str name: Name of the resulting layer.
    """
    def __init__(self, name):
        super(Flatten, self).__init__(name=name)
        self.layer_type = consts.LayerType.FLATTEN.value
        self.SymbolClass = mx.sym.flatten

    def _from_dict(layer_dict):
        name = layer_dict[consts.NAME]
        input_symbol = _get_input_symbol(layer_dict)
        return Flatten(name=name), input_symbol


class Dropout(LayerFactory):
    """Class for dropout layer.

    :param str name: Name of the resulting layer.
    :param float p: Fraction of the input that gets dropped out during training time.
    """
    def __init__(self, name, p=0.5):
        super(Dropout, self).__init__(name=name)
        self.attributes['p'] = p
        self.layer_type = consts.LayerType.DROPOUT.value
        self.SymbolClass = mx.sym.Dropout

    def _from_dict(layer_dict):
        name = layer_dict[consts.NAME]
        default_p = 0.5  # default dropout fraction
        p = float(layer_dict[consts.ATTRIBUTES].get('p', default_p))
        input_symbol = _get_input_symbol(layer_dict)
        return Dropout(name=name, p=p), input_symbol


class Concat(LayerFactory):
    """Class for concat layer.

    :param str name: Name of the resulting layer.
    :param int dim: The dimension to be concatenated.
    """
    def __init__(self, name, dim=1):
        super(Concat, self).__init__(name=name)
        self.attributes['dim'] = dim
        self.layer_type = consts.LayerType.CONCAT.value
        self.SymbolClass = mx.sym.concat

    def create_layer(self, layer_input):
        """
        Add layer to symbol graph provided in layer_input.

        :param layer_input: List of symbols to have their output arrays concatenated.
        :type layer_input: list[:class:`Symbol`]
        :rtype: :class:`mx.symbol.Symbol`
        """
        data = layer_input if type(layer_input) == list else [layer_input]
        return mx.sym.concat(*data, **self.attributes)

    def _from_dict(layer_dict):
        name = layer_dict[consts.NAME]
        dim = int(layer_dict[consts.ATTRIBUTES]['dim'])
        input_symbol = []
        for input_tuple in layer_dict[consts.INPUTS][:-1]:
            input_id = input_tuple[0]
            input_name = layer_dict[consts.ID2NAME][input_id]
            input_symbol.append(layer_dict[consts.PREV_SYMBOLS][input_name])
        return Concat(name=name, dim=dim), input_symbol


class BatchNorm(LayerFactory):
    """Class for batch normalisation layer.

    :param str name: Name of the resulting layer.
    :param float eps: Epsilon to prevent div 0.
    :param boolean fix_gamma: Fix gamma while training.
    :param float momentum: Momentum for moving average.
    :param boolean use_global_stats: Whether to use global moving statistics instead of local batch-norm. This will
                                     force change batch-norm into a scale shift operator.
    """
    def __init__(self, name, eps=0.001, fix_gamma=True, momentum=0.9, use_global_stats=False):
        super(BatchNorm, self).__init__(name=name)
        self.attributes['eps'] = eps
        self.attributes['fix_gamma'] = fix_gamma
        self.attributes['momentum'] = momentum
        self.attributes['use_global_stats'] = use_global_stats
        self.layer_type = consts.LayerType.BATCHNORM.value
        self.SymbolClass = mx.sym.BatchNorm

    def _from_dict(layer_dict):
        name = layer_dict[consts.NAME]
        eps = float(layer_dict[consts.ATTRIBUTES].get('eps', 0.001))
        momentum = float(layer_dict[consts.ATTRIBUTES].get('momentum', 0.9))
        fix_gamma = layer_dict[consts.ATTRIBUTES].get('fix_gamma', 'True') == 'True'
        use_global_stats = layer_dict[consts.ATTRIBUTES].get('use_global_stats', 'False') == 'True'
        input_symbol = _get_input_symbol(layer_dict)
        return BatchNorm(name=name, eps=eps, fix_gamma=fix_gamma, momentum=momentum,
                         use_global_stats=use_global_stats), input_symbol


class Add(LayerFactory):
    """Class for add layer.

    :param str name: Name of the resulting layer.
    """
    def __init__(self, name):
        super(Add, self).__init__(name=name)
        self.layer_type = consts.LayerType.PLUS.value

    def create_layer(self, layer_input):
        """
        Add layer to symbol graph provided in layer_input.

        :param layer_input: List of two symbols to perform addition on.
        :type layer_input: list[:class:`mx.symbol.Symbol`]
        """
        if len(layer_input) != 2:
            raise ValueError('layer_input must be a list of 2 Symbols'.format(len(layer_input)))
        return layer_input[0].__add__(layer_input[1])

    def _from_dict(layer_dict):
        name = layer_dict[consts.NAME]
        input_symbol = _get_input_symbol(layer_dict)
        return Add(name=name), input_symbol


class SVMOutput(LayerFactory):
    """Class for SVM output layer.

    :param str name: Name of the resulting layer.
    :param label: Class label for the input data.
    :type label: :class:`Symbol`
    :param float margin: The loss function penalises outputs that lie outside this margin. Default margin is 1.
    :param float regularization_coefficient: Regularization parameter for the SVM. This balances the tradeoff between
                                             coefficient size and error.
    :param boolean use_linear: Whether to use L1-SVM objective. L2-SVM objective is used by default.
    """
    def __init__(self, name, label=None, margin=1, regularization_coefficient=1, use_linear=0):
        super(SVMOutput, self).__init__(name=name)
        self.attributes['label'] = label
        self.attributes['margin'] = margin
        self.attributes['regularization_coefficient'] = regularization_coefficient
        self.attributes['use_linear'] = use_linear
        self.layer_type = consts.LayerType.SVMOUTPUT.value
        self.output = True
        self.SymbolClass = mx.sym.SVMOutput

    def _from_dict(layer_dict):
        name = layer_dict[consts.NAME]
        margin = float(layer_dict[consts.ATTRIBUTES].get('margin', 1))
        regularization_coefficient = float(layer_dict[consts.ATTRIBUTES].get('regularization_coefficient', 1))
        use_linear = layer_dict[consts.ATTRIBUTES]['use_linear'] == 'True' or \
            int(layer_dict[consts.ATTRIBUTES]['use_linear']) == 1
        input_symbol = _get_input_symbol(layer_dict)
        return SVMOutput(name=name, margin=margin, regularization_coefficient=regularization_coefficient,
                         use_linear=use_linear), input_symbol


class Embedding(LayerFactory):
    """Class for Embedding layer. Maps integer indices to vector representations (embeddings).

    :param str name: Name of the resulting layer.
    :param int input_dim: Vocabulary size of the input indices.
    :param int output_dim: Dimension of the embedding vectors.
    :param weight: The embedding weight matrix.
    :type weight: :class:`mx.symbol.Symbol`
    :param dtype: Data type of weight.
    :type dtype: {'float16', 'float32', 'float64', 'int32', 'int64', 'int8', 'uint8'}, default='float32'
    """
    def __init__(self, name, input_dim, output_dim, weight=None, dtype='float32'):
        super(Embedding, self).__init__(name=name)
        self.attributes['input_dim'] = input_dim
        self.attributes['output_dim'] = output_dim
        self.attributes['weight'] = weight
        self.attributes['dtype'] = dtype
        self.layer_type = consts.LayerType.EMBEDDING.value
        self.SymbolClass = mx.sym.Embedding

    def _from_dict(layer_dict):
        name = layer_dict[consts.NAME]
        input_dim = int(layer_dict[consts.ATTRIBUTES]['input_dim'])
        output_dim = int(layer_dict[consts.ATTRIBUTES]['output_dim'])
        dtype = layer_dict[consts.ATTRIBUTES].get('dtype', 'float32')
        input_symbol = _get_input_symbol(layer_dict)
        return Embedding(name=name, input_dim=input_dim, output_dim=output_dim, dtype=dtype), input_symbol


class Mean(LayerFactory):
    """Class for mean layer. Computes the mean of array elements over given axis.

    :param str name: Name of the resulting layer.
    :param axis: The axis along which to perform the reduction.
    :type axis: Shape or None, optional, default=None
    :param boolean keepdims: If this is set to True, the reduced axis are left in the result as dimension with size one.
    :param boolean exclude: Whether to perform reduction on axis that are NOT in axis instead.
    """
    def __init__(self, name, axis=[], keepdims=0, exclude=0):
        super(Mean, self).__init__(name=name)
        self.attributes['axis'] = axis
        self.attributes['keepdims'] = keepdims
        self.attributes['exclude'] = exclude
        self.layer_type = consts.LayerType.MEAN.value
        self.SymbolClass = mx.sym.mean

    def _from_dict(layer_dict):
        name = layer_dict[consts.NAME]
        keepdims = _parse_bool_string(layer_dict[consts.ATTRIBUTES].get('keepdims', 'False'))
        exclude = _parse_bool_string(layer_dict[consts.ATTRIBUTES].get('exclude', 'False'))
        axis = eval(layer_dict[consts.ATTRIBUTES]['axis'])
        input_symbol = _get_input_symbol(layer_dict)
        return Mean(name=name, keepdims=keepdims, exclude=exclude, axis=axis), input_symbol
