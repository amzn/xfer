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
from unittest import TestCase

import mxnet as mx
import numpy as np

import xfer.model_handler.layer_factory as layer_factory


class TestLayerFactory(TestCase):
    def setUp(self):
        np.random.seed(1)
        self.data = mx.symbol.Variable('data')

    def tearDown(self):
        pass

    def test_parse_bool_string(self):
        for string in ['true', 'True', 'tRue', '1']:
            assert layer_factory._parse_bool_string(string)
        for string in ['false', 'False', 'falSe', 'f', 'F', '0', 'null']:
            assert not layer_factory._parse_bool_string(string)

    def test_from_dict_invalid_layer_type(self):
        layer_dict = {"op": "FakeLayer"}
        self.assertRaisesRegex(ValueError, 'Unsupported layer type found in dict: FakeLayer',
                               layer_factory.LayerFactory._from_dict, layer_dict)

    def _assert_from_dict(self, layer_type, output, reference_symbol):
        layer, input_symbol = self.LayerFactoryClass._from_dict(self.layer_dict)

        self._assert_layer(layer, layer_type, output)
        self._assert_input_symbol(input_symbol, reference_symbol)

    def _assert_layer(self, layer, layer_type, output):
        assert layer.__dict__ == layer_factory.LayerFactory._from_dict(self.layer_dict)[0].__dict__
        assert layer.attributes == self.attributes
        assert layer.layer_type == layer_type
        assert layer.output is output

    def _assert_input_symbol(self, input_symbol, reference_symbol):
        assert input_symbol == layer_factory.LayerFactory._from_dict(self.layer_dict)[1]
        assert input_symbol == reference_symbol


class TestFullyConnected(TestLayerFactory):
    def test_from_dict(self):
        self.layer_dict = {"op": "FullyConnected", "name": "fullyconnected0", "attrs": {"num_hidden": "4"},
                           'elements_offset': 0, 'id2name': {1: 'name1'}, 'prev_symbols': {'name1': 'sym'},
                           "inputs": [[1, 0, 0], [2, 0, 0], [3, 0, 0]]}
        self.attributes = {'name': 'fullyconnected0', 'num_hidden': 4, 'no_bias': False}
        self.LayerFactoryClass = layer_factory.FullyConnected

        self._assert_from_dict('FullyConnected', False, 'sym')

    def test_create_layer(self):
        layer = layer_factory.FullyConnected(name='fc1', num_hidden=10, no_bias=False)
        sym = layer.create_layer(self.data)

        assert sorted(sym.attr_dict().keys()) == sorted(['fc1', 'fc1_weight', 'fc1_bias'])
        assert sym.attr_dict()['fc1'] == {'num_hidden': '10', 'no_bias': 'False'}
        assert sym.attr_dict()['fc1_weight'] == {'num_hidden': '10', 'no_bias': 'False'}
        assert sym.attr_dict()['fc1_bias'] == {'num_hidden': '10', 'no_bias': 'False'}
        assert sym.get_internals().list_outputs() == ['data', 'fc1_weight', 'fc1_bias', 'fc1_output']


class TestSoftmaxOutput(TestLayerFactory):
    def test_from_dict(self):
        self.layer_dict = {"op": "SoftmaxOutput", "name": "softmaxoutput1", "inputs": [[1, 0, 0], [2, 0, 0]],
                           'elements_offset': 0, 'id2name': {1: 'name1'}, 'prev_symbols': {'name1': 'sym'}}
        self.attributes = {'name': 'softmaxoutput1'}
        self.LayerFactoryClass = layer_factory.SoftmaxOutput

        self._assert_from_dict('SoftmaxOutput', True, 'sym')

    def test_create_layer(self):
        layer = layer_factory.SoftmaxOutput(name='softmax')
        sym = layer.create_layer(self.data)

        assert sym.attr_dict() == {}
        assert sym.get_internals().list_outputs() == ['data', 'softmax_label', 'softmax_output']


class TestActivation(TestLayerFactory):
    def test_from_dict(self):
        self.layer_dict = {"op": "Activation", "name": "act2", "attrs": {"act_type": "relu"}, "inputs": [[1, 0, 0]],
                           'elements_offset': 0, 'id2name': {1: 'name1'}, 'prev_symbols': {'name1': 'sym'}}
        self.attributes = {'name': 'act2', 'act_type': 'relu'}
        self.LayerFactoryClass = layer_factory.Activation

        self._assert_from_dict('Activation', False, 'sym')

    def test_create_layer(self):
        layer = layer_factory.Activation(name='activation1', act_type='sigmoid')
        sym = layer.create_layer(self.data)

        assert sym.attr_dict() == {'activation1': {'act_type': 'sigmoid'}}
        assert sym.get_internals().list_outputs() == ['data', 'activation1_output']


class TestConvolution(TestLayerFactory):
    def test_from_dict(self):
        self.layer_dict = {"op": "Convolution", "name": "conv2", "attrs": {"kernel": "(4, 4)", "num_filter": "1"},
                           'elements_offset': 0, 'id2name': {1: 'name1'}, 'prev_symbols': {'name1': 'sym'},
                           "inputs": [[1, 0, 0], [2, 0, 0], [3, 0, 0]]}
        self.attributes = {'kernel': (4, 4), 'stride': (), 'layout': None, 'pad': (), 'num_filter': 1,
                           'dilate': (), 'no_bias': False, 'num_group': 1, 'cudnn_tune': 'None',
                           'cudnn_off': False, 'name': 'conv2'}
        self.LayerFactoryClass = layer_factory.Convolution

        self._assert_from_dict('Convolution', False, 'sym')

    def test_create_layer(self):
        layer = layer_factory.Convolution(name='conv1', kernel=(3, 3), num_filter=30)
        sym = layer.create_layer(self.data)

        assert sorted(sym.attr_dict().keys()) == sorted(['conv1', 'conv1_bias', 'conv1_weight'])
        assert sym.attr_dict()['conv1'] == {'layout': 'None', 'num_filter': '30', 'no_bias': 'False',
                                            'cudnn_tune': 'None', 'num_group': '1', 'dilate': '()', 'pad': '()',
                                            'cudnn_off': 'False', 'stride': '()', 'kernel': '(3, 3)'}
        assert sym.attr_dict()['conv1_bias'] == {'layout': 'None', 'num_filter': '30', 'no_bias': 'False',
                                                 'cudnn_tune': 'None', 'num_group': '1', 'dilate': '()', 'pad': '()',
                                                 'cudnn_off': 'False', 'stride': '()', 'kernel': '(3, 3)'}
        assert sym.attr_dict()['conv1_weight'] == {'layout': 'None', 'num_filter': '30', 'no_bias': 'False',
                                                   'cudnn_tune': 'None', 'num_group': '1', 'dilate': '()', 'pad': '()',
                                                   'cudnn_off': 'False', 'stride': '()', 'kernel': '(3, 3)'}
        assert sym.get_internals().list_outputs() == ['data', 'conv1_weight', 'conv1_bias', 'conv1_output']


class TestData(TestLayerFactory):
    def test_from_dict(self):
        self.layer_dict = {"op": "null", "name": "data", "inputs": [], 'data': 'data'}
        self.attributes = {'name': 'data'}
        self.LayerFactoryClass = layer_factory.Data

        self._assert_from_dict('Data', False, None)

    def test_from_dict_not_data(self):
        self.layer_dict = {"op": "null", "name": "not_data", "inputs": [], 'data': 'data'}
        layer, input_symbol = layer_factory.Data._from_dict(self.layer_dict)

        assert layer is None

    def test_create_layer(self):
        layer = layer_factory.Data(name='data')
        sym = layer.create_layer()

        assert sym.attr_dict() == {}
        assert sym.get_internals().list_outputs() == ['data']


class TestPooling(TestLayerFactory):
    def test_from_dict(self):
        self.layer_dict = {"op": "Pooling", "name": "pool1", "attrs": {"kernel": "(4, 4)", "pool_type": "sum"},
                           "inputs": [[1, 0, 0]], 'elements_offset': 0, 'id2name': {1: 'name1'},
                           'prev_symbols': {'name1': 'sym'}}
        self.attributes = {'pool_type': 'sum', 'stride': (), 'pad': (), 'global_pool': False, 'kernel': (4, 4),
                           'name': 'pool1', 'pooling_convention': 'valid'}
        self.LayerFactoryClass = layer_factory.Pooling

        self._assert_from_dict('Pooling', False, 'sym')

    def test_create_layer(self):
        layer = layer_factory.Pooling(name='pool1', pool_type='max', kernel=(3, 3))
        sym = layer.create_layer(self.data)

        assert sym.attr_dict() == {'pool1': {'stride': '()', 'pad': '()', 'global_pool': 'False', 'kernel': '(3, 3)',
                                   'pool_type': 'max', 'pooling_convention': 'valid'}}
        assert sym.get_internals().list_outputs() == ['data', 'pool1_output']


class TestFlatten(TestLayerFactory):
    def test_from_dict(self):
        self.layer_dict = {"op": "Flatten", "name": "flatten1", "inputs": [[1, 0, 0]], 'elements_offset': 0,
                           'id2name': {1: 'name1'}, 'prev_symbols': {'name1': 'sym'}}
        self.attributes = {'name': 'flatten1'}
        self.LayerFactoryClass = layer_factory.Flatten

        self._assert_from_dict('Flatten', False, 'sym')

    def test_create_layer(self):
        layer = layer_factory.Flatten(name='flat1')
        sym = layer.create_layer(self.data)

        assert sym.attr_dict() == {}
        assert sym.get_internals().list_outputs() == ['data', 'flat1_output']


class TestDropout(TestLayerFactory):
    def test_from_dict(self):
        self.layer_dict = {"op": "Dropout", "attrs": {"p": "0.55"}, "name": "drop6", "inputs": [[1, 0]],
                           "backward_source_id": -1, 'elements_offset': 0, 'id2name': {1: 'name1'},
                           'prev_symbols': {'name1': 'sym'}}
        self.attributes = {'name': 'drop6', 'p': 0.55}
        self.LayerFactoryClass = layer_factory.Dropout

        self._assert_from_dict('Dropout', False, 'sym')

    def test_create_layer(self):
        layer = layer_factory.Dropout(name='drop1', p=0.4)
        sym = layer.create_layer(self.data)

        assert sym.attr_dict() == {'drop1': {'p': '0.4'}}
        assert sym.get_internals().list_outputs() == ['data', 'drop1_output']


class TestConcat(TestLayerFactory):
    def test_from_dict(self):
        self.layer_dict = {"op": "Concat", "attrs": {"dim": "1", "num_args": "2"}, "name": "concat1",
                           "inputs": [[13, 0], [17, 0]], "backward_source_id": -1,
                           'id2name': {13: 'layer13', 17: 'layer17'},
                           'prev_symbols': {'layer13': self.data, 'layer17': self.data}}
        self.attributes = {'name': 'concat1', 'dim': 1}
        self.LayerFactoryClass = layer_factory.Concat

        self._assert_from_dict('Concat', False, [self.data])

    def test_create_layer(self):
        layer = layer_factory.Concat(name='concat1', dim=2)
        sym = layer.create_layer([self.data, self.data])

        assert sym.attr_dict() == {'concat1': {'dim': '2', 'num_args': '2'}}
        assert sym.get_internals().list_outputs() == ['data', 'concat1_output']

    def test_create_layer_three_arg(self):
        layer = layer_factory.Concat(name='concat1', dim=2)
        sym = layer.create_layer([self.data, self.data, self.data])

        assert sym.attr_dict() == {'concat1': {'dim': '2', 'num_args': '3'}}
        assert sym.get_internals().list_outputs() == ['data', 'concat1_output']

    def test_create_layer_one_arg(self):
        layer = layer_factory.Concat(name='concat1', dim=2)
        sym = layer.create_layer(self.data)

        assert sym.attr_dict() == {'concat1': {'dim': '2', 'num_args': '1'}}
        assert sym.get_internals().list_outputs() == ['data', 'concat1_output']


class TestBatchNorm(TestLayerFactory):
    def test_from_dict(self):
        self.layer_dict = {"op": "BatchNorm", "attrs": {"eps": "2e-05", "fix_gamma": "True", "momentum": "0.9",
                           "use_global_stats": "False"}, "name": "bn_data", "backward_source_id": -1,
                           "inputs": [[0, 0], [1, 0], [2, 0]], 'elements_offset': 0, 'id2name': {0: 'layer0'},
                           'prev_symbols': {'layer0': 'sym'}}
        self.attributes = {'name': 'bn_data', 'eps': 2e-05, 'fix_gamma': True, 'momentum': 0.9,
                           'use_global_stats': False}
        self.LayerFactoryClass = layer_factory.BatchNorm

        self._assert_from_dict('BatchNorm', False, 'sym')

    def test_create_layer(self):
        layer = layer_factory.BatchNorm(name='flat1')
        sym = layer.create_layer(self.data)

        assert sym.attr_dict() == {
                                    'flat1_beta': {'fix_gamma': 'True', 'momentum': '0.9', 'use_global_stats': 'False',
                                                   'eps': '0.001'},
                                    'flat1_moving_var': {'__init__': '["one", {}]', 'momentum': '0.9', 'eps': '0.001',
                                                         'use_global_stats': 'False', 'fix_gamma': 'True'},
                                    'flat1': {'fix_gamma': 'True', 'momentum': '0.9', 'use_global_stats': 'False',
                                              'eps': '0.001'},
                                    'flat1_gamma': {'fix_gamma': 'True', 'momentum': '0.9', 'use_global_stats': 'False',
                                                    'eps': '0.001'},
                                    'flat1_moving_mean': {'__init__': '["zero", {}]', 'momentum': '0.9', 'eps': '0.001',
                                                          'use_global_stats': 'False', 'fix_gamma': 'True'}
                                    }
        assert sym.get_internals().list_outputs() == ['data', 'flat1_gamma', 'flat1_beta', 'flat1_moving_mean',
                                                      'flat1_moving_var', 'flat1_output']


class TestAdd(TestLayerFactory):
    def test_from_dict(self):
        self.layer_dict = {"op": "_Plus", "attrs": {}, "name": "_plus0", "inputs": [[28, 0], [30, 0]],
                           "backward_source_id": -1, "elements_offset": 0, 'id2name': {28: 'layer28', 30: 'layer30'},
                           'prev_symbols': {'layer28': 'sym1', 'layer30': 'sym2'}}
        self.attributes = {'name': '_plus0'}
        self.LayerFactoryClass = layer_factory.Add

        self._assert_from_dict('_Plus', False, 'sym1')

    def test_create_layer(self):
        layer = layer_factory.Add(name='pool1')
        sym = layer.create_layer([self.data, self.data])

        assert sym.attr_dict() == {}
        assert sym.get_internals().list_outputs() == ['data', '_plus0_output']

    def test_create_layer_len_1(self):
        layer = layer_factory.Add(name='pool1')
        with self.assertRaises(ValueError):
            layer.create_layer([self.data])


class TestSVMOutput(TestLayerFactory):
    def test_from_dict(self):
        self.layer_dict = {'op': "SVMOutput", 'name': 'svm0', 'attrs': {'margin': '1',
                           'regularization_coefficient': '1', 'use_linear': 0}, 'inputs': [[1, 0, 0], [2, 0, 0]],
                           'elements_offset': 0, 'id2name': {1: 'name1'}, 'prev_symbols': {'name1': 'sym'}}
        self.attributes = {'name': 'svm0', 'label': None, 'margin': 1.0, 'regularization_coefficient': 1.0,
                           'use_linear': False}
        self.LayerFactoryClass = layer_factory.SVMOutput

        self._assert_from_dict('SVMOutput', True, 'sym')

    def test_create_layer(self):
        layer = layer_factory.SVMOutput(name='svm1', margin=2)
        sym = layer.create_layer(self.data)

        assert sym.attr_dict() == {'svm1': {'margin': '2', 'regularization_coefficient': '1', 'use_linear': '0'},
                                   'svm1_label': {'margin': '2', 'regularization_coefficient': '1', 'use_linear': '0'}}
        assert sym.get_internals().list_outputs() == ['data', 'svm1_label', 'svm1_output']


class TestEmbedding(TestLayerFactory):
    def test_from_dict(self):
        self.layer_dict = {"op": "Embedding", "name": "embed", "attrs": {"input_dim": "27774", "output_dim": "50"},
                           "inputs": [[0, 0, 0], [1, 0, 0]], 'elements_offset': 0, 'id2name': {0: 'name1'},
                           'prev_symbols': {'name1': 'sym'}}
        self.attributes = {'name': 'embed', 'input_dim': 27774, 'output_dim': 50, 'weight': None, 'dtype': 'float32'}
        self.LayerFactoryClass = layer_factory.Embedding

        self._assert_from_dict('Embedding', False, 'sym')

    def test_create_layer(self):
        layer = layer_factory.Embedding(name='embed1', input_dim=2000, output_dim=50)
        sym = layer.create_layer(self.data)

        assert sym.attr_dict() == {'embed1_weight': {'dtype': 'float32', 'output_dim': '50', 'input_dim': '2000'},
                                   'embed1': {'dtype': 'float32', 'output_dim': '50', 'input_dim': '2000'}}
        assert sym.get_internals().list_outputs() == ['data', 'embed1_weight', 'embed1_output']


class TestMean(TestLayerFactory):
    def test_from_dict(self):
        self.layer_dict = {"op": "mean", "name": "mean_embed", "attrs": {"axis": "1"}, "inputs": [[2, 0, 0]],
                           'elements_offset': 0, 'id2name': {2: 'name1'}, 'prev_symbols': {'name1': 'sym'}}
        self.attributes = {'name': 'mean_embed', 'axis': 1, 'keepdims': False, 'exclude': False}
        self.LayerFactoryClass = layer_factory.Mean

        self._assert_from_dict('mean', False, 'sym')

    def test_from_dict_axis_tuple(self):
        self.layer_dict = {"op": "mean", "name": "mean_embed", "attrs": {"axis": "(3,)"}, "inputs": [[2, 0, 0]],
                           'elements_offset': 0, 'id2name': {2: 'name1'}, 'prev_symbols': {'name1': 'sym'}}
        self.attributes = {'name': 'mean_embed', 'axis': (3,), 'keepdims': False, 'exclude': False}
        self.LayerFactoryClass = layer_factory.Mean

        self._assert_from_dict('mean', False, 'sym')

    def test_add_layer(self):
        layer = layer_factory.Mean(name='mean1', axis=1)
        sym = layer.create_layer(self.data)

        assert sym.attr_dict() == {'mean1': {'axis': '1', 'keepdims': '0', 'exclude': '0'}}
        assert sym.get_internals().list_outputs() == ['data', 'mean1_output']
