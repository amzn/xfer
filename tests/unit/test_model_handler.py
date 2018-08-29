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
from unittest import mock, TestCase

import json
import os
import mxnet as mx
import numpy as np
from collections import OrderedDict

from xfer import model_handler
from xfer.model_handler import consts, layer_factory, exceptions


class TestModelHandler(TestCase):
    def setUp(self):
        np.random.seed(1)
        self.data_name = 'data'
        mod = mx.module.Module.load('tests/data/testnetv1', 0, label_names=['softmaxoutput1_label'],
                                    data_names=(self.data_name,))
        self.mh = model_handler.ModelHandler(mod, mx.context.cpu, 1, self.data_name)
        self.imglist = [[0, 'accordion/image_0001.jpg'], [0, 'accordion/image_0002.jpg'], [1, 'ant/image_0001.jpg'],
                        [1, 'ant/image_0002.jpg'], [2, 'anchor/image_0001.jpg'], [2, 'anchor/image_0002.jpg']]
        self.image_iter = mx.image.ImageIter(2, (3, 224, 224), imglist=self.imglist, path_root='tests/data/test_images',
                                             label_name='softmaxoutput1_label', data_name=self.data_name)

        self.act1_id = 4
        self.conv2_id = 7

    def tearDown(self):
        del self.mh

    def test_constructor_binded_module(self):
        # Assert module that is binded can be used to init a ModelHandler object and can add/drop layers
        mod = mx.module.Module.load('tests/data/testnetv1', 0, label_names=['softmaxoutput1_label'])
        mod.bind(data_shapes=[('data', self.image_iter.provide_data[0][1])], label_shapes=self.image_iter.provide_label)
        assert mod.binded and mod.params_initialized
        mh = model_handler.ModelHandler(mod, mx.context.cpu, 1)
        mh.drop_layer_top()
        mh.drop_layer_bottom()

    def test_constructor_no_weight(self):
        # Assert module that is unbinded can be used to init a ModelHandler object and can add/drop layers
        sym, _, _ = mx.model.load_checkpoint('tests/data/testnetv1', 0)
        mod = mx.module.Module(sym, label_names=['softmaxoutput1_label'])
        mh = model_handler.ModelHandler(mod, mx.context.cpu, 1)
        assert np.array_equal(mh.arg_params, {})
        assert np.array_equal(mh.aux_params, {})
        mh.drop_layer_top()
        mh.drop_layer_bottom()

    def test_drop_layer_top_1(self):
        assert 'softmaxoutput1' in list(self.mh.layer_type_dict.keys())

        outputs_pre = self.mh.symbol.get_internals().list_outputs()
        self.mh.drop_layer_top()
        outputs_post = self.mh.symbol.get_internals().list_outputs()

        assert 'softmaxoutput1' not in list(self.mh.layer_type_dict.keys())
        assert set(outputs_pre).symmetric_difference(set(outputs_post)) == {'softmaxoutput1_output',
                                                                            'softmaxoutput1_label'}

    def test_drop_layer_top_3(self):
        for layer_name in ['softmaxoutput1', 'fullyconnected0', 'flatten1']:
            assert layer_name in list(self.mh.layer_type_dict.keys())

        outputs_pre = self.mh.symbol.get_internals().list_outputs()
        self.mh.drop_layer_top(3)
        outputs_post = self.mh.symbol.get_internals().list_outputs()

        for layer_name in ['softmaxoutput1', 'fullyconnected0', 'flatten1']:
            assert layer_name not in list(self.mh.layer_type_dict.keys())
        assert set(outputs_pre).symmetric_difference(set(outputs_post)) == {'flatten1_output', 'fullyconnected0_weight',
                                                                            'fullyconnected0_bias',
                                                                            'fullyconnected0_output',
                                                                            'softmaxoutput1_label',
                                                                            'softmaxoutput1_output'}

    def test_drop_layer_top_too_many(self):
        with self.assertRaises(model_handler.exceptions.ModelError):
            self.mh.drop_layer_top(8)

    def test_drop_layer_bottom_1(self):
        assert 'conv1' in list(self.mh.layer_type_dict.keys())

        outputs_pre = self.mh.symbol.get_internals().list_outputs()
        self.mh.drop_layer_bottom()
        outputs_post = self.mh.symbol.get_internals().list_outputs()

        assert 'conv1' not in list(self.mh.layer_type_dict.keys())
        assert set(outputs_pre).symmetric_difference(set(outputs_post)) == {'conv1_bias', 'conv1_weight',
                                                                            'conv1_output'}

    def test_drop_layer_bottom_3(self):
        for layer_name in ['conv1', 'act1', 'conv2']:
            assert layer_name in list(self.mh.layer_type_dict.keys())

        outputs_pre = self.mh.symbol.get_internals().list_outputs()
        self.mh.drop_layer_bottom(3)
        outputs_post = self.mh.symbol.get_internals().list_outputs()

        for layer_name in ['conv1', 'act1', 'conv2']:
            assert layer_name not in list(self.mh.layer_type_dict.keys())
        assert set(outputs_pre).symmetric_difference(set(outputs_post)) == {'conv1_bias', 'conv1_weight',
                                                                            'conv1_output', 'act1_output',
                                                                            'conv2_bias',  'conv2_output',
                                                                            'conv2_weight'}

    def test_drop_layer_bottom_too_many(self):
        with self.assertRaises(model_handler.exceptions.ModelError):
            self.mh.drop_layer_bottom(8)

    def test_drop_layer_bottom_no_first_layer(self):
        with open('tests/data/symbol_dict.json') as json_data:
            sym_dict = json.load(json_data)
        sym_dict[consts.NODES] = [node for node in sym_dict[consts.NODES] if node[consts.OPERATION] == consts.NO_OP]

        with mock.patch('xfer.model_handler.ModelHandler._get_symbol_dict', return_value=sym_dict):
            with self.assertRaises(model_handler.exceptions.ModelError):
                self.mh.drop_layer_bottom()

    def test_add_layer_top_model_error(self):
        # Assert model error raised when a layer is added above an output layer
        layer1 = layer_factory.FullyConnected(name='fc1', num_hidden=5)
        with self.assertRaises(model_handler.exceptions.ModelError):
            self.mh.add_layer_top([layer1])

    def test_add_layer_top(self):
        # Drop output layer so that layers can be added to top
        self.mh.drop_layer_top()
        layer1 = layer_factory.FullyConnected(name='fc1', num_hidden=5)
        assert 'fc1' not in list(self.mh.layer_type_dict.keys())

        outputs_pre = self.mh.symbol.get_internals().list_outputs()
        self.mh.add_layer_top([layer1])
        outputs_post = self.mh.symbol.get_internals().list_outputs()

        assert 'fc1' in list(self.mh.layer_type_dict.keys())
        assert outputs_post == outputs_pre + ['fc1_weight', 'fc1_bias', 'fc1_output']

    def test_add_layer_top_2(self):
        self.mh.drop_layer_top()
        layer1 = layer_factory.FullyConnected(name='fc1', num_hidden=5)
        layer2 = layer_factory.Convolution(name='conv1_1', kernel=(3, 3), num_filter=10)
        for layer_name in ['fc1', 'conv1_1']:
            assert layer_name not in list(self.mh.layer_type_dict.keys())

        outputs_pre = self.mh.symbol.get_internals().list_outputs()
        self.mh.add_layer_top([layer1])
        self.mh.add_layer_top([layer2])
        outputs_post = self.mh.symbol.get_internals().list_outputs()

        for layer_name in ['fc1', 'conv1_1']:
            assert layer_name in list(self.mh.layer_type_dict.keys())
        assert outputs_post == outputs_pre + ['fc1_weight', 'fc1_bias', 'fc1_output', 'conv1_1_weight', 'conv1_1_bias',
                                              'conv1_1_output']

    def test_add_layer_top_list(self):
        self.mh.drop_layer_top()
        layer1 = layer_factory.FullyConnected(name='fc1', num_hidden=5)
        layer2 = layer_factory.Convolution(name='conv1_1', kernel=(3, 3), num_filter=10)
        for layer_name in ['fc1', 'conv1_1']:
            assert layer_name not in list(self.mh.layer_type_dict.keys())

        outputs_pre = self.mh.symbol.get_internals().list_outputs()
        self.mh.add_layer_top([layer1, layer2])
        outputs_post = self.mh.symbol.get_internals().list_outputs()

        for layer_name in ['fc1', 'conv1_1']:
            assert layer_name in list(self.mh.layer_type_dict.keys())
        assert outputs_post == outputs_pre + ['fc1_weight', 'fc1_bias', 'fc1_output', 'conv1_1_weight', 'conv1_1_bias',
                                              'conv1_1_output']

    def test_add_layer_bottom_output_layer(self):
        # Assert that adding an output layer to the bottom of the model raises a model error
        layer1 = layer_factory.SoftmaxOutput(name='softmax')
        with self.assertRaises(model_handler.exceptions.ModelError):
            self.mh.add_layer_bottom([layer1])

    def test_add_layer_bottom(self):
        layer1 = layer_factory.Convolution(name='conv1_1', kernel=(3, 3), num_filter=10)
        assert 'conv1_1' not in list(self.mh.layer_type_dict.keys())

        outputs_pre = self.mh.symbol.get_internals().list_outputs()
        self.mh.add_layer_bottom([layer1])
        outputs_post = self.mh.symbol.get_internals().list_outputs()

        assert 'conv1_1' in list(self.mh.layer_type_dict.keys())
        assert outputs_post == [self.data_name, 'conv1_1_weight', 'conv1_1_bias', 'conv1_1_output'] + outputs_pre[1:]

    def test_add_layer_bottom_2(self):
        layer1 = layer_factory.Convolution(name='conv1_1', kernel=(3, 3), num_filter=10)
        layer2 = layer_factory.FullyConnected(name='fc1', num_hidden=10)
        for layer_name in ['fc1', 'conv1_1']:
            assert layer_name not in list(self.mh.layer_type_dict.keys())

        outputs_pre = self.mh.symbol.get_internals().list_outputs()
        self.mh.add_layer_bottom([layer1])
        self.mh.add_layer_bottom([layer2])
        outputs_post = self.mh.symbol.get_internals().list_outputs()

        for layer_name in ['fc1', 'conv1_1']:
            assert layer_name in list(self.mh.layer_type_dict.keys())
        assert outputs_post == [self.data_name, 'fc1_weight', 'fc1_bias', 'fc1_output', 'conv1_1_weight',
                                'conv1_1_bias', 'conv1_1_output'] + outputs_pre[1:]

    def test_add_layer_bottom_list(self):
        layer1 = layer_factory.Convolution(name='conv1_1', kernel=(3, 3), num_filter=10)
        layer2 = layer_factory.FullyConnected(name='fc1', num_hidden=10)
        for layer_name in ['fc1', 'conv1_1']:
            assert layer_name not in list(self.mh.layer_type_dict.keys())

        outputs_pre = self.mh.symbol.get_internals().list_outputs()
        self.mh.add_layer_bottom([layer1, layer2])
        outputs_post = self.mh.symbol.get_internals().list_outputs()

        for layer_name in ['fc1', 'conv1_1']:
            assert layer_name in list(self.mh.layer_type_dict.keys())
        assert outputs_post == [self.data_name, 'conv1_1_weight', 'conv1_1_bias', 'conv1_1_output', 'fc1_weight',
                                'fc1_bias', 'fc1_output'] + outputs_pre[1:]

    @staticmethod
    def create_csv_iterator(batch_size=1):
        # Save data as csv
        np.random.seed(1)
        csv_data = np.random.rand(6, 3 * 224 * 224)
        data_path = './csv_iterator_data.csv'
        np.savetxt(data_path, csv_data, delimiter=',')

        # Save labels as csv
        labels = np.array([[0], [0], [1], [1], [2], [2]])
        label_path = './csv_iterator_labels.csv'
        np.savetxt(label_path, labels, delimiter=',')

        # Create CSV iterator using data file and labels file
        # round_batch should be False for resets to work
        # (https://mxnet.incubator.apache.org/api/python/io/io.html#mxnet.io.CSVIter)
        csv_iterator = mx.io.CSVIter(data_csv=data_path, data_shape=(3, 224, 224), label_csv=label_path,
                                     batch_size=batch_size, label_name='softmaxoutput1_label', round_batch=False)

        # Remove data and label files
        os.remove(data_path)
        os.remove(label_path)

        return csv_iterator

    def test_get_module_csv_iterator(self):
        self._test_get_module(iterator=self.create_csv_iterator())

    def test_get_module_image_iterator(self):
        self._test_get_module(iterator=self.image_iter)

    def _test_get_module(self, iterator):
        fixed_layer_parameters = ['conv1_weight', 'conv1_bias', 'conv1_foo']
        random_layer_parameters = ['conv2_weight', 'conv2_bias', 'conv2_foo']
        # Set biases to random arrays so they are not empty arrays
        self.mh.arg_params['conv1_bias'] = mx.nd.random.normal(shape=(1))
        self.mh.arg_params['conv2_bias'] = mx.nd.random.normal(shape=(8))
        self.mh.arg_params['fullyconnected0_bias'] = mx.nd.random.normal(shape=(4))
        with self.assertLogs() as cm:
            mod = self.mh.get_module(iterator, fixed_layer_parameters=fixed_layer_parameters,
                                     random_layer_parameters=random_layer_parameters)
        assert type(mod) == mx.mod.Module
        assert mod._fixed_param_names == ['conv1_weight', 'conv1_bias']
        # Assert finetune layer weights are equal to pretrained weights
        for f_tune_key in ['fullyconnected0_weight', 'conv1_weight', 'fullyconnected0_bias', 'conv1_bias']:
            assert np.array_equal(self.mh.arg_params[f_tune_key].asnumpy(), mod.get_params()[0][f_tune_key].asnumpy())
        # Assert random layer weights are different to pretrained weights
        for rand_key in ['conv2_weight', 'conv2_bias']:
            assert not np.array_equal(self.mh.arg_params[rand_key].asnumpy(), mod.get_params()[0][rand_key].asnumpy())
        # Assert logs written for fixed and random parameters that were not present in model
        assert cm.output == ['WARNING:root:Could not find layer parameters: conv1_foo',
                             'WARNING:root:Could not find layer parameters: conv2_foo']

    def test_get_module_set_params_error(self):
        image_iter = mx.image.ImageIter(2, (3, 123, 123), imglist=self.imglist, path_root='test_images',
                                        label_name='softmaxoutput1_label')
        self.assertRaisesRegex(model_handler.exceptions.ModelArchitectureError, 'Weight shape mismatch: Expected shape='
                               '\(4,46225\), Actual shape=\(4,12996\). This can be caused by incorrect layer shapes or '
                               'incorrect input data shapes.',
                               self.mh.get_module, image_iter)

    def test_get_layer_type(self):
        layer_type = []
        for name in ['conv1', 'act1', 'conv2', 'act2', 'pool1', 'flatten1',
                     'fullyconnected0', 'softmaxoutput1']:
            layer_type.append(self.mh.get_layer_type(name))

        assert layer_type == ['Convolution', 'Activation', 'Convolution',
                              'Activation', 'Pooling', 'Flatten',
                              'FullyConnected', 'SoftmaxOutput']

    def test_get_layer_type_failure(self):
        with self.assertRaises(ValueError):
            self.mh.get_layer_type('fake_layer_name')

    def test_get_layer_names_matching_type(self):
        layers_found = self.mh.get_layer_names_matching_type(consts.LayerType.CONVOLUTION)
        assert sorted(layers_found) == sorted(['conv1', 'conv2'])
        layers_found = self.mh.get_layer_names_matching_type(consts.LayerType.ACTIVATION)
        assert sorted(layers_found) == sorted(['act1', 'act2'])
        layers_found = self.mh.get_layer_names_matching_type(consts.LayerType.POOLING)
        assert layers_found == ['pool1']
        layers_found = self.mh.get_layer_names_matching_type(consts.LayerType.FLATTEN)
        assert layers_found == ['flatten1']
        layers_found = self.mh.get_layer_names_matching_type(consts.LayerType.FULLYCONNECTED)
        assert layers_found == ['fullyconnected0']
        layers_found = self.mh.get_layer_names_matching_type(consts.LayerType.SOFTMAXOUTPUT)
        assert layers_found == ['softmaxoutput1']
        layers_found = self.mh.get_layer_names_matching_type(consts.LayerType.BATCHNORM)
        assert layers_found == []

    def _test_get_layer_output_image_iterator(self, batch_size):
        layer1, layer2 = ('fullyconnected0', 'flatten1')
        image_iter = mx.image.ImageIter(batch_size, (3, 224, 224), imglist=self.imglist,
                                        path_root='tests/data/test_images', label_name='softmaxoutput1_label',
                                        data_name=self.data_name)
        features, labels = self.mh.get_layer_output(image_iter, [layer1, layer2])
        assert type(features) == OrderedDict
        assert list(features.keys()) == ['fullyconnected0', 'flatten1']
        tolerance = 5e-2

        expected_labels = [0, 0, 1, 1, 2, 2]
        assert (labels == expected_labels).all(), 'Expected {}, got {}'.format(expected_labels, labels)
        assert sorted(features.keys()) == sorted([layer1, layer2])

        expected_features = {}
        expected_features[layer1] = np.array([[-2026513.375, -1203439., -1706091., 402870.3125],
                                              [-5010966., -611501.375, -1003263.4375, 2879037.],
                                              [-145177.96875, -4538920.5, 219717.46875, 1895801.25],
                                              [-2568541.5, -341903.75, -2446673., 716777.3125],
                                              [491937.84375, -336953.78125, -717204.75, 941894.],
                                              [-3536730., -3252726.75, 196532.796875, 1375444.125]])
        assert np.allclose(features[layer1], expected_features[layer1], rtol=tolerance)
        assert features[layer2].shape == (6, 46225)

    def _test_get_layer_output_csv_iterator(self, batch_size):
        self._test_get_layer_output(iterator=self.create_csv_iterator(batch_size=batch_size))

    def _test_get_layer_output_NDArray_iterator(self, batch_size):
        data = np.random.rand(6, 3, 224, 224)
        nda_iter = mx.io.NDArrayIter(data=data, label=np.array([0, 0, 1, 1, 2, 2]), batch_size=batch_size)
        self._test_get_layer_output(iterator=nda_iter)

    def _test_get_layer_output(self, iterator):
        layer1, layer2 = ('fullyconnected0', 'flatten1')
        features, labels = self.mh.get_layer_output(iterator, [layer1, layer2])
        assert type(features) == OrderedDict
        assert list(features.keys()) == ['fullyconnected0', 'flatten1']
        tolerance = 1e-4

        assert (labels == [0, 0, 1, 1, 2, 2]).all()
        assert sorted(features.keys()) == sorted([layer1, layer2])

        expected_features = {}
        expected_features[layer1] = np.array([[-14632.44140625, -15410.29882812, 401.85760498, 3048.72143555],
                                              [-14077.45507812, -14598.43457031, 1666.03442383, 7222.70019531],
                                              [-16092.02929688, -16345.94042969, -2379.76391602, 6223.95605469],
                                              [-15005.03808594, -15088.01660156, 5304.77539062, 6113.68164062],
                                              [-12687.98339844, -14598.75097656, 1158.26794434, 4819.8671875],
                                              [-11366.10351562, -16437.70507812, -4690.74951172, 8055.31982422]])
        assert np.allclose(features[layer1], expected_features[layer1], rtol=tolerance)
        assert features[layer2].shape == (6, 46225)

    def test_get_layer_output_image_iterator(self):
        # test for when there is no padding
        self._test_get_layer_output_image_iterator(batch_size=2)

    def test_get_layer_output_image_iterator_padding(self):
        # test for padding
        self._test_get_layer_output_image_iterator(batch_size=4)

    def test_get_layer_output_NDArray_iterator(self):
        # test for when there is no padding
        self._test_get_layer_output_NDArray_iterator(batch_size=2)

    def test_get_layer_output_NDArray_iterator_padding(self):
        # test for padding
        self._test_get_layer_output_NDArray_iterator(batch_size=4)

    def test_get_layer_output_csv_iterator(self):
        # test for when there is no padding
        self._test_get_layer_output_csv_iterator(batch_size=2)

    def test_get_layer_output_csv_iterator_padding(self):
        # test for padding
        self._test_get_layer_output_csv_iterator(batch_size=4)

    def test_get_layer_output_invalid_type(self):
        layer1, layer2 = ('fullyconnected0', 'flatten1')
        with self.assertRaises(TypeError):
            self.mh.get_layer_output(None, [layer1, layer2])
        with self.assertRaises(TypeError):
            self.mh.get_layer_output('iter', [layer1, layer2])
        with self.assertRaises(TypeError):
            self.mh.get_layer_output(self.image_iter, layer1)

    def test_get_layer_parameters(self):
        layer_params = self.mh.get_layer_parameters(self.mh.layer_names)
        assert sorted(layer_params) == sorted(['conv1_bias', 'conv1_weight', 'conv2_bias', 'conv2_weight',
                                               'fullyconnected0_bias', 'fullyconnected0_weight'])
        layer_params = self.mh.get_layer_parameters(['fullyconnected0'])
        assert sorted(layer_params) == sorted(['fullyconnected0_bias', 'fullyconnected0_weight'])

    def test_get_layer_parameters_not_list(self):
        with self.assertRaises(TypeError):
            self.mh.get_layer_parameters('fullyconnected0')

    def test_visualize_net(self):
        with open('tests/data/visualize.json', 'r') as json_data:
            expected_symbol_dict = json.load(json_data)
        assert expected_symbol_dict == self.mh.visualize_net().__dict__

    def test_save_symbol(self):
        assert not os.path.isfile('temp-test-symbol.json')
        self.mh.save_symbol('temp-test')
        assert os.path.isfile('temp-test-symbol.json')

        saved_sym = mx.sym.load('temp-test-symbol.json')
        assert self.mh.symbol.get_internals().list_outputs() == saved_sym.get_internals().list_outputs()
        os.remove('temp-test-symbol.json')

    def test_validate_layer_name(self):
        with self.assertRaises(ValueError):
            self.mh._validate_layer_name('conv2')
        with self.assertRaises(ValueError):
            self.mh._validate_layer_name('prob_label')
        with self.assertRaises(ValueError):
            self.mh._validate_layer_name('conv3_output')
        with self.assertRaises(ValueError):
            self.mh._validate_layer_name('conv3_weight')
        with self.assertRaises(ValueError):
            self.mh._validate_layer_name('conv3_moving_mean')
        with self.assertRaises(ValueError):
            self.mh._validate_layer_name('conv3_beta_q')
        with self.assertRaises(ValueError):
            self.mh._validate_layer_name(self.data_name)
        self.mh._validate_layer_name('conv3')
        self.mh._validate_layer_name('weighted')

    def test_model_from_nodes(self):
        nodes = json.loads(self.mh.symbol.tojson())[consts.NODES]
        id2name = {0: self.data_name, 1: 'conv1_weight', 2: 'conv1_bias', 3: 'conv1', 4: 'act1', 5: 'conv2_weight',
                   6: 'conv2_bias', 7: 'conv2', 8: 'act2', 9: 'pool1', 10: 'flatten1', 11: 'fullyconnected0_weight',
                   12: 'fullyconnected0_bias', 13: 'fullyconnected0', 14: 'softmaxoutput1_label', 15: 'softmaxoutput1'}

        sym = self.mh._model_from_nodes(symbol=None, symbol_nodes=nodes, elements_offset=0, prev_symbols={},
                                        id2name=id2name)

        assert sym == self.mh.symbol
        assert sym.get_internals().list_outputs() == self.mh.symbol.get_internals().list_outputs()

    def test_prune_parameters(self):
        # Assert that parameters are pruned and logs are written
        parameter_names_pre = ['conv1_weight', 'conv1_bias', 'conv1_moving_mean', 'made_up_param']
        with self.assertLogs() as cm:
            parameter_names_post = self.mh._prune_parameters(parameter_names_pre)
        assert set(parameter_names_pre).symmetric_difference(set(parameter_names_post)) == {'conv1_moving_mean',
                                                                                            'made_up_param'}
        assert cm.output == ['WARNING:root:Could not find layer parameters: conv1_moving_mean, made_up_param']

    def test_prune_parameters_no_extra(self):
        # Assert no parameters are removed and no logs are written
        parameter_names_pre = ['conv1_weight', 'conv1_bias']
        with self.assertRaises(AssertionError):
            with self.assertLogs():
                parameter_names_post = self.mh._prune_parameters(parameter_names_pre)
        assert set(parameter_names_pre).symmetric_difference(set(parameter_names_post)) == set([])

    def test_remove_random_parameters(self):
        random_parameters = ['conv1_weight', 'conv1_bias', 'conv1_moving_var', 'conv3_weight', 'conv3_moving_var']
        self.mh.arg_params = {'conv1_weight': 1, 'conv1_bias': 1, 'conv2_weight': 1, 'conv2_bias': 1}
        self.mh.aux_params = {'conv1_moving_var': 1, 'conv2_moving_var': 1}
        with self.assertLogs() as cm:
            arg_params, aux_params = self.mh._remove_random_parameters(random_parameters)

        assert arg_params == {'conv2_weight': 1, 'conv2_bias': 1}
        assert aux_params == {'conv2_moving_var': 1}
        cm.output == 'WARNING:root:Could not find layers parameters: conv3_weight, conv3_moving_var'

    def test_get_devices_cpu(self):
        devices = self.mh._get_devices(mx.context.cpu, 3)
        assert devices == [mx.cpu(0), mx.cpu(1), mx.cpu(2)]

    def test_get_devices_gpu(self):
        devices = self.mh._get_devices(mx.context.gpu, 3)
        assert devices == [mx.gpu(0), mx.gpu(1), mx.gpu(2)]

    def test_assert_drop_layer_valid(self):
        self.mh.layer_type_dict = {}
        with self.assertRaises(exceptions.ModelError):
            self.mh._assert_drop_layer_valid(1)
        self.mh.layer_type_dict = {'a': 1}
        with self.assertRaises(exceptions.ModelError):
            self.mh._assert_drop_layer_valid(1)
        self.mh.layer_type_dict = {'a': 1, 'b': 2}
        self.mh._assert_drop_layer_valid(1)
        with self.assertRaises(exceptions.ModelError):
            self.mh._assert_drop_layer_valid(2)
        self.mh.layer_type_dict = {'a': 1, 'b': 2, 'c': 3}
        self.mh._assert_drop_layer_valid(1)
        self.mh._assert_drop_layer_valid(2)
        with self.assertRaises(exceptions.ModelError):
            self.mh._assert_drop_layer_valid(3)

    def test_get_layer_type_dict(self):
        expected_dict = OrderedDict([('conv1', 'Convolution'), ('act1', 'Activation'), ('conv2', 'Convolution'),
                                     ('act2', 'Activation'), ('pool1', 'Pooling'), ('flatten1', 'Flatten'),
                                     ('fullyconnected0', 'FullyConnected'), ('softmaxoutput1', 'SoftmaxOutput')])
        assert list(self.mh._get_layer_type_dict().items()) == list(expected_dict.items())

    def test_get_symbol_dict(self):
        with open('tests/data/symbol_dict.json') as json_data:
            expected_symbol_dict = json.load(json_data)

        real_symbol_dict = self.mh._get_symbol_dict()

        # Remove mxnet version number from symbol dictionaries so test doesn't break on new version
        MXNET_VERSION = 'mxnet_version'
        if MXNET_VERSION in expected_symbol_dict[consts.ATTRIBUTES].keys():
            del expected_symbol_dict[consts.ATTRIBUTES]['mxnet_version']
        if MXNET_VERSION in real_symbol_dict[consts.ATTRIBUTES].keys():
            del real_symbol_dict[consts.ATTRIBUTES]['mxnet_version']

        assert expected_symbol_dict == real_symbol_dict

    def test_layer_names(self):
        expected_layer_names = ['conv1', 'act1', 'conv2', 'act2', 'pool1', 'flatten1', 'fullyconnected0',
                                'softmaxoutput1']

        assert self.mh.layer_names == expected_layer_names

    def test_clean_params(self):
        symbol = self.mh.symbol.get_internals()[self.act1_id]
        assert sorted(self.mh.arg_params.keys()) == sorted(['conv1_weight', 'conv2_bias', 'conv1_bias', 'conv2_weight',
                                                            'fullyconnected0_weight', 'fullyconnected0_bias'])

        arg_params = self.mh._clean_params(symbol, self.mh.arg_params)

        assert sorted(arg_params.keys()) == sorted(['conv1_weight', 'conv1_bias'])

        symbol = self.mh.symbol.get_internals()[self.conv2_id]
        assert sorted(self.mh.arg_params.keys()) == sorted(['conv1_weight', 'conv2_bias', 'conv1_bias', 'conv2_weight',
                                                            'fullyconnected0_weight', 'fullyconnected0_bias'])

        arg_params = self.mh._clean_params(symbol, self.mh.arg_params)

        assert sorted(arg_params.keys()) == sorted(['conv1_weight', 'conv2_bias', 'conv1_bias', 'conv2_weight'])

    def test_update_sym(self):
        mod = mx.module.Module.load('tests/data/testnetv1', 0, label_names=['softmaxoutput1_label'])
        mh = model_handler.ModelHandler(mod, mx.context.cpu, 1)
        symbol = mh.symbol.get_internals()[self.act1_id]
        self.mh.aux_params = {1: 2, 2: 3, 4: 5, 7: 3, 6: 8, 3: 5, 12: 5}

        mh.update_sym(symbol)

        assert mh.layer_type_dict == OrderedDict([('conv1', 'Convolution'), ('act1', 'Activation')])
        assert mh.symbol == symbol
        assert sorted(mh.arg_params.keys()) == sorted(['conv1_weight', 'conv1_bias'])
        assert mh.aux_params == {}

        mod = mx.module.Module.load('tests/data/testnetv1', 0, label_names=['softmaxoutput1_label'])
        mh = model_handler.ModelHandler(mod, mx.context.cpu, 1)
        symbol = mh.symbol.get_internals()[self.conv2_id]

        mh.update_sym(symbol)

        assert mh.layer_type_dict == OrderedDict([('conv1', 'Convolution'), ('act1', 'Activation'),
                                                  ('conv2', 'Convolution')])
        assert mh.symbol == symbol
        assert sorted(mh.arg_params.keys()) == sorted(['conv1_weight', 'conv2_bias', 'conv1_bias', 'conv2_weight'])
        assert mh.aux_params == {}
