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

import json
import os
from collections import OrderedDict
import graphviz
import mxnet as mx
import numpy as np

from xfer import model_handler
from xfer.model_handler import consts, exceptions
from ..repurposer_test_utils import RepurposerTestUtils


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
        self.symbol_dict = json.loads(self.mh.symbol.tojson())
        self.nodes = self.symbol_dict['nodes']
        self.arg_nodes = self.symbol_dict['arg_nodes']
        self.heads = self.symbol_dict['heads']
        self.act1_id = 4
        self.conv2_id = 7

    def tearDown(self):
        del self.mh

    def test_resnet(self):
        """Assert that ModelHandler can drop layers from resnet without errors"""
        RepurposerTestUtils.download_resnet()
        mod = mx.mod.Module.load('resnet-101', 0)
        mh = model_handler.ModelHandler(mod)
        old_layer_names = mh.layer_names

        mh.drop_layer_top()
        mh.drop_layer_bottom()

        assert sorted(list(set(old_layer_names).difference(set(mh.layer_names)))) == sorted(['bn_data', 'softmax'])

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

    def test_drop_layer_top_two_outputs(self):
        # Build a symbol with two softmax output layers
        data = mx.sym.Variable('data')
        fc1 = mx.sym.FullyConnected(data=data, name='fc1', num_hidden=128)
        act1 = mx.sym.Activation(data=fc1, name='relu1', act_type="relu")
        fc2 = mx.sym.FullyConnected(data=act1, name='fc2', num_hidden=64)
        act2 = mx.sym.Activation(data=fc2, name='relu2', act_type="relu")
        fc3 = mx.sym.FullyConnected(data=act2, name='fc3', num_hidden=10)

        fc4 = mx.sym.FullyConnected(data=fc3, name='fc4_1', num_hidden=10)
        sm1 = mx.sym.SoftmaxOutput(data=fc4, name='softmax1')
        fc5 = mx.sym.FullyConnected(data=fc3, name='fc4_2', num_hidden=10)
        sm2 = mx.sym.SoftmaxOutput(data=fc5, name='softmax2')

        softmax = mx.sym.Group([sm1, sm2])

        mod = mx.mod.Module(softmax, label_names=['softmax1_label', 'softmax2_label'])
        mh = model_handler.ModelHandler(mod)

        with self.assertRaises(exceptions.ModelError):
            mh.drop_layer_top()

    def test_drop_layer_top_split(self):
        mh, plus_layer_name = self._build_split_net()
        mh.drop_layer_top()
        with self.assertRaises(exceptions.ModelError):
            mh.drop_layer_top()
        with self.assertRaises(exceptions.ModelError):
            mh.drop_layer_top(2)

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
                                                                            'conv2_bias', 'conv2_output',
                                                                            'conv2_weight'}

    def test_drop_layer_bottom_too_many(self):
        with self.assertRaises(model_handler.exceptions.ModelError):
            self.mh.drop_layer_bottom(8)

    @staticmethod
    def _build_split_net():
        """Instantiate MH for a model that diverges into two and then joins back into one."""
        data = mx.sym.var('data')
        data = mx.sym.flatten(data=data, name='flatten0')
        fc1 = mx.sym.FullyConnected(data, num_hidden=5, name='a_1')
        fc2 = mx.sym.FullyConnected(fc1, num_hidden=5, name='a_2')
        fc3 = mx.sym.FullyConnected(fc2, num_hidden=5, name='a_3')
        fc2 = mx.sym.FullyConnected(data, num_hidden=5, name='b_1')
        fc2b = mx.sym.FullyConnected(fc2, num_hidden=5, name='b_2')
        plus = fc3.__add__(fc2b)

        softmax = mx.sym.SoftmaxOutput(plus, name='softmax')
        mod = mx.mod.Module(softmax)
        mh = model_handler.ModelHandler(mod)

        plus_layer_name = mh.layer_names[6]

        return mh, plus_layer_name

    def test_drop_layer_bottom_split(self):
        mh, _ = self._build_split_net()
        mh.drop_layer_bottom()
        with self.assertRaises(exceptions.ModelError):
            mh.drop_layer_bottom()
        with self.assertRaises(exceptions.ModelError):
            mh.drop_layer_bottom(2)

    def test_add_layer_top(self):
        # Drop output layer so that layers can be added to top
        self.mh.drop_layer_top()
        layer1 = mx.sym.FullyConnected(name='fc1', num_hidden=5)
        assert 'fc1' not in list(self.mh.layer_type_dict.keys())

        outputs_pre = self.mh.symbol.get_internals().list_outputs()
        self.mh.add_layer_top([layer1])
        outputs_post = self.mh.symbol.get_internals().list_outputs()

        assert 'fc1' in list(self.mh.layer_type_dict.keys())
        assert outputs_post == outputs_pre + ['fc1_weight', 'fc1_bias', 'fc1_output']

    def test_add_layer_top_2(self):
        self.mh.drop_layer_top()
        layer1 = mx.sym.FullyConnected(name='fc1', num_hidden=5)
        layer2 = mx.sym.Convolution(name='conv1_1', kernel=(3, 3), num_filter=10)
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
        layer1 = mx.sym.FullyConnected(name='fc1', num_hidden=5)
        layer2 = mx.sym.Convolution(name='conv1_1', kernel=(3, 3), num_filter=10)
        for layer_name in ['fc1', 'conv1_1']:
            assert layer_name not in list(self.mh.layer_type_dict.keys())

        outputs_pre = self.mh.symbol.get_internals().list_outputs()
        self.mh.add_layer_top([layer1, layer2])
        outputs_post = self.mh.symbol.get_internals().list_outputs()

        for layer_name in ['fc1', 'conv1_1']:
            assert layer_name in list(self.mh.layer_type_dict.keys())
        assert outputs_post == outputs_pre + ['fc1_weight', 'fc1_bias', 'fc1_output', 'conv1_1_weight', 'conv1_1_bias',
                                              'conv1_1_output']

    def test_add_layer_top_over_output(self):
        # Assert model error raised when a layer is added above an output layer
        layer = mx.sym.FullyConnected(num_hidden=5)
        with self.assertRaises(exceptions.ModelError):
            self.mh.add_layer_top(layer)

    def test_add_layer_bottom(self):
        layer1 = mx.sym.Convolution(name='conv1_1', kernel=(3, 3), num_filter=10)
        assert 'conv1_1' not in list(self.mh.layer_type_dict.keys())

        outputs_pre = self.mh.symbol.get_internals().list_outputs()
        self.mh.add_layer_bottom([layer1])
        outputs_post = self.mh.symbol.get_internals().list_outputs()

        assert 'conv1_1' in list(self.mh.layer_type_dict.keys())
        assert outputs_post == [self.data_name, 'conv1_1_weight', 'conv1_1_bias', 'conv1_1_output'] + outputs_pre[1:]

    def test_add_layer_bottom_2(self):
        layer1 = mx.sym.Convolution(name='conv1_1', kernel=(3, 3), num_filter=10)
        layer2 = mx.sym.FullyConnected(name='fc1', num_hidden=10)
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
        layer1 = mx.sym.Convolution(name='conv1_1', kernel=(3, 3), num_filter=10)
        layer2 = mx.sym.FullyConnected(name='fc1', num_hidden=10)
        for layer_name in ['fc1', 'conv1_1']:
            assert layer_name not in list(self.mh.layer_type_dict.keys())

        outputs_pre = self.mh.symbol.get_internals().list_outputs()
        self.mh.add_layer_bottom([layer1, layer2])
        outputs_post = self.mh.symbol.get_internals().list_outputs()

        for layer_name in ['fc1', 'conv1_1']:
            assert layer_name in list(self.mh.layer_type_dict.keys())
        assert outputs_post == [self.data_name, 'conv1_1_weight', 'conv1_1_bias', 'conv1_1_output', 'fc1_weight',
                                'fc1_bias', 'fc1_output'] + outputs_pre[1:]

    def test_assert_model_has_single_output(self):
        data = mx.sym.Variable('data')
        fc1 = mx.sym.FullyConnected(data=data, name='fc1', num_hidden=128)
        act1 = mx.sym.Activation(data=fc1, name='relu1', act_type="relu")
        fc2 = mx.sym.FullyConnected(data=act1, name='fc2', num_hidden=64)
        act2 = mx.sym.Activation(data=fc2, name='relu2', act_type="relu")
        fc3 = mx.sym.FullyConnected(data=act2, name='fc3', num_hidden=10)
        fc4 = mx.sym.FullyConnected(data=fc3, name='fc4_1', num_hidden=10)
        sm1 = mx.sym.SoftmaxOutput(data=fc4, name='softmax1')
        fc5 = mx.sym.FullyConnected(data=fc3, name='fc4_2', num_hidden=10)
        sm2 = mx.sym.SoftmaxOutput(data=fc5, name='softmax2')
        sm3 = mx.sym.SoftmaxOutput(data=fc2, name='softmax3')

        output_1 = sm1
        output_2 = mx.sym.Group([sm1, sm2])
        output_3 = mx.sym.Group([sm1, sm2, sm3])

        self.mh._assert_model_has_single_output(self.mh._get_symbol_dict(output_1))
        with self.assertRaises(exceptions.ModelError):
            self.mh._assert_model_has_single_output(self.mh._get_symbol_dict(output_2))
        with self.assertRaises(exceptions.ModelError):
            self.mh._assert_model_has_single_output(self.mh._get_symbol_dict(output_3))

    def test_get_names_of_inputs_to_layer(self):
        symbol_dict = self.mh._get_symbol_dict(self.mh.symbol)

        assert self.mh._get_names_of_inputs_to_layer(symbol_dict, 7) == [4]
        assert self.mh._get_names_of_inputs_to_layer(symbol_dict, 8) == [7]
        assert self.mh._get_names_of_inputs_to_layer(symbol_dict, 15) == [13]

    def test_get_names_of_inputs_to_layer_split_2(self):
        data = mx.sym.Variable('data')
        fc1 = mx.sym.FullyConnected(data=data, name='fc1', num_hidden=128)
        act1 = mx.sym.Activation(data=fc1, name='relu1', act_type="relu")
        fc2 = mx.sym.FullyConnected(data=act1, name='fc2', num_hidden=64)
        act2 = mx.sym.Activation(data=fc2, name='relu2', act_type="relu")
        fc3 = mx.sym.FullyConnected(data=act2, name='fc3', num_hidden=10)
        plus = fc2.__add__(fc3)

        symbol_dict = self.mh._get_symbol_dict(plus)

        assert self.mh._get_names_of_inputs_to_layer(symbol_dict, 7) == [4]
        assert self.mh._get_names_of_inputs_to_layer(symbol_dict, 8) == [7]
        assert self.mh._get_names_of_inputs_to_layer(symbol_dict, 12) == [7, 11]

    def test_get_names_of_inputs_to_layer_split_3(self):
        data = mx.sym.Variable('data')
        fc1 = mx.sym.FullyConnected(data=data, name='fc1', num_hidden=128)
        act1 = mx.sym.Activation(data=fc1, name='relu1', act_type="relu")
        fc2 = mx.sym.FullyConnected(data=act1, name='fc2', num_hidden=64)
        act2 = mx.sym.Activation(data=fc2, name='relu2', act_type="relu")
        fc3 = mx.sym.FullyConnected(data=act2, name='fc3', num_hidden=10)
        concat = mx.sym.concat(fc1, fc2, fc3, name='concat1')

        symbol_dict = self.mh._get_symbol_dict(concat)

        assert self.mh._get_names_of_inputs_to_layer(symbol_dict, 4) == [3]
        assert self.mh._get_names_of_inputs_to_layer(symbol_dict, 11) == [8]
        assert self.mh._get_names_of_inputs_to_layer(symbol_dict, 12) == [3, 7, 11]

    def test_get_arg_nodes(self):
        assert self.mh._get_arg_nodes(self.nodes) == [0, 1, 2, 5, 6, 11, 12, 14]

        null_node = {'op': 'null'}
        op_node = {'op': 'FullyConnected'}
        nodes = [null_node, op_node, op_node, null_node, null_node, null_node, op_node]

        assert self.mh._get_arg_nodes(nodes) == [0, 3, 4, 5]

    def test_get_heads(self):
        assert self.mh._get_heads(self.nodes, 'softmaxoutput1') in [[[15, 0, 0]], [[15, 0]]]
        assert self.mh._get_heads(self.nodes[-11:], 'softmaxoutput1') in [[[10, 0, 0]], [[10, 0]]]
        assert self.mh._get_heads(self.nodes[-7:], 'softmaxoutput1') in [[[6, 0, 0]], [[6, 0]]]

    def test_get_output_layer_names(self):
        assert self.mh._get_output_layer_names(self.symbol_dict) == ['softmaxoutput1']

        data = mx.sym.Variable('data')
        fc1 = mx.sym.FullyConnected(data=data, name='fc1', num_hidden=128)
        act1 = mx.sym.Activation(data=fc1, name='relu1', act_type="relu")
        fc2 = mx.sym.FullyConnected(data=act1, name='fc2', num_hidden=64)
        act2 = mx.sym.Activation(data=fc2, name='relu2', act_type="relu")
        fc3 = mx.sym.FullyConnected(data=act2, name='fc3', num_hidden=10)
        fc4 = mx.sym.FullyConnected(data=fc3, name='fc4_1', num_hidden=10)
        sm1 = mx.sym.SoftmaxOutput(data=fc4, name='softmax1')
        fc5 = mx.sym.FullyConnected(data=fc3, name='fc4_2', num_hidden=10)
        sm2 = mx.sym.SoftmaxOutput(data=fc5, name='softmax2')
        sm3 = mx.sym.SoftmaxOutput(data=fc2, name='softmax3')

        outputs = [sm1, mx.sym.Group([sm1, sm2]), mx.sym.Group([sm1, sm2, sm3])]
        output_names = [['softmax1'], ['softmax1', 'softmax2'], ['softmax1', 'softmax2', 'softmax3']]

        for output, output_name in zip(outputs, output_names):
            symbol_dict = self.mh._get_symbol_dict(output)
            self.mh._get_output_layer_names(symbol_dict) == output_name

    @staticmethod
    def _build_symbol_with_nodes_with_zero_input():
        data = mx.sym.Variable('data')
        fc1a = mx.sym.FullyConnected(data=data, name='fc1a', num_hidden=128)
        act1a = mx.sym.Activation(data=fc1a, name='relu1a', act_type="relu")
        fc1b = mx.sym.FullyConnected(data=data, name='fc1b', num_hidden=64)
        act1b = mx.sym.Activation(data=fc1b, name='relu1b', act_type="relu")
        plus = act1a.__add__(act1b)
        softmax = mx.sym.SoftmaxOutput(data=plus, name='softmax')
        return softmax

    def test_get_layers_with_node_idx_as_input(self):
        softmax = self._build_symbol_with_nodes_with_zero_input()
        symbol_dict = self.mh._get_symbol_dict(softmax)

        expected_ids = [
            [3, 7],
            [3],
            [3],
            [4],
            [9],
            [7],
            [7],
            [8],
            [9],
            [11]
        ]
        for i in range(10):
            assert self.mh._get_layers_with_node_idx_as_input(i, symbol_dict['nodes']) == expected_ids[i]

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
        # Assert ModelArchitectureError is raised when iterator dimensions do not match module weights
        image_iter = mx.image.ImageIter(2, (3, 123, 123), imglist=self.imglist, path_root='test_images',
                                        label_name='softmaxoutput1_label')
        with self.assertRaises(model_handler.exceptions.ModelArchitectureError):
            self.mh.get_module(image_iter)
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
        layers_found = self.mh.get_layer_names_matching_type('CONVOLUTION')
        assert sorted(layers_found) == sorted(['conv1', 'conv2'])
        layers_found = self.mh.get_layer_names_matching_type('activation')
        assert sorted(layers_found) == sorted(['act1', 'act2'])
        layers_found = self.mh.get_layer_names_matching_type('Pooling')
        assert layers_found == ['pool1']
        layers_found = self.mh.get_layer_names_matching_type('flatten')
        assert layers_found == ['flatten1']
        layers_found = self.mh.get_layer_names_matching_type('fullyconnected')
        assert layers_found == ['fullyconnected0']
        layers_found = self.mh.get_layer_names_matching_type('SoftmaxOutput')
        assert layers_found == ['softmaxoutput1']
        layers_found = self.mh.get_layer_names_matching_type('BatchNorm')
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

        assert sorted(features.keys()) == sorted([layer1, layer2])

        expected_features = {}
        expected_features[layer1] = np.array([[-14632.44140625, -15410.29882812, 401.85760498, 3048.72143555],
                                              [-14077.45507812, -14598.43457031, 1666.03442383, 7222.70019531],
                                              [-16092.02929688, -16345.94042969, -2379.76391602, 6223.95605469],
                                              [-15005.03808594, -15088.01660156, 5304.77539062, 6113.68164062],
                                              [-12687.98339844, -14598.75097656, 1158.26794434, 4819.8671875],
                                              [-11366.10351562, -16437.70507812, -4690.74951172, 8055.31982422]])
        expected_labels = [0, 0, 1, 1, 2, 2]
        num_test_instances = len(expected_labels)

        expected_feature_label_dict = {}
        for index, expected_label in enumerate(expected_labels):
            key = hash(str(expected_features[layer1][index].astype(int)))
            expected_feature_label_dict[key] = (expected_features[layer1][index], expected_label)
        assert len(expected_feature_label_dict) == num_test_instances

        actual_feature_label_dict = {}
        for index, actual_label in enumerate(labels):
            key = hash(str(features[layer1][index].astype(int)))
            actual_feature_label_dict[key] = (features[layer1][index], actual_label)
        assert len(actual_feature_label_dict) == num_test_instances

        # Compare if <feature, label> pairs are returned as expected
        for key in expected_feature_label_dict:
            self.assertTrue(key in actual_feature_label_dict, "Expected features not found")

            expected_features = expected_feature_label_dict[key][0]
            actual_features = actual_feature_label_dict[key][0]
            self.assertTrue(np.allclose(expected_features, actual_features, rtol=tolerance),
                            "Expected features:{}. Actual:{}".format(expected_features, actual_features))

            expected_label = expected_feature_label_dict[key][1]
            actual_label = actual_feature_label_dict[key][1]
            self.assertTrue(expected_label == actual_label,
                            "Expected label:{}. Actual:{}".format(expected_label, actual_label))

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

    # TODO: Review test to handle ordering offset in CSVIter
    # def test_get_layer_output_csv_iterator(self):
    #     # test for when there is no padding
    #     self._test_get_layer_output_csv_iterator(batch_size=2)

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
        assert isinstance(self.mh.visualize_net(), graphviz.dot.Digraph)

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
        assert cm.output == ['WARNING:root:Could not find layer parameters: conv3_weight, conv3_moving_var']

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

        real_symbol_dict = self.mh._get_symbol_dict(self.mh.symbol)

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
        assert mh.symbol.tojson() == symbol.tojson()
        self._compare_symbols(mh.symbol, symbol)
        assert sorted(mh.arg_params.keys()) == sorted(['conv1_weight', 'conv1_bias'])
        assert mh.aux_params == {}

        mod = mx.module.Module.load('tests/data/testnetv1', 0, label_names=['softmaxoutput1_label'])
        mh = model_handler.ModelHandler(mod, mx.context.cpu, 1)
        symbol = mh.symbol.get_internals()[self.conv2_id]

        mh.update_sym(symbol)

        assert mh.layer_type_dict == OrderedDict([('conv1', 'Convolution'), ('act1', 'Activation'),
                                                  ('conv2', 'Convolution')])
        assert mh.symbol.tojson() == symbol.tojson()
        self._compare_symbols(mh.symbol, symbol)
        assert sorted(mh.arg_params.keys()) == sorted(['conv1_weight', 'conv2_bias', 'conv1_bias', 'conv2_weight'])
        assert mh.aux_params == {}

    @staticmethod
    def _compare_symbols(sym1, sym2):
        """
        Compare two symbols.

        :param sym1: Actual symbol
        :param sym2: Expected symbol
        """
        assert sym1.get_internals().list_outputs() == sym2.get_internals().list_outputs(), 'Symbol outputs \
            mismatch. Expected: {}, Got: {}'.format(sym2.get_internals().list_outputs(),
                                                    sym1.get_internals().list_outputs())

        for i, _ in enumerate(sym1.get_internals()):
            assert sym1.get_internals()[i].get_internals().list_outputs() == \
                   sym1.get_internals()[i].get_internals().list_outputs()

        data_shape = (5, 3, 224, 224)
        assert sym1.infer_shape(data=data_shape) == sym2.infer_shape(data=data_shape)
