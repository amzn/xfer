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
from ..repurposer_test_utils import RepurposerTestUtils
from .test_neural_network_repurposer import NeuralNetworkRepurposerTestCase
from xfer import NeuralNetworkFineTuneRepurposer


class NeuralNetworkFineTuneRepurposerTestCase(NeuralNetworkRepurposerTestCase):

    def setUp(self):
        super().setUp()
        self.repurposer_class = NeuralNetworkFineTuneRepurposer

    def test_get_target_model_symbol__invalid_transfer_layer_name(self):
        invalid_transfer_layer_name = 'phantom_layer'
        self.assertTrue(invalid_transfer_layer_name not in self.source_layers,
                        RepurposerTestUtils.ERROR_INCORRECT_INPUT)

        repurposer = self.repurposer_class(self.mxnet_model, transfer_layer_name=invalid_transfer_layer_name,
                                           target_class_count=2)
        expected_error_msg = 'transfer_layer_name: {} not found in source model'.format(invalid_transfer_layer_name)
        self.assertRaisesRegex(ValueError, expected_error_msg, repurposer._get_target_symbol, self.source_layers)

    def test_get_target_model_symbol__valid_transfer_layer_name(self):
        # Create repurposer with valid transfer layer name
        valid_transfer_layer_name = 'relu1'
        self.assertTrue(valid_transfer_layer_name in self.source_layers, RepurposerTestUtils.ERROR_INCORRECT_INPUT)
        repurposer = self.repurposer_class(self.mxnet_model, transfer_layer_name=valid_transfer_layer_name,
                                           target_class_count=2)

        # Get target symbol
        target_symbol = repurposer._get_target_symbol(self.source_layers)

        # Validate if target symbol is transferred at the given 'transfer layer'
        self.assertTrue(target_symbol.name == valid_transfer_layer_name,
                        'Target symbol is transferred at incorrect layer:{} instead of:{}'.format
                        (target_symbol.name, valid_transfer_layer_name))

        # Valid if target symbol contains all layer outputs up to the transfer layer
        source_symbol_outputs = self.mxnet_model.symbol.get_internals().list_outputs()
        self.assertTrue(source_symbol_outputs == ['data', 'fc1_weight', 'fc1_bias', 'fc1_output', 'relu1_output',
                                                  'fc2_weight', 'fc2_bias', 'fc2_output', 'softmax_label',
                                                  'softmax_output'],
                        RepurposerTestUtils.ERROR_INCORRECT_INPUT)
        target_symbol_outputs = target_symbol.get_internals().list_outputs()
        self.assertTrue(target_symbol_outputs == ['data', 'fc1_weight', 'fc1_bias', 'fc1_output', 'relu1_output'],
                        'Target symbol has incorrect layer outputs')

    def test_get_params(self):
        repurposer = self.repurposer_class(self.mxnet_model, 'fc1', 4)

        params = repurposer.get_params()
        expected_params = {
            'context_function': 'cpu',
            'num_devices': 1,
            'optimizer': 'sgd',
            'optimizer_params': {'learning_rate': 0.001},
            'batch_size': 64,
            'num_epochs': 5,
            'transfer_layer_name': 'fc1',
            'target_class_count': 4
        }

        assert params == expected_params

    def _get_repurposer(self, source_model):
        return self.repurposer_class(source_model, transfer_layer_name='fullyconnected0', target_class_count=4,
                                     num_epochs=2)
