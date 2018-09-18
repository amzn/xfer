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
import mxnet as mx

from .test_neural_network_repurposer import NeuralNetworkRepurposerTestCase
from xfer import NeuralNetworkRandomFreezeRepurposer


class NeuralNetworkRandomFreezeRepurposerTestCase(NeuralNetworkRepurposerTestCase):

    def setUp(self):
        super().setUp()
        self.repurposer_class = NeuralNetworkRandomFreezeRepurposer

        self.data_name = 'data'
        self.imglist = [[0, 'accordion/image_0001.jpg'], [0, 'accordion/image_0002.jpg'], [1, 'ant/image_0001.jpg'],
                        [1, 'ant/image_0002.jpg'], [2, 'anchor/image_0001.jpg'], [2, 'anchor/image_0002.jpg']]
        self.train_iter = mx.image.ImageIter(2, (3, 224, 224), imglist=self.imglist, path_root='tests/data/test_images',
                                             label_name='softmax_label', data_name=self.data_name)

    def test_create_target_module__invalid_fixed_layer(self):
        repurposer = self.repurposer_class(self.mxnet_model, target_class_count=2, fixed_layers=['invented_layer'],
                                           random_layers=[])

        with self.assertRaises(ValueError):
            repurposer._create_target_module(4)

    def test_create_target_module__invalid_random_layer(self):
        repurposer = self.repurposer_class(self.mxnet_model, target_class_count=2, random_layers=['invented_layer'],
                                           fixed_layers=[])

        with self.assertRaises(ValueError):
            repurposer._create_target_module(4)

    def test_get_target_model_symbol__valid_transfer_layer_name(self):
        repurposer = self.repurposer_class(self.mxnet_model, target_class_count=2, random_layers=['fc2'],
                                           fixed_layers=['fc1'])

        # Get target module
        target_module = repurposer._create_target_module(self.train_iter)
        target_model_outputs = target_module.symbol.get_internals().list_outputs()

        assert target_model_outputs == ['data', 'fc1_weight', 'fc1_bias', 'fc1_output', 'relu1_output',
                                        'new_fully_connected_layer_weight', 'new_fully_connected_layer_bias',
                                        'new_fully_connected_layer_output', 'softmax_label', 'softmax_output']

    def test_get_params(self):
        repurposer = self.repurposer_class(self.mxnet_model, target_class_count=4, random_layers=['rand1', 'rand2'],
                                           fixed_layers=['fixed1', 'fixed2', 'fixed3'])

        params = repurposer.get_params()
        expected_params = {
            'context_function': 'cpu',
            'num_devices': 1,
            'optimizer': 'sgd',
            'optimizer_params': {'learning_rate': 0.001},
            'batch_size': 64,
            'num_epochs': 5,
            'target_class_count': 4,
            'fixed_layers': ['fixed1', 'fixed2', 'fixed3'],
            'random_layers': ['rand1', 'rand2'],
            'num_layers_to_drop': 2
        }

        assert params == expected_params

    def _get_repurposer(self, source_model):
        return self.repurposer_class(source_model, target_class_count=2, random_layers=['conv2'],
                                     fixed_layers=['conv1'], num_epochs=2)
