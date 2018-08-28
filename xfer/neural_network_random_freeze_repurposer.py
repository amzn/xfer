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

from .neural_network_repurposer import NeuralNetworkRepurposer
from .model_handler.model_handler import ModelHandler
from .model_handler import layer_factory
from .constants import neural_network_repurposer_keys as keys


class NeuralNetworkRandomFreezeRepurposer(NeuralNetworkRepurposer):
    """
    Class that creates a target neural network from a source neural network through Transfer Learning.
    It transfers layers from source model and fine-tunes the network to learn from target data set.

    Steps involved:

    - Layers and weights of source model are transferred
    - N layers are removed from the output of the model
    - A fully connected layer with nodes equal to number of classes in target data set is added to the model output
    - A softmax layer is added on top of the new fully connected layer
    - Some layers are frozen and some randomly initialized and then model is fine-tuned

    :param source_model: Source neural network to do transfer leaning from. Can be None in a predict-only case
    :type source_model: :class:`mxnet.mod.Module`
    :param int target_class_count: Number of classes to train the target neural network for. Can be None in a
                                   predict-only case
    :param list[str] fixed_layers: List of layers to keep weights frozen for
    :param list[str] random_layers: List of layers to randomly reinitialise
    :param int num_layers_to_drop: Number of layers to remove from model output
    :param context_function: MXNet context function that provides device type context
    :type context_function: function
    :param int num_devices: Number of devices to use to train target neural network
    :param int batch_size: Size of data batches to be used for training the target neural network
    :param int num_epochs: Number of epochs to be used for training the target neural network
    :param str optimizer: Optimizer required by MXNet to train target neural network. Default: 'sgd'
    :param optimizer_params: Optimizer params required by MXNet to train target neural network.
           Default: {'learning_rate': 1e-3}
    :type optimizer_params: dict(str, float)
    """
    def __init__(self, source_model: mx.mod.Module, target_class_count, fixed_layers, random_layers,
                 num_layers_to_drop=2, context_function=mx.context.cpu, num_devices=1, batch_size=64, num_epochs=5,
                 optimizer='sgd', optimizer_params=None):
        super().__init__(source_model, context_function, num_devices, batch_size, num_epochs,
                         optimizer, optimizer_params)

        self.target_class_count = target_class_count
        self.fixed_layers = fixed_layers
        self.random_layers = random_layers
        self.num_layers_to_drop = num_layers_to_drop

    def get_params(self):
        """
        Get parameters of repurposer that are in the constructor

        :rtype: dict
        """
        param_dict = super().get_params()
        param_dict[keys.TARGET_CLASS_COUNT] = self.target_class_count
        param_dict[keys.FIXED_LAYERS] = self.fixed_layers
        param_dict[keys.RANDOM_LAYERS] = self.random_layers
        param_dict[keys.NUM_LAYERS_TO_DROP] = self.num_layers_to_drop
        return param_dict

    def _create_target_module(self, train_iterator: mx.io.DataIter):
        """
        Function to create target module by transferring layers and weights of source model, removing layers from output
        and freezing and randomising selected layers

        Overrides abstract method in base NeuralNetworkRepurposer class. Called by repurpose method in base class to
        create target MXNet module that needs to be trained with target data set.

        :param train_iterator: Training data iterator. Used by this function to bind target module
        :type train_iterator: :class: `mxnet.io.DataIter`
        :return: Target MXNet module created by transferring layers from source model
        :rtype: :class:`mxnet.mod.Module`
        """
        # Create model handler to manipulate the source model
        model_handler = ModelHandler(self.source_model, self.context_function, self.num_devices)

        # Check if random and frozen layers are present in source model
        assert type(self.fixed_layers) == list, 'fixed_layers should be a list'
        assert type(self.random_layers) == list, 'random_layers should be a list'
        for layer_name in self.fixed_layers + self.random_layers:
            if layer_name not in model_handler.layer_names:
                raise ValueError('layer name: {} not found in source model'.format(layer_name))

        # Get lists of parameters to fix and randomise
        fixed_params = model_handler.get_layer_parameters(self.fixed_layers)
        random_params = model_handler.get_layer_parameters(self.random_layers)

        # Drop layers
        model_handler.drop_layer_top(self.num_layers_to_drop)

        # Add fully connected layer and softmax output
        fc = layer_factory.FullyConnected(name='new_fully_connected_layer', num_hidden=self.target_class_count)
        # Softmax layer name should be set to the name provided by the iterator
        softmax_name = train_iterator.provide_label[0][0].replace('_label', '')
        softmax = layer_factory.SoftmaxOutput(name=softmax_name)
        model_handler.add_layer_top([fc, softmax])

        # Create and return target MXNet module using the fixed/random parameters
        return model_handler.get_module(train_iterator, fixed_layer_parameters=fixed_params,
                                        random_layer_parameters=random_params)
