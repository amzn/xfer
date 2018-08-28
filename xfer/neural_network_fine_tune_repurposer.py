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


class NeuralNetworkFineTuneRepurposer(NeuralNetworkRepurposer):
    """
    Class that creates a target neural network from a source neural network through Transfer Learning.
    It transfers layers from source model and fine-tunes the network to learn from target data set.

    Steps involved:

    - Layers and weights of source model are transferred from input layer up to and including layer
      'transfer_layer_name'
    - A fully connected layer with nodes equal to number of classes in target data set is added after the transfer layer
    - A softmax layer is added on top of the new fully connected layer
    - This modified network represents the target model and is fine tuned to adapt weights to the target data set

    :param source_model: Source neural network to do transfer leaning from. Can be None in a predict-only case
    :type source_model: :class:`mxnet.mod.Module`
    :param str transfer_layer_name: Name of layer up to which layers/weights in source_model need to be transferred.
                                    For example, name of layer before the last fully connected layer in source network.
                                    Can be None in a predict-only case
    :param int target_class_count: Number of classes to train the target neural network for. Can be None in a
                                   predict-only case
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
    def __init__(self, source_model: mx.mod.Module, transfer_layer_name, target_class_count,
                 context_function=mx.context.cpu, num_devices=1, batch_size=64, num_epochs=5,
                 optimizer='sgd', optimizer_params=None):
        super().__init__(source_model, context_function, num_devices, batch_size, num_epochs,
                         optimizer, optimizer_params)

        self.transfer_layer_name = transfer_layer_name
        self.target_class_count = target_class_count

    def get_params(self):
        """
        Get parameters of repurposer that are in the constructor

        :rtype: dict
        """
        param_dict = super().get_params()
        param_dict[keys.TRANSFER_LAYER_NAME] = self.transfer_layer_name
        param_dict[keys.TARGET_CLASS_COUNT] = self.target_class_count
        return param_dict

    def _create_target_module(self, train_iterator: mx.io.DataIter):
        """
        Function to create target module by transferring layers and weights of source model from input layer up to and
        including layer 'transfer_layer_name'. A fully connected layer and softmax layer are added after the transfer
        layer.

        Overrides abstract method in base NeuralNetworkRepurposer class. Called by repurpose method in base class to
        create target MXNet module that needs to be trained with target data set.

        :param train_iterator: Training data iterator. Used by this function to bind target module
        :type train_iterator: :class: `mxnet.io.DataIter`
        :return: Target MXNet module created by transferring layers from source model
        :rtype: :class:`mxnet.mod.Module`
        """

        # Create model handler to manipulate the source model
        model_handler = ModelHandler(self.source_model, self.context_function, self.num_devices)

        # Create target symbol by transferring layers from source model up to and including 'transfer_layer_name'
        target_symbol = self._get_target_symbol(model_handler.layer_names)

        # Update model handler by replacing source symbol with target symbol
        # and cleaning up weights of layers that were not transferred
        model_handler.update_sym(target_symbol)

        # Add a fully connected layer (with nodes equal to number of target classes) and a softmax output layer on top
        fully_connected_layer = layer_factory.FullyConnected(num_hidden=self.target_class_count,
                                                             name='fc_from_fine_tune_repurposer')
        # Softmax layer name should be set to the name provided by the iterator
        softmax_name = train_iterator.provide_label[0][0].replace('_label', '')
        softmax_output_layer = layer_factory.SoftmaxOutput(name=softmax_name)
        model_handler.add_layer_top([fully_connected_layer, softmax_output_layer])

        # Create and return target MXNet module using the new symbol and params
        return model_handler.get_module(train_iterator)

    def _get_target_symbol(self, source_model_layer_names):
        """
        Create target symbol with source_model layers from input layer up to and including 'transfer_layer_name'

        :param source_model_layer_names: All layer names in source model. Used to check if transfer_layer_name is valid.

        :return: target_symbol
        :rtype: :class: `mx.symbol.Symbol`
        """
        # Check if 'transfer_layer_name' is present in source model
        if self.transfer_layer_name not in source_model_layer_names:
            raise ValueError('transfer_layer_name: {} not found in source model'.format(self.transfer_layer_name))

        # Create target symbol by transferring layers from source model up to 'transfer_layer_name'
        # Get layer key with output suffix to lookup MXNet symbol group
        transfer_layer_key = self.transfer_layer_name + '_output'
        source_symbol = self.source_model.symbol.get_internals()
        target_symbol = source_symbol[transfer_layer_key]
        return target_symbol
