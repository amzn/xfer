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
import numpy as np
from abc import abstractmethod

from .repurposer import Repurposer
from .constants import serialization_keys, repurposer_keys, serialization_constants
from .constants import neural_network_repurposer_keys as keys
from .constants import neural_network_repurposer_constants as consts
from .model_handler import ModelHandler
from . import utils


class NeuralNetworkRepurposer(Repurposer):
    """
    Base class for repurposers that create a target neural network from a source neural network through Transfer
    Learning.

    - Transfer layer architecture and weights from a source neural network (Transfer) and
    - Train a target neural network adapting the transferred network to new data set (Learn)

    :param source_model: Source neural network to do transfer leaning from
    :type source_model: :class:`mxnet.mod.Module`
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
    def __init__(self, source_model: mx.mod.Module, context_function=mx.context.cpu, num_devices=1, batch_size=64,
                 num_epochs=5, optimizer='sgd', optimizer_params=None):
        super(NeuralNetworkRepurposer, self).__init__(source_model, context_function, num_devices)
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self._save_source_model_default = False
        self.optimizer = optimizer
        self.optimizer_params = optimizer_params
        if self.optimizer_params is None:
            self.optimizer_params = {keys.LEARNING_RATE: consts.DEFAULT_LEARNING_RATE}

    def get_params(self):
        """
        Get parameters of repurposer that are in the constructor

        :rtype: dict
        """
        param_dict = super(NeuralNetworkRepurposer, self).get_params()
        param_dict[keys.OPTIMIZER] = self.optimizer
        param_dict[keys.OPTIMIZER_PARAMS] = self.optimizer_params
        param_dict[keys.BATCH_SIZE] = self.batch_size
        param_dict[keys.NUM_EPOCHS] = self.num_epochs
        return param_dict

    def _get_attributes(self):
        """
        Get parameters of repurposer not in constructor

        :rtype: dict
        """
        attr_dict = super()._get_attributes()
        if self.target_model is not None:
            attr_dict[serialization_keys.LAST_LAYER_NAME_TARGET] = ModelHandler(self.target_model).layer_names[-1]
        return attr_dict

    def repurpose(self, train_iterator: mx.io.DataIter):
        """
        Train a neural network by transferring layers/weights from source_model
        Set self.target_model to the repurposed neural network

        :param train_iterator: Training data iterator to use to extract features from source_model
        :type train_iterator: :class:`mxnet.io.DataIter`
        """

        # Validate the repurpose call
        self._validate_before_repurpose()

        # Prepare target model symbol (add or remove layers/ transfer weights/ freeze/ fine tune/ etc.)
        model = self._create_target_module(train_iterator)

        # Reset iterator before using it to train
        train_iterator.reset()

        # Train the model using given training data in batches
        model.fit(train_iterator,
                  optimizer=self.optimizer,
                  optimizer_params=self.optimizer_params,
                  kvstore='device',
                  eval_metric='acc',
                  allow_missing=True,
                  batch_end_callback=mx.callback.Speedometer(self.batch_size),
                  initializer=mx.init.Xavier(rnd_type='gaussian', factor_type='in', magnitude=2),
                  num_epoch=self.num_epochs)

        # Set self.target_model to the repurposed model
        self.target_model = model

        # Save data_shape and label_shapes
        self.provide_data = train_iterator.provide_data
        self.provide_label = train_iterator.provide_label

    def predict_probability(self, test_iterator: mx.io.DataIter):
        """
        Perform predictions on test data using the target_model i.e. repurposed neural network

        :param test_iterator: Test data iterator to return predictions for
        :type test_iterator: :class:`mxnet.io.DataIter`
        :return: Predicted probabilities
        :rtype: :class:`numpy.ndarray`
        """

        # Validate the predict call
        self._validate_before_predict()

        # Bind target model symbol with data if not already bound
        if not self.target_model.binded:
            self.target_model.bind(data_shapes=test_iterator.provide_data, for_training=False)

        # Call mxnet.BaseModule.predict. It handles batch padding and resets iterator before prediction.
        predictions = self.target_model.predict(eval_data=test_iterator)
        return predictions.asnumpy()

    def predict_label(self, test_iterator: mx.io.DataIter):
        """
        Perform predictions on test data using the target_model i.e. repurposed neural network

        :param test_iterator: Test data iterator to return predictions for
        :type test_iterator: mxnet.io.DataIter
        :return: Predicted labels
        :rtype: :class:`numpy.ndarray`
        """

        # Call predict_probability method to get soft predictions
        probabilities = self.predict_probability(test_iterator)

        # Select and return label with maximum probability for each test instance
        labels = np.argmax(probabilities, axis=1)

        return labels

    @abstractmethod
    def _create_target_module(self, train_iterator: mx.io.DataIter):
        """
        Abstract method.
        """
        pass

    def _validate_before_predict(self):
        # For neural network repurposers, target_model should be an MXNet module object
        if not isinstance(self.target_model, mx.mod.Module):
            error = ("Cannot predict because target_model is not an `mxnet.mod.Module` object. "
                     " Instead got type: {}".format(type(self.target_model)))
            raise TypeError(error)

        if not self.target_model.params_initialized:
            error = "target_model params aren't initialized. Ensure model is trained before calling predict"
            raise ValueError(error)

    def serialize(self, file_prefix):
        """
        Serialize repurposer to dictionary

        :return: Dictionary describing repurposer
        :rtype: dict
        """
        output_dict = {}
        output_dict[repurposer_keys.PARAMS] = self.get_params()
        output_dict.update(self._get_attributes())

        # save serialised model file_path.json
        utils.save_json(file_prefix, output_dict)

        # Save target_model as file-path-symbol.json and file-path-0000.params
        utils.save_mxnet_model(self.target_model, file_prefix, 0, self.provide_data, self.provide_label)

    def deserialize(self, input_dict):
        """
        Uses dictionary to set attributes of repurposer

        :param dict input_dict: Dictionary containing values for attributes to be set to
        """
        self.target_model = mx.mod.Module.load(input_dict[serialization_keys.FILE_PATH], 0,
                                               label_names=[input_dict[serialization_keys.LAST_LAYER_NAME_TARGET] +
                                                            serialization_constants.LABEL_SUFFIX])
        self._set_attributes(input_dict)  # Set attributes of repurposer from input_dict
