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
from abc import ABCMeta, abstractmethod
import os
import warnings
import mxnet as mx

from . import utils
from .constants import serialization_keys, constants, serialization_constants
from .constants import repurposer_keys as keys
from .model_handler import ModelHandler


class Repurposer:
    """
    Base Class for repurposers that train models using Transfer Learning (source_model -> target_model).

    :param source_model: Source neural network to do transfer learning from.
    :type source_model: :class:`mxnet.mod.Module`
    :param context_function: MXNet context function that provides device type context.
    :type context_function: function(int)->:class:`mx.context.Context`
    :param int num_devices: Number of devices to use with context_function.
    """
    __metaclass__ = ABCMeta

    def __init__(self, source_model: mx.mod.Module, context_function=mx.context.cpu, num_devices=1):
        self.source_model = source_model
        self.context_function = context_function
        self.num_devices = num_devices
        self.target_model = None
        self._save_source_model_default = None
        self.provide_data = None
        self.provide_label = None

    @abstractmethod
    def repurpose(self, train_iterator: mx.io.DataIter):
        """
        Abstract method.
        """
        pass

    @abstractmethod
    def predict_probability(self, test_iterator: mx.io.DataIter):
        """
        Abstract method.
        """
        pass

    @abstractmethod
    def predict_label(self, test_iterator: mx.io.DataIter):
        """
        Abstract method.
        """
        pass

    def get_params(self):
        """
        Get parameters of repurposer that are in the constructor's argument list.

        :rtype: dict
        """
        param_dict = {}
        param_dict[keys.CONTEXT_FN] = utils.serialize_ctx_fn(self.context_function)
        param_dict[keys.NUM_DEVICES] = self.num_devices
        return param_dict

    def _get_attributes(self):
        """
        Get parameters of repurposer not in constructor's argument list.

        :rtype: dict
        """
        attr_dict = {}
        attr_dict[keys.PROVIDE_DATA] = self.provide_data
        attr_dict[keys.PROVIDE_LABEL] = self.provide_label
        attr_dict[serialization_keys.VERSION] = constants.VERSION
        attr_dict[keys.REPURPOSER_CLASS] = self.__class__.__name__
        attr_dict[serialization_keys.LAST_LAYER_NAME_SOURCE] = ModelHandler(self.source_model).layer_names[-1]
        return attr_dict

    def _set_attributes(self, input_dict):
        """
        Set attributes of class from input_dict.
        These attributes are the same as those returned by get_attributes method.

        :param input_dict: Dictionary containing attribute values.
        :return: None
        """

        # Raise warning if there is a version mismatch with the Xfer version saved in input_dict
        if not constants.VERSION == input_dict[serialization_keys.VERSION]:
            warnings.warn("Running Xfer version: {} which is different from version saved in dictionary: {}. "
                          "Ensure that you are using the correct version of Xfer"
                          .format(constants.VERSION, input_dict[serialization_keys.VERSION]))
        self.provide_data = input_dict[keys.PROVIDE_DATA]
        self.provide_label = input_dict[keys.PROVIDE_LABEL]

    @abstractmethod
    def serialize(self, file_prefix):
        """
        Abstract method.
        """
        pass

    @abstractmethod
    def deserialize(self, input_dict):
        """
        Abstract method.
        """
        pass

    def save_repurposer(self, model_name, model_directory='', save_source_model=None):
        """
        Serialize the repurposed model (source_model, target_model and supporting info) and save it to given file_path.

        :param str model_name: Name to save repurposer to.
        :param str model_directory: File directory to save repurposer in.
        :param boolean save_source_model: Flag to choose whether to save repurposer source model.
                                          Will use default if set to None. (MetaModelRepurposer default: True,
                                          NeuralNetworkRepurposer: False)
        """
        # Assert repurpose() has been called successfully
        if self.target_model is None:
            raise ValueError('Cannot save repurposer before source model has been repurposed')

        file_prefix = os.path.join(model_directory, model_name)

        self._save_source_model(file_prefix, save_source_model)

        # Serialize remainder of repurposer
        self.serialize(file_prefix)

    def _save_source_model(self, file_prefix, save_source_model):
        # Set save_source_model flag to class default if None
        save_source_model = self._save_source_model_default if save_source_model is None else save_source_model
        if save_source_model:
            # save source_model as file-path_source-symbol.json and file-path_source-0000.params
            utils.save_mxnet_model(self.source_model, file_prefix + serialization_constants.SOURCE_SUFFIX, 0,
                                   self.provide_data, self.provide_label)

    def _validate_before_repurpose(self):
        if not isinstance(self.source_model, mx.mod.Module):
            error = ("Cannot repurpose because source_model is not an `mxnet.mod.Module` object."
                     " Instead got type: {}".format(type(self.source_model)))
            raise TypeError(error)

    def _validate_before_predict(self):
        if self.target_model is None:
            raise ValueError("Cannot predict because target_model is not initialized")
