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
from abc import abstractmethod
from collections import OrderedDict
import mxnet as mx
import numpy as np

from .repurposer import Repurposer
from .model_handler import ModelHandler
from .constants import meta_model_repurposer_keys as keys


class MetaModelRepurposer(Repurposer):
    """
    Base class for repurposers that extract features from layers in source neural network (Transfer) and train a
    meta-model using the extracted features (Learn).

    :param source_model: Source neural network to do transfer learning from.
    :type source_model: :class:`mxnet.mod.Module`
    :param feature_layer_names: Name of layer(s) in source_model from which features should be transferred.
    :type feature_layer_names: list[str]
    :param context_function: MXNet context function that provides device type context.
    :type context_function: function(int)->:class:`mx.context.Context`
    :param int num_devices: Number of devices to use to extract features from source_model.
    """
    def __init__(self, source_model: mx.mod.Module, feature_layer_names, context_function=mx.context.cpu,
                 num_devices=1):
        super(MetaModelRepurposer, self).__init__(source_model, context_function, num_devices)

        # Create model handler for source model. Required to extract features(layer output) during repurpose and predict
        self.source_model_handler = ModelHandler(self.source_model, self.context_function, self.num_devices)

        self.feature_layer_names = feature_layer_names
        self.provide_data = None
        self.provide_label = None
        self._save_source_model_default = True

    def get_params(self):
        """
        Get parameters of repurposer that are in the constructor's argument list.

        :rtype: dict
        """
        param_dict = super(MetaModelRepurposer, self).get_params()
        param_dict[keys.FEATURE_LAYERS] = self.feature_layer_names
        return param_dict

    @property
    def feature_layer_names(self):
        """
        Names of the layers to extract features from.
        """
        return self._feature_layer_names

    @feature_layer_names.setter
    def feature_layer_names(self, value):
        if not isinstance(value, list):
            raise TypeError("feature_layer_names must be a list. Instead got type: {}".format(type(value)))

        if len(value) == 0:
            raise ValueError(
                "feature_layer_names cannot be empty. At least one layer name is required to extract features")

        # Validate if feature_layer_names passed are found in source_model
        for layer_name in value:
            if layer_name not in self.source_model_handler.layer_names:
                raise ValueError("feature_layer_name \'{}\' is not found in source_model".format(layer_name))

        self._feature_layer_names = value

    @property
    def source_model(self):
        """
        Model to extract features from.
        """
        return self._source_model

    @source_model.setter
    def source_model(self, value):
        if not isinstance(value, mx.mod.Module):
            error = ("source_model must be a valid `mxnet.mod.Module` object because"
                     " MetaModelRepurposer requires source_model to extract features for both repurpose and predict."
                     " Instead got type: {}".format(type(value)))
            raise TypeError(error)

        self._source_model = value

    def repurpose(self, train_iterator: mx.io.DataIter):
        """
        Train a meta-model using features extracted from training data through the source neural network.

        Set self.target_model to the trained meta-model.

        :param train_iterator: Training data iterator to use to extract features from source_model.
        :type train_iterator: :class: `mxnet.io.DataIter`
        """

        # Validate if repurpose call is valid
        self._validate_before_repurpose()

        # For given training data, extract features from source neural network
        train_data = self.get_features_from_source_model(train_iterator)

        # Train target model using the extracted features (transfer and learn)
        self.target_model = self._train_model_from_features(train_data.features, train_data.labels,
                                                            train_data.feature_indices_per_layer)

        # Save data_shape and label_shapes
        self.provide_data = train_iterator.provide_data
        self.provide_label = train_iterator.provide_label

    def predict_probability(self, test_iterator: mx.io.DataIter):
        """
        Predict class probabilities on test data using the target_model i.e. repurposed meta-model.

        :param test_iterator: Test data iterator to return predictions for.
        :type test_iterator: mxnet.io.DataIter
        :return: Predicted probabilities.
        :rtype: :class:`numpy.ndarray`
        """

        # Validate if predict call is valid
        self._validate_before_predict()

        # For given test data, extract features from source neural network
        test_data = self.get_features_from_source_model(test_iterator)

        # Predict and return probabilities
        return self._predict_probability_from_features(test_data.features)

    def predict_label(self, test_iterator: mx.io.DataIter):
        """
        Predict class labels on test data using the target_model i.e. repurposed meta-model.

        :param test_iterator: Test data iterator to return predictions for.
        :type test_iterator: mxnet.io.DataIter
        :return: Predicted labels.
        :rtype: :class:`numpy.ndarray`
        """
        self._validate_before_predict()

        # For given test data, extract features from source neural network
        test_data = self.get_features_from_source_model(test_iterator)

        # Predict and return labels
        return self._predict_label_from_features(test_data.features)

    def get_features_from_source_model(self, data_iterator: mx.io.DataIter):
        """
        Extract feature outputs from feature_layer_names in source_model, merge and return all features and labels.

        In addition, return mapping of feature_layer_name to indices in feature array.

        :param data_iterator: Iterator for data to be passed through the source network and extract features.
        :type data_iterator: :class:`mxnet.io.DataIter`
        :return: features, feature_indices_per_layer and labels.
        :rtype: :class:`MetaModelData`
        """

        feature_dict, labels = self.source_model_handler.get_layer_output(data_iterator, self.feature_layer_names)

        # Merge feature outputs from all layers and save feature indices per layer
        # e.g. output of feature_dict {layer1:[f1,f2,f3], layer2:[f4,f5]} would be
        # [f1,f2,f3,f4,f5] and {layer1:[0,1,2], layer2:[3,4]}
        features = None
        feature_indices_per_layer = OrderedDict()  # mapping for {layer name, feature indices}
        for layer_name in feature_dict:
            if features is None:
                next_feature_index = 0
                features = feature_dict[layer_name]
            else:
                next_feature_index = features.shape[1]  # New feature column(axis=1) will be added at the end
                features = np.hstack([features, feature_dict[layer_name]])
            total_num_features = features.shape[1]
            feature_indices_per_layer[layer_name] = np.arange(next_feature_index, total_num_features)

        meta_model_data = MetaModelData(features, feature_indices_per_layer, labels)
        return meta_model_data

    @abstractmethod
    def _train_model_from_features(self, features, labels, feature_indices_per_layer):
        """
        Abstract method.
        """
        pass

    @abstractmethod
    def _predict_probability_from_features(self, features):
        """
        Abstract method.
        """
        pass

    @abstractmethod
    def _predict_label_from_features(self, features):
        """
        Abstract method.
        """
        pass


class MetaModelData:
    def __init__(self, features, feature_indices_per_layer, labels):
        """
        Structured data used by meta model repurposer to pass information extracted from source model.

        :param features: Features extracted from source neural network
        :type features: :class:`numpy.ndarray`
        :param feature_indices_per_layer: Mapping of feature_layer_names to indices in features array i.e. {layer_name,
            feature_indices}
        :type feature_indices_per_layer: OrderedDict[str, :class:`numpy.ndarray`]
        :param labels: Labels for target dataset.
        :type labels: :class:`numpy.ndarray`
        """
        self.features = features
        self.feature_indices_per_layer = feature_indices_per_layer
        self.labels = labels
