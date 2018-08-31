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
from sklearn.linear_model import LogisticRegression

from .meta_model_repurposer import MetaModelRepurposer
from .constants import repurposer_keys
from .constants import meta_model_repurposer_keys as keys
from . import utils


class LrRepurposer(MetaModelRepurposer):
    """
    Perform Transfer Learning through a Logistic Regression meta-model which repurposes the source neural network.

    :param source_model: Source neural network to do Transfer Learning from.
    :type source_model: :class:`mxnet.mod.Module`
    :param feature_layer_names: Name of layer(s) in source_model from which features should be transferred.
    :type feature_layer_names: list[str]
    :param context_function: MXnet context function that provides device type context.
    :type context_function: function(int)->:class:`mx.context.Context`
    :param int num_devices: Number of devices to use to extract features from source_model.
    :param float tol: Tolerance for stopping criteria.
    :param float c: Inverse of regularization strength; must be a positive float.
    :param int n_jobs: Number of CPU cores used during the cross-validation loop. If given a value of -1, all cores
        are used.
    """
    def __init__(self, source_model: mx.mod.Module, feature_layer_names, context_function=mx.context.cpu, num_devices=1,
                 tol=1e-4, c=1.0, n_jobs=-1):
        # Call base class constructor with parameters required for meta-models
        super(LrRepurposer, self).__init__(source_model, feature_layer_names, context_function, num_devices)

        # Initialize LR specific parameters
        self.tol = tol
        self.c = c
        self.n_jobs = n_jobs

    def get_params(self):
        """
        Get parameters of repurposer that are in the constructor.

        :rtype: dict
        """
        param_dict = super(LrRepurposer, self).get_params()
        param_dict[keys.TOL] = self.tol
        param_dict[keys.C] = self.c
        param_dict[keys.N_JOBS] = self.n_jobs
        return param_dict

    def _train_model_from_features(self, features, labels, feature_indices_per_layer=None):
        """
        Train a Logistic Regression model using features extracted from source neural network.

        :param features: Features extracted from source neural network.
        :type features: :class:`numpy.ndarray`
        :param labels: Labels to use for training.
        :type labels: :class:`numpy.ndarray`
        :param feature_indices_per_layer: Mapping of feature_layer_names to indices in features array.
        i.e. {layer_name, feature_indices}. Note that this param is currently not consumed by lr_repurposer.
        :type feature_indices_per_layer: OrderedDict[str, :class:`numpy.ndarray`]
        :return: LR model trained with given features and labels using sci-kit learn library.
        :rtype: :class: `sklearn.linear_model.LogisticRegression`
        """
        lr_model = LogisticRegression(penalty='l2',
                                      tol=self.tol,
                                      C=self.c,
                                      fit_intercept=False,
                                      class_weight='balanced',
                                      random_state=1,
                                      solver='sag',
                                      multi_class='multinomial',
                                      n_jobs=self.n_jobs)
        lr_model.fit(features, labels)
        return lr_model

    def _predict_probability_from_features(self, features):
        """
        Run predictions with target_model on features extracted from source neural network.
        Use sklearn's LR predict_proba method and return predicted probabilities.

        :param features: Features extracted from source neural network.
        :type features: :class:`numpy.ndarray`
        :return: Predicted probabilities.
        :rtype: :class:`numpy.ndarray`
        """
        return LogisticRegression.predict_proba(self.target_model, features)

    def _predict_label_from_features(self, features):
        """
        Run predictions with target_model on features extracted from source neural network.
        Use sklearn's LR predict method and return predicted labels.

        :param features: Features extracted from source neural network.
        :type features: :class:`numpy.ndarray`
        :return: Predicted labels.
        :rtype: :class:`numpy.ndarray`
        """
        return LogisticRegression.predict(self.target_model, features)

    def serialize(self, file_prefix):
        """
        Saves repurposer (excluding source model) to file_prefix.json.

        :param str file_prefix: Prefix to save file with.
        """
        output_dict = {}
        output_dict[repurposer_keys.PARAMS] = self.get_params()
        output_dict[repurposer_keys.TARGET_MODEL] = utils.sklearn_model_to_dict(self.target_model)
        output_dict.update(self._get_attributes())

        utils.save_json(file_prefix, output_dict)

    def deserialize(self, input_dict):
        """
        Uses dictionary to set attributes of repurposer.

        :param dict input_dict: Dictionary containing values for attributes to be set to.
        """
        self._set_attributes(input_dict)  # Set attributes of the repurposer from input_dict
        self.target_model = utils.sklearn_model_from_dict(LogisticRegression, input_dict[repurposer_keys.TARGET_MODEL])
