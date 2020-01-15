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
from sklearn.svm import SVC

from .meta_model_repurposer import MetaModelRepurposer
from .constants import repurposer_keys
from .constants import meta_model_repurposer_keys as keys
from . import utils


class SvmRepurposer(MetaModelRepurposer):
    """
    Perform Transfer Learning through a Support Vector Machine (SVM) meta-model which repurposes the source neural
    network.

    :param source_model: Source neural network to do transfer learning from.
    :type source_model: :class:`mxnet.mod.Module`
    :param feature_layer_names: Name of layer(s) in source_model from which features should be transferred.
    :type feature_layer_names: list[str]
    :param context_function: MXNet context function that provides device type context.
    :type context_function: function(int)->:class:`mx.context.Context`
    :param int num_devices: Number of devices to use to extract features from source_model.
    :param float c: Penalty parameter C of the error term.
    :param string kernel: Specifies the kernel type to be used in the SVM algorithm in sklearn library. It must
                          be one of 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed' or a callable.
    :param float gamma: Kernel coefficient for 'rbf', 'poly' and 'sigmoid'. If gamma is 'auto' then 1/n_features will
                        be used instead.
    :param bool enable_probability_estimates: Whether to enable probability estimates.
                                              This must be enabled for predict_probability to work and will slow down
                                              training.
    """
    def __init__(self, source_model: mx.mod.Module, feature_layer_names, context_function=mx.cpu, num_devices=1,
                 c=1.0, kernel='linear', gamma='auto', enable_probability_estimates=False):
        # Call base class constructor with parameters required for meta-models
        super(SvmRepurposer, self).__init__(source_model, feature_layer_names, context_function, num_devices)

        # Initialize SVM specific parameters
        self.c = c
        self.kernel = kernel
        self.gamma = gamma
        self.enable_probability_estimates = enable_probability_estimates

    def get_params(self):
        """
        Get parameters of repurposer that are in the constructor.

        :rtype: dict
        """
        param_dict = super(SvmRepurposer, self).get_params()
        param_dict[keys.C] = self.c
        param_dict[keys.KERNEL] = self.kernel
        param_dict[keys.GAMMA] = self.gamma
        param_dict[keys.PROB_ESTIMATES] = self.enable_probability_estimates
        return param_dict

    def _train_model_from_features(self, features, labels, feature_indices_per_layer=None):
        """
        Train an SVM model using features extracted from source neural network.

        :param features: Features extracted from source neural network.
        :type features: :class:`numpy.ndarray`
        :param labels: Labels to use for training.
        :type labels: :class:`numpy.ndarray`
        :param feature_indices_per_layer: Mapping of feature_layer_names to indices in features array
                                          i.e. {layer_name, feature_indices} Note that this param is currently not
                                          consumed by svm_repurposer.
        :type feature_indices_per_layer: OrderedDict[str, :class:`numpy.ndarray`]
        :return: SVM model trained with given features and labels using sci-kit learn library.
        :rtype: :class: `sklearn.svm.SVC`
        """

        svm_classifier = SVC(C=self.c,
                             kernel=self.kernel,
                             gamma=self.gamma,
                             decision_function_shape='ovr',
                             random_state=1,
                             probability=self.enable_probability_estimates)
        svm_classifier.fit(features, labels)
        return svm_classifier

    def _predict_probability_from_features(self, features):
        """
        Run predictions with target_model on features extracted from source neural network.
        Use sklearn's SVM predict_proba method and return predicted probabilities.

        :param features: Features extracted from source neural network.
        :type features: :class:`numpy.ndarray`
        :return: Predicted probabilities.
        :rtype: :class:`numpy.ndarray`
        """
        if not self.target_model.probability:
            raise ValueError("Probability estimates should have been enabled during model training for this method to \
                             work")
        return self.target_model.predict_proba(features)

    def _predict_label_from_features(self, features):
        """
        Run predictions with target_model on features extracted from source neural network.
        Use sklearn's SVM predict method and return predicted labels.

        :param features: Features extracted from source neural network.
        :type features: :class:`numpy.ndarray`
        :return: Predicted labels.
        :rtype: :class:`numpy.ndarray`
        """
        return self.target_model.predict(features)

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
        self.target_model = utils.sklearn_model_from_dict(SVC, input_dict[repurposer_keys.TARGET_MODEL])
