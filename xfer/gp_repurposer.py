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
from sklearn.preprocessing import LabelBinarizer, Normalizer

from .meta_model_repurposer import MetaModelRepurposer
from .constants import gp_repurposer_keys as keys
from .constants import repurposer_keys
from . import utils


class GpRepurposer(MetaModelRepurposer):
    """
    Repurpose source neural network to create a Gaussian Process (GP) meta-model through Transfer Learning.

    :param source_model: Source neural network to do transfer learning from.
    :type source_model: :class:`mxnet.mod.Module`
    :param feature_layer_names: Name of layer(s) in source_model from which features should be transferred.
    :type feature_layer_names: list[str]
    :param context_function: MXNet context function that provides device type context.
    :type context_function: function(int)->:class:`mx.context.Context`
    :param int num_devices: Number of devices to use to extract features from source_model.
    :param int max_function_evaluations: Maximum number of function evaluations to perform in GP optimization.
    :param bool apply_l2_norm: Whether to apply L2 normalization after extracting features from source neural network.
                               If set to True, L2 normalization will be applied to features before passing to GP during
                               training and prediction.
    """
    def __init__(self, source_model: mx.mod.Module, feature_layer_names, context_function=mx.context.cpu, num_devices=1,
                 max_function_evaluations=100, apply_l2_norm=False):
        # Call base class constructor with parameters required for meta-models
        super().__init__(source_model, feature_layer_names, context_function, num_devices)
        self.max_function_evaluations = max_function_evaluations
        self.apply_l2_norm = apply_l2_norm

        # Mean of features to use for normalization. Computed in training phase.
        # Used to normalize features in training and in prediction.
        self.feature_mean = None

        # Optimizer to use for training GP model
        self.optimizer = 'lbfgs'

        # Number of inducing points to use for sparse GP
        self.NUM_INDUCING_SPARSE_GP = 100

        # Normalizer to use when apply_l2_norm flag is set
        self.l2_normalizer = Normalizer(norm='l2')

    def _train_model_from_features(self, features, labels, feature_indices_per_layer):
        """
        Train GP classification models using features extracted from the source neural network.

        :param features: Features extracted from source neural network.
        :type features: :class:`numpy.ndarray`
        :param labels: Labels to use for training.
        :type labels: :class:`numpy.ndarray`
        :param feature_indices_per_layer: Mapping of feature_layer_names to dimension indices in features array
                                          i.e. {layer_name, feature_indices}.
                                          Used to build separate kernels for features from different layers.
        :type feature_indices_per_layer: OrderedDict[str, :class:`numpy.ndarray`]
        :return: List of GP classification models trained for each class (one-vs-all) in given training data.
                 List of Sparse GP models returned if number of training instances is greater than
                 NUM_INDUCING_SPARSE_GP.
                 If there are only two classes in training data, then the output list contains a single model.
        :rtype: list[:class:`GPy.models.GPClassification`] or list[:class:`GPy.models.SparseGPClassification`]
        """
        # Normalize features to train on
        self.feature_mean = features.mean(axis=0)  # Compute mean for each feature across all training instances
        normalized_features = features - self.feature_mean  # Normalize features to have zero mean
        if self.apply_l2_norm:  # Apply L2 normalization if flag is set
            normalized_features = self.l2_normalizer.fit_transform(normalized_features)

        # Binarize labels in a one-vs-all fashion to train a separate model for each class
        # Output contains 'c' columns (c=number of classes) and each column contains binary labels w.r.t to that class.
        # If number of classes is two, then the output contains a single column with values 0 and 1
        binarized_labels = LabelBinarizer().fit_transform(labels)

        # Build kernel using given feature indices
        kernel = self._build_kernel(feature_indices_per_layer)

        # Do spare GP if number of training instances is greater than chosen number of inducing points
        # Otherwise, do normal GP classification because the data set is small
        num_training_instances = features.shape[0]
        do_sparse_gp = (num_training_instances > self.NUM_INDUCING_SPARSE_GP)

        # Train a GP model for each class (one-vs-all) if there are more than two classes, and one model if there
        # are only two classes
        num_models = binarized_labels.shape[1]
        models = [None] * num_models
        for model_index in range(num_models):
            binary_labels_for_current_class = binarized_labels[:, model_index:model_index+1]
            input_kernel = kernel.copy()  # Pass copy of kernel to avoid original kernel being updated by GPy
            models[model_index] = self._train_model_for_binary_label(binary_label=binary_labels_for_current_class,
                                                                     features=normalized_features,
                                                                     kernel=input_kernel,
                                                                     do_sparse_gp=do_sparse_gp)
        return models

    def _train_model_for_binary_label(self, features, binary_label, kernel, do_sparse_gp):
        # GPy is imported here in order to avoid importing it during 'import xfer'
        import GPy
        # Train a GPy model for binary classification with given features and kernel
        if do_sparse_gp:
            model = GPy.models.SparseGPClassification(X=features, Y=binary_label, kernel=kernel,
                                                      num_inducing=self.NUM_INDUCING_SPARSE_GP)
        else:
            model = GPy.models.GPClassification(X=features, Y=binary_label, kernel=kernel)
        model.optimize(optimizer=self.optimizer, max_iters=self.max_function_evaluations)
        return model

    @staticmethod
    def _build_kernel(feature_indices_per_layer):
        """
        Build separate RBF kernels for features from different layers and return the kernel that results from adding all
        feature specific kernels.

        :param feature_indices_per_layer: Mapping of feature_layer_names to dimension indices in features array
                                          i.e. {layer_name, feature_indices}.
        :type feature_indices_per_layer: dict[str, :class:`numpy.ndarray`]
        :return: GPy RBF kernel if all features are from single layer or GPy Add kernel if features are from multiple
                 layers.
        :rtype: :class:`GPy.kern.RBF` or :class:`GPy.kern.Add`
        """
        # GPy is imported here in order to avoid importing it during 'import xfer'
        import GPy
        all_kernels = None
        for layer_name in feature_indices_per_layer:
            active_dims = feature_indices_per_layer[layer_name]  # feature indices corresponding to current layer
            kernel = GPy.kern.RBF(input_dim=active_dims.size, name=layer_name, active_dims=active_dims.tolist())
            if all_kernels is None:
                all_kernels = kernel
            else:
                all_kernels += kernel
        return all_kernels

    def _predict_probability_from_features(self, features):
        """
        Compute predictions using self.target_model with features extracted from source neural network.
        self.target_model is a list of GP classification models trained for each class in a one-vs-all fashion.
        Use GPy's predict method on each model and compute probabilities predicted for individual classes.
        The individual class probabilities are then normalized such that their sum is 1.

        :param features: Features extracted from source neural network.
        :type features: :class:`numpy.ndarray`
        :return: Normalized one-vs-all probabilities.
        :rtype: :class:`numpy.ndarray`
        """
        normalized_features = features - self.feature_mean  # Normalize features using the training features' means
        if self.apply_l2_norm:  # Apply L2 normalization if flag is set
            normalized_features = self.l2_normalizer.transform(normalized_features)

        num_gp_models = len(self.target_model)
        if num_gp_models == 1:
            # When there are only two classes, get probability for class_1 (P) and calculate probability for class_0
            # (1-P)
            prediction, _ = self.target_model[0].predict(normalized_features)
            normalized_predictions = np.hstack([1.0-prediction, prediction])

        else:
            # When there are more than two classes, get one-vs-all prediction scores for each class
            # from binary GP models
            predictions_per_class = []
            for model_id in range(num_gp_models):
                binary_gp_model = self.target_model[model_id]
                binary_prediction, _ = binary_gp_model.predict(normalized_features)
                predictions_per_class.append(binary_prediction)

            # Convert scores list to numpy array
            predictions = np.nan_to_num(np.hstack(predictions_per_class))

            # Normalize individual predictions to sum up to 1
            sum_of_predictions_per_instance = np.sum(predictions, axis=1).reshape(predictions.shape[0], 1)
            normalized_predictions = predictions / sum_of_predictions_per_instance

        return normalized_predictions

    def _predict_label_from_features(self, features):
        """
        Return labels predicted using target_model with features extracted from source neural network.
        target_model is a list of GP classification models trained for each class in a one-vs-all fashion.
        Using GPy's predict method, compute one-vs-all probabilities per class and return the class with
        maximum probability.

        :param features: Features extracted from source neural network.
        :type features: :class:`numpy.ndarray`
        :return: Predicted labels i.e. Class with maximum probability for each test instance.
        :rtype: :class:`numpy.ndarray`
        """
        predictions = self._predict_probability_from_features(features)
        labels = np.argmax(predictions, axis=1)  # Select label with maximum prediction for each test instance
        return labels

    def get_params(self):
        """
        Get parameters of repurposer that are in the constructor's argument list.

        :rtype: dict
        """
        params_dict = super().get_params()
        params_dict[keys.MAX_FUNCTION_EVALUATIONS] = self.max_function_evaluations
        params_dict[keys.APPLY_L2_NORM] = self.apply_l2_norm
        return params_dict

    def _get_attributes(self):
        """
        Get parameters of repurposer not in constructor's argument list.

        :rtype: dict
        """
        attributes_dict = super()._get_attributes()
        attributes_dict[keys.OPTIMIZER] = self.optimizer
        attributes_dict[keys.NUM_INDUCING_SPARSE_GP] = self.NUM_INDUCING_SPARSE_GP
        attributes_dict[keys.FEATURE_MEAN] = self.feature_mean.tolist()
        return attributes_dict

    def _set_attributes(self, input_dict):
        """
        Set attributes of class from input_dict.
        These attributes are the same as those returned by get_attributes method.

        :param input_dict: Dictionary containing attribute values.
        :return: None
        """
        super()._set_attributes(input_dict)
        self.optimizer = input_dict[keys.OPTIMIZER]
        self.NUM_INDUCING_SPARSE_GP = input_dict[keys.NUM_INDUCING_SPARSE_GP]
        self.feature_mean = np.array(input_dict[keys.FEATURE_MEAN])

    def serialize(self, file_prefix):
        """
        Serialize the GP repurposer (model and supporting info) and save to file.

        :param str file_prefix: Prefix of file path to save the serialized repurposer to.
        :return: None
        """
        # Get constructor params. This will be used to recreate repurposer object in deserialization flow.
        output_dict = {repurposer_keys.PARAMS: self.get_params()}

        # Get rest of the attributes to save with the repurposer.
        output_dict.update(self._get_attributes())

        # Serialize the GP models and save to output dictionary
        output_dict[repurposer_keys.TARGET_MODEL] = self._serialize_target_gp_models(save_data=True)

        # Save the serialized params and attributes to file
        utils.save_json(file_prefix, output_dict)

    def deserialize(self, input_dict):
        """
        Uses dictionary to set attributes of repurposer.

        :param dict input_dict: Dictionary containing values for attributes to be set to.
        :return: None
        """
        # Set attributes of the repurposer from input_dict
        self._set_attributes(input_dict)

        # Deserialize and set the target GP models
        self.target_model = self._deserialize_target_gp_models(input_dict[repurposer_keys.TARGET_MODEL])

    def _serialize_target_gp_models(self, save_data=True):
        # Serialize the gp models trained per class and add to output list
        serialized_models = []
        for model in self.target_model:
            serialized_models.append(model.to_dict(save_data=save_data))
        return serialized_models

    @staticmethod
    def _deserialize_target_gp_models(serialized_target_model):
        # GPy is imported here in order to avoid importing it during 'import xfer'
        import GPy
        # Deserialize the GP models trained per class and return
        deserialized_models = []
        for model_dict in serialized_target_model:
            deserialized_models.append(GPy.core.Model.from_dict(model_dict))
        return deserialized_models
