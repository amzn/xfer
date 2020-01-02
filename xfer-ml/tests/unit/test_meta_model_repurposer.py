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
from unittest import TestCase
from unittest.mock import Mock, patch

from collections import OrderedDict
import numpy as np
from abc import abstractmethod
import pickle
import mxnet as mx
import os

from xfer import MetaModelRepurposer, SvmRepurposer, BnnRepurposer, GpRepurposer, load
from ..repurposer_test_utils import RepurposerTestUtils


@patch(RepurposerTestUtils.META_MODEL_REPURPOSER_MODEL_HANDLER_CLASS)
class MetaModelRepurposerTestCase(TestCase):
    _test_data_dir = 'tests/data/meta_model_repurposer_data/'

    def setUp(self):
        self.repurposer_class = MetaModelRepurposer
        self.source_model = RepurposerTestUtils.create_mxnet_module()
        self.source_model_layer_1 = RepurposerTestUtils.LAYER_FC1
        self.source_model_layer_2 = RepurposerTestUtils.LAYER_FC2
        self.source_model_layers = [self.source_model_layer_1, self.source_model_layer_2]

        # Load data (features and labels) to run tests
        self.features = np.loadtxt(self._test_data_dir + '_all_features.out')
        self.labels = np.loadtxt(self._test_data_dir + '_labels.out')
        self.n_classes = len(np.unique(self.labels))
        self._train_indices = np.loadtxt(self._test_data_dir + '_train_indices.out').astype(int)
        self._test_indices = np.loadtxt(self._test_data_dir + '_test_indices.out').astype(int)
        self.n_test_instances = len(self._test_indices)
        self.train_features = self.features[self._train_indices]
        self.train_labels = self.labels[self._train_indices]
        self.test_features = self.features[self._test_indices]
        self.test_labels = self.labels[self._test_indices]
        self.train_feature_dict = {'layer1': self.train_features}
        self.test_feature_dict = {'layer1': self.test_features}
        self.mock_object = Mock()

        # Overridden in derived classes
        self.target_model_path = None
        self.expected_accuracy = None
        self.minimum_expected_accuracy = None

    def test_instantiation_valid_input(self, mock_model_handler):
        mock_model_handler.return_value = RepurposerTestUtils.get_mock_model_handler_object()
        feature_layer_names_in_source_model = [self.source_model_layer_1]
        repurposer = self.repurposer_class(self.source_model, feature_layer_names_in_source_model)
        self.assertTrue(repurposer.source_model == self.source_model)
        self.assertTrue(repurposer.feature_layer_names == feature_layer_names_in_source_model)

    def test_instantiation_source_model_is_none(self, mock_model_handler):
        source_model_none = None
        mock_feature_layer_names = Mock()
        self.assertRaisesRegex(TypeError, "source_model must be a valid `mxnet.mod.Module` object",
                               self.repurposer_class, source_model_none, mock_feature_layer_names)

    def test_instantiation_feature_layer_names_empty(self, mock_model_handler):
        # Empty feature_layer_names list raises ValueError
        feature_layer_names_empty = []
        self.assertRaisesRegex(ValueError, "feature_layer_names cannot be empty",
                               self.repurposer_class, self.source_model, feature_layer_names_empty)

    def test_instantiation_feature_layer_names_invalid_type(self, mock_model_handler):
        # feature_layer_names not being a list raises TypeError
        feature_layer_names_int = 1
        self.assertRaisesRegex(TypeError, "feature_layer_names must be a list",
                               self.repurposer_class, self.source_model, feature_layer_names_int)

        feature_layer_names_str = ''
        self.assertRaisesRegex(TypeError, "feature_layer_names must be a list",
                               self.repurposer_class, self.source_model, feature_layer_names_str)

        feature_layer_names_dict = {'': ''}
        self.assertRaisesRegex(TypeError, "feature_layer_names must be a list",
                               self.repurposer_class, self.source_model, feature_layer_names_dict)

    def test_instantiation_feature_layer_names_not_in_source_model(self, mock_model_handler):
        mock_model_handler.return_value = RepurposerTestUtils.get_mock_model_handler_object()

        # Some feature_layer_names not found in source_model
        feature_layer_names_some_not_in_source_model = [self.source_model_layer_1, 'phantom_layer_2']
        self.assertRaisesRegex(ValueError, "feature_layer_name 'phantom_layer_2' is not found in source_model",
                               self.repurposer_class, self.source_model, feature_layer_names_some_not_in_source_model)

        # All feature_layer_names not found in source_model
        feature_layer_names_all_not_in_source_model = ['phantom_layer_1', 'phantom_layer_2']
        self.assertRaisesRegex(ValueError, "feature_layer_name 'phantom_layer_1' is not found in source_model",
                               self.repurposer_class, self.source_model, feature_layer_names_all_not_in_source_model)

    def test_validate_before_predict(self, mock_model_handler):
        mock_model_handler.return_value = RepurposerTestUtils.get_mock_model_handler_object()

        # Create meta model repurposer object
        repurposer = self.repurposer_class(self.source_model, self.source_model_layers)

        # Target model is neither created through repurpose nor explicitly assigned
        # So, calling predict should raise ValueError
        self.assertRaisesRegex(ValueError, "Cannot predict because target_model is not initialized",
                               repurposer._validate_before_predict)

        # Test valid input
        repurposer.target_model = RepurposerTestUtils.create_mxnet_module()
        repurposer._validate_before_predict()

    def test_get_features_from_source_model(self, mock_model_handler):
        mock_model_handler.return_value = RepurposerTestUtils.get_mock_model_handler_object()

        # Dummy layer outputs to test with
        layer1_output = np.array([[1.1, 1.2, 1.3], [1.4, 1.5, 1.6], [1.7, 1.8, 1.9]])
        layer2_output = np.array([[2.1, 2.2], [2.3, 2.4], [2.5, 2.6]])

        # Test with 1 feature layer
        feature_dict = OrderedDict()
        feature_dict['layer1'] = layer1_output
        expected_feature_indices = OrderedDict()
        expected_feature_indices['layer1'] = np.array([0, 1, 2])
        expected_features = layer1_output
        self._test_get_features_from_source_model(mock_model_handler, feature_dict, expected_features,
                                                  expected_feature_indices)

        # Test with 2 feature layers: layer1, layer2
        feature_dict = OrderedDict()
        feature_dict['layer1'] = layer1_output
        feature_dict['layer2'] = layer2_output
        expected_feature_indices = OrderedDict()
        expected_feature_indices['layer1'] = np.array([0, 1, 2])
        expected_feature_indices['layer2'] = np.array([3, 4])
        expected_features = np.array([[1.1, 1.2, 1.3, 2.1, 2.2], [1.4, 1.5, 1.6, 2.3, 2.4], [1.7, 1.8, 1.9, 2.5, 2.6]])
        self._test_get_features_from_source_model(mock_model_handler, feature_dict, expected_features,
                                                  expected_feature_indices)

        # Test with 2 feature layers: layer2, layer1
        feature_dict = OrderedDict()
        feature_dict['layer2'] = layer2_output
        feature_dict['layer1'] = layer1_output
        expected_feature_indices = OrderedDict()
        expected_feature_indices['layer2'] = np.array([0, 1])
        expected_feature_indices['layer1'] = np.array([2, 3, 4])
        expected_features = np.array([[2.1, 2.2, 1.1, 1.2, 1.3], [2.3, 2.4, 1.4, 1.5, 1.6], [2.5, 2.6, 1.7, 1.8, 1.9]])
        self._test_get_features_from_source_model(mock_model_handler, feature_dict, expected_features,
                                                  expected_feature_indices)

    def _test_get_features_from_source_model(self, mock_model_handler, feature_dict, expected_features,
                                             expected_feature_indices):
        repurposer = self.repurposer_class(self.source_model, self.source_model_layers)
        labels_from_model_handler = np.array([0, 1, 0])
        mock_model_handler.return_value.get_layer_output.return_value = feature_dict, labels_from_model_handler
        meta_model_data = repurposer.get_features_from_source_model(data_iterator=Mock())
        self.assertTrue(np.array_equal(meta_model_data.features, expected_features))
        self.assertTrue(np.array_equal(meta_model_data.labels, labels_from_model_handler))
        RepurposerTestUtils.assert_feature_indices_equal(expected_feature_indices,
                                                         meta_model_data.feature_indices_per_layer)

    def test_serialisation(self, mock_model_handler):
        if self.repurposer_class == MetaModelRepurposer:  # base class
            return
        self._test_save_load_repurposed_model(mock_model_handler, save_source_model=True)
        self._test_save_load_repurposed_model(mock_model_handler, save_source_model=False)

    def _test_save_load_repurposed_model(self, mock_model_handler, save_source_model):
        # To speed-up unit test running time. Accuracy is validated in integration tests.
        num_train_points = 2
        self.train_features = self.train_features[:num_train_points]
        self.train_labels = self.train_labels[:num_train_points]

        mock_model_handler.return_value = RepurposerTestUtils.get_mock_model_handler_object()
        file_path = 'test_serialisation'
        RepurposerTestUtils._remove_files_with_prefix(file_path)
        source_model = mx.module.Module.load('tests/data/testnetv1', 0, label_names=['softmaxoutput1_label'],
                                             data_names=('data',))
        repurposer = self.repurposer_class(source_model, self.source_model_layers)
        if self.repurposer_class == BnnRepurposer:
            repurposer = BnnRepurposer(source_model, self.source_model_layers, num_epochs=1,
                                       num_samples_mc_prediction=15)
        repurposer.target_model = repurposer._train_model_from_features(self.train_features, self.train_labels)
        # Manually setting provide_data and provide_label because repurpose() is not called
        repurposer.provide_data = [('data', (2, 3, 224, 224))]
        repurposer.provide_label = [('softmaxoutput1_label', (2,))]
        # Mocking iterator because get_layer_output is patched
        mock_model_handler.return_value.get_layer_output.return_value = self.test_feature_dict, self.test_labels
        results = repurposer.predict_label(test_iterator=self.mock_object)
        assert not os.path.isfile(file_path + '.json')

        if save_source_model:
            assert not os.path.isfile(file_path + '_source-symbol.json')
            assert not os.path.isfile(file_path + '_source-0000.params')
            repurposer.save_repurposer(model_name=file_path, save_source_model=save_source_model)
            assert os.path.isfile(file_path + '_source-symbol.json')
            assert os.path.isfile(file_path + '_source-0000.params')
            loaded_repurposer = load(file_path)
        else:
            repurposer.save_repurposer(model_name=file_path, save_source_model=save_source_model)
            loaded_repurposer = load(file_path, source_model=repurposer.source_model)

        assert os.path.isfile(file_path + '.json')
        RepurposerTestUtils._remove_files_with_prefix(file_path)
        results_loaded = loaded_repurposer.predict_label(test_iterator=self.mock_object)
        assert type(repurposer) == type(loaded_repurposer)
        self._assert_target_model_equal(repurposer.target_model, loaded_repurposer.target_model)
        accuracy1 = np.mean(results == self.test_labels)
        accuracy2 = np.mean(results_loaded == self.test_labels)

        if self.repurposer_class == BnnRepurposer:
            assert np.isclose(accuracy1, accuracy2, atol=0.1), 'Inconsistent accuracies: {}, {}.'.format(accuracy1,
                                                                                                         accuracy2)
        else:
            assert accuracy1 == accuracy2, 'Inconsistent accuracies: {}, {}.'.format(accuracy1, accuracy2)

        self._assert_attributes_equal(repurposer, loaded_repurposer)

    def _assert_target_model_equal(self, model1, model2):
        assert model1.__dict__.keys() == model2.__dict__.keys()
        for key in model1.__dict__.keys():
            if type(model1.__dict__[key]) == np.ndarray:
                assert isinstance(model2.__dict__[key], np.ndarray)
                assert np.array_equal(model1.__dict__[key], model2.__dict__[key])
            elif type(model1.__dict__[key]) == tuple:
                assert isinstance(model2.__dict__[key], tuple)
                assert list(model1.__dict__[key]) == list(model2.__dict__[key])
            else:
                assert model1.__dict__[key] == model2.__dict__[key]

    def _test_predict(self, mock_model_handler, mock_validate_method, test_predict_probability, expected_accuracy):
        """ Test for predict wrapper in meta model base class """
        # Patch model_handler and then create repurposer object
        mock_model_handler.return_value = RepurposerTestUtils.get_mock_model_handler_object()

        # Create repurposer
        if self.repurposer_class == SvmRepurposer:
            repurposer = self.repurposer_class(self.source_model, self.source_model_layers,
                                               enable_probability_estimates=test_predict_probability)
        elif self.repurposer_class == GpRepurposer:
            repurposer = self.repurposer_class(self.source_model, self.source_model_layers, apply_l2_norm=True)
        else:
            repurposer = self.repurposer_class(self.source_model, self.source_model_layers)

        # Identify which predict function to test
        if test_predict_probability:
            predict_function = repurposer.predict_probability
        else:
            predict_function = repurposer.predict_label

        # Train or Load target model from file
        if self.repurposer_class == GpRepurposer:
            num_datapoints_train = 10
            mock_model_handler.return_value.get_layer_output.return_value =\
                {'l1': self.train_features[:num_datapoints_train]}, self.train_labels[:num_datapoints_train]
            repurposer.repurpose(self.mock_object)
        else:
            with open(self.target_model_path, 'rb') as target_model:
                repurposer.target_model = pickle.load(target_model)

        # Call predict method and get prediction results
        mock_model_handler.return_value.get_layer_output.return_value = self.test_feature_dict, self.test_labels
        mock_validate_method.reset_mock()
        results = predict_function(
            test_iterator=self.mock_object)  # Mocking iterator because get_layer_output is patched

        # Check if predict called validate
        self.assertTrue(mock_validate_method.call_count == 1,
                        "Predict expected to called {} once. Found {} calls".
                        format(RepurposerTestUtils.VALIDATE_PREDICT_METHOD_NAME, mock_validate_method.call_count))

        self._validate_prediction_results(results, test_predict_probability, expected_accuracy)

    def _test_predict_from_features(self, test_predict_probability, expected_accuracy):
        """ Used to test 'predict_from_features' implementation in derived classes """
        # Create repurposer
        if self.repurposer_class == SvmRepurposer:
            repurposer = self.repurposer_class(self.source_model, self.source_model_layers,
                                               enable_probability_estimates=test_predict_probability)
        else:
            repurposer = self.repurposer_class(self.source_model, self.source_model_layers)

        # Load target model from file
        with open(self.target_model_path, 'rb') as target_model:
            repurposer.target_model = pickle.load(target_model)

        if test_predict_probability:
            results = repurposer._predict_probability_from_features(self.test_features)
        else:
            results = repurposer._predict_label_from_features(self.test_features)

        self._validate_prediction_results(results, test_predict_probability, expected_accuracy)

    def _validate_prediction_results(self, results, test_predict_probability, expected_accuracy, num_predictions=None):
        if num_predictions is None:
            test_labels = self.test_labels
            n_test_instances = self.n_test_instances
        else:
            assert num_predictions < len(self.test_labels), 'More predictions ({}), than labels ({})'.format(
                                                                                num_predictions, len(self.test_labels))
            test_labels = self.test_labels[:num_predictions]
            n_test_instances = len(test_labels)

        # Validate type of prediction results
        self.assertTrue(type(results) == np.ndarray,
                        "Prediction results expected to be numpy array. Instead got: {}".format(type(results)))

        # Validate shape of prediction results
        expected_shape = (n_test_instances, self.n_classes) if test_predict_probability else (
            n_test_instances,)
        self.assertTrue(results.shape == expected_shape,
                        "Prediction results shape is incorrect. Expected: {}. Got: {}".format(expected_shape,
                                                                                              results.shape))

        # Validate if prediction probabilities sum to 1
        if test_predict_probability:
            probability_sum = np.sum(results, axis=1)
            array_of_ones = np.ones(shape=(n_test_instances,))
            self.assertTrue(np.allclose(probability_sum, array_of_ones), "Sum of predicted probabilities is not 1")

        # Validate accuracy of prediction results
        labels = np.argmax(results, axis=1) if test_predict_probability else results
        accuracy = np.mean(labels == test_labels)
        self.assertTrue(np.isclose(accuracy, expected_accuracy),
                        "Prediction accuracy is incorrect. Expected: {}. Actual: {}".format(expected_accuracy,
                                                                                            accuracy))

    def _run_common_repurposer_tests(self, repurposer):
        # Target model is not initialized yet
        self.assertTrue(repurposer.target_model is None, "Target model not expected to be initialized at this point")

        # Call repurpose
        repurposer.repurpose(self.mock_object)

        # Validate target model is now set
        self.assertTrue(repurposer.target_model is not None, "Repurpose failed to set target model")

        # Validate trained model
        self._validate_trained_model(repurposer.target_model)

    def _test_repurpose_calls_validate(self, mock_model_handler, mock_validate_method):
        mock_model_handler.return_value = RepurposerTestUtils.get_mock_model_handler_object()
        # Use subset of train_feature_dict and train_labels to speed up test
        N = 2
        train_feature_dict = {'layer1': self.train_feature_dict['layer1'][:N]}
        train_labels = self.train_labels[:N]

        mock_model_handler.return_value.get_layer_output.return_value = train_feature_dict, train_labels
        repurposer = self.repurposer_class(self.source_model, self.source_model_layers)

        mock_validate_method.reset_mock()
        repurposer.repurpose(self.mock_object)
        self.assertTrue(mock_validate_method.call_count == 1,
                        "Repurpose expected to called {} once. Found {} calls".
                        format(RepurposerTestUtils.VALIDATE_REPURPOSE_METHOD_NAME, mock_validate_method.call_count))

    def _assert_attributes_equal(self, repurposer1, repurposer2):
        RepurposerTestUtils._assert_common_attributes_equal(repurposer1, repurposer2)

    @abstractmethod
    def _validate_trained_model(self, target_model):
        pass
