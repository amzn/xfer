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
from unittest.mock import patch
import numpy as np
import os

from xfer import load, GpRepurposer
from .test_meta_model_repurposer import MetaModelRepurposerTestCase
from ..repurposer_test_utils import RepurposerTestUtils

import GPy  # Must be imported after xfer to avoid matplotlib backend errors because backend is set in xfer init


class GpRepurposerTestCase(MetaModelRepurposerTestCase):

    def setUp(self):
        super().setUp()
        self.repurposer_class = GpRepurposer
        self.expected_accuracy = 0.45714285714285713
        self.expected_accuracy_from_features = 0.7
        self.train_feature_indices = np.arange(0, self.train_features.shape[1])
        self.feature_mean = self.train_features.mean(axis=0)
        self.num_data_points_to_predict = 10
        self.num_data_points_to_train = 10

    def test_train_model_from_features(self):
        self._test_train_model_from_features(sparse_gp=True, multiple_kernels=True)
        self._test_train_model_from_features(sparse_gp=True, multiple_kernels=False)
        self._test_train_model_from_features(sparse_gp=False, multiple_kernels=True)
        self._test_train_model_from_features(sparse_gp=False, multiple_kernels=False)

    def _test_train_model_from_features(self, sparse_gp, multiple_kernels):
        gp_repurposer = GpRepurposer(self.source_model, self.source_model_layers)

        num_inducing = self.num_data_points_to_train
        gp_repurposer.NUM_INDUCING_SPARSE_GP = num_inducing

        if not sparse_gp:  # Select a small data set to apply normal GP classification
            self.train_features = self.train_features[:num_inducing]
            self.train_labels = self.train_labels[:num_inducing]
            self.feature_mean = self.train_features.mean(axis=0)

        if multiple_kernels:
            trained_model = gp_repurposer._train_model_from_features(self.train_features, self.train_labels,
                                                                     {'l1': self.train_feature_indices[:4],
                                                                      'l2': self.train_feature_indices[4:]})
        else:
            trained_model = gp_repurposer._train_model_from_features(self.train_features, self.train_labels,
                                                                     {'l1': self.train_feature_indices})

        assert np.array_equal(gp_repurposer.feature_mean, self.feature_mean)
        self._validate_trained_gp_model(trained_model, sparse_gp, num_inducing, multiple_kernels)

    def _validate_trained_gp_model(self, trained_model, sparse_gp, num_inducing, multiple_kernels):
        # Validate type of model
        if sparse_gp:
            assert all(isinstance(model, GPy.models.SparseGPClassification) for model in trained_model)
        else:
            assert all(isinstance(model, GPy.models.GPClassification) for model in trained_model)

        for index, model in enumerate(trained_model):
            if multiple_kernels:
                assert isinstance(model.kern, GPy.kern.Add)
                assert isinstance(model.kern.l1, GPy.kern.RBF)
                assert isinstance(model.kern.l2, GPy.kern.RBF)
            else:
                assert isinstance(model.kern, GPy.kern.RBF)
            assert np.array_equal(model.kern.active_dims, self.train_feature_indices)
            expected_labels = np.loadtxt('{}GPmodel.{}.Y.out'.format(self._test_data_dir, index)).reshape(103, 1)
            expected_features = self.train_features - self.feature_mean
            if not sparse_gp:  # A smaller data set was selected to apply normal GP classification
                expected_labels = expected_labels[:num_inducing]
                expected_features = expected_features[:num_inducing]
            assert np.array_equal(model.Y, expected_labels)
            assert np.array_equal(model.X, expected_features)

    def test_predict_label_from_features(self):
        gp_repurposer = GpRepurposer(self.source_model, self.source_model_layers, apply_l2_norm=True)
        gp_repurposer.target_model = gp_repurposer._train_model_from_features(
            self.train_features[:self.num_data_points_to_train],
            self.train_labels[:self.num_data_points_to_train],
            {'l1': self.train_feature_indices})
        predicted_labels = gp_repurposer._predict_label_from_features(self.test_features
                                                                      [:self.num_data_points_to_predict])
        self._validate_prediction_results(predicted_labels, test_predict_probability=False,
                                          expected_accuracy=self.expected_accuracy_from_features,
                                          num_predictions=self.num_data_points_to_predict)

    def test_predict_probability_from_features(self):
        gp_repurposer = GpRepurposer(self.source_model, self.source_model_layers, apply_l2_norm=True)
        gp_repurposer.target_model = gp_repurposer._train_model_from_features(
            self.train_features[:self.num_data_points_to_train],
            self.train_labels[:self.num_data_points_to_train],
            {'l1': self.train_feature_indices})
        predictions = gp_repurposer._predict_probability_from_features(self.test_features
                                                                       [:self.num_data_points_to_predict])
        self._validate_prediction_results(predictions, test_predict_probability=True,
                                          expected_accuracy=self.expected_accuracy_from_features,
                                          num_predictions=self.num_data_points_to_predict)

    @patch(RepurposerTestUtils.META_MODEL_REPURPOSER_MODEL_HANDLER_CLASS)
    def test_repurpose(self, mock_model_handler):
        # Patch model_handler and then create gp_repurposer
        mock_model_handler.return_value = RepurposerTestUtils.get_mock_model_handler_object()
        mock_model_handler.return_value.get_layer_output.return_value = {'l1': self.train_features}, self.train_labels
        gp_repurposer = GpRepurposer(self.source_model, self.source_model_layers)
        gp_repurposer.NUM_INDUCING_SPARSE_GP = 5  # To speed-up unit test running time
        self._run_common_repurposer_tests(gp_repurposer)

    def _validate_trained_model(self, target_model):
        self._validate_trained_gp_model(target_model, sparse_gp=True, num_inducing=100, multiple_kernels=False)

    @patch.object(GpRepurposer, RepurposerTestUtils.VALIDATE_REPURPOSE_METHOD_NAME)
    @patch(RepurposerTestUtils.META_MODEL_REPURPOSER_MODEL_HANDLER_CLASS)
    def test_repurpose_calls_validate(self, mock_model_handler, mock_validate_method):
        self._test_repurpose_calls_validate(mock_model_handler, mock_validate_method)

    @patch.object(GpRepurposer, RepurposerTestUtils.VALIDATE_PREDICT_METHOD_NAME)
    @patch(RepurposerTestUtils.META_MODEL_REPURPOSER_MODEL_HANDLER_CLASS)
    def test_predict_label(self, mock_model_handler, validate_method):
        self._test_predict(mock_model_handler, validate_method, test_predict_probability=False,
                           expected_accuracy=self.expected_accuracy)

    @patch.object(GpRepurposer, RepurposerTestUtils.VALIDATE_PREDICT_METHOD_NAME)
    @patch(RepurposerTestUtils.META_MODEL_REPURPOSER_MODEL_HANDLER_CLASS)
    def test_predict_probability(self, mock_model_handler, validate_method):
        self._test_predict(mock_model_handler, validate_method, test_predict_probability=True,
                           expected_accuracy=self.expected_accuracy)

    def test_serialisation(self):
        self._test_gp_serialisation(sparse_gp=True, multiple_kernels=False)
        self._test_gp_serialisation(sparse_gp=True, multiple_kernels=True)
        self._test_gp_serialisation(sparse_gp=False, multiple_kernels=True)
        self._test_gp_serialisation(sparse_gp=False, multiple_kernels=False)

    def _test_gp_serialisation(self, sparse_gp, multiple_kernels):
        gp_repurposer = GpRepurposer(self.source_model, self.source_model_layers, apply_l2_norm=True)
        num_inducing = 2
        gp_repurposer.NUM_INDUCING_SPARSE_GP = num_inducing

        if not sparse_gp:  # Select a small data set to apply normal GP classification
            self.train_features = self.train_features[:num_inducing]
            self.train_labels = self.train_labels[:num_inducing]
            self.feature_mean = self.train_features.mean(axis=0)

        if multiple_kernels:
            gp_repurposer.target_model = gp_repurposer._train_model_from_features(self.train_features,
                                                                                  self.train_labels,
                                                                                  {'l1': self.train_feature_indices[:4],
                                                                                   'l2': self.train_feature_indices[
                                                                                         4:]})
        else:
            gp_repurposer.target_model = gp_repurposer._train_model_from_features(self.train_features,
                                                                                  self.train_labels,
                                                                                  {'l1': self.train_feature_indices})

        # Save and load repurposer to test serialization
        loaded_repurposer = self._save_and_load_repurposer(gp_repurposer)

        # Validate repurposer properties
        self._compare_gp_repurposers(gp_repurposer, loaded_repurposer)

        # Get prediction results using both repurposers
        predictions_before = gp_repurposer._predict_probability_from_features(self.test_features
                                                                              [:self.num_data_points_to_predict])
        predictions_after = loaded_repurposer._predict_probability_from_features(self.test_features
                                                                                 [:self.num_data_points_to_predict])

        # Compare probabilities predicted per test instance
        self.assertTrue(predictions_before.shape == predictions_after.shape,
                        "Prediction shape is incorrect. Expected: {} Actual: {}"
                        .format(predictions_before.shape, predictions_after.shape))

        for sample_id, prediction in enumerate(predictions_before):
            self.assertTrue(np.allclose(prediction, predictions_after[sample_id]),
                            "Incorrect prediction for sample id: {}. Expected: {} Actual: {}"
                            .format(sample_id, predictions_before[sample_id], predictions_after[sample_id]))

        # Validate if accuracy is above expected threshold
        predicted_labels = np.argmax(predictions_after, axis=1)
        accuracy = np.mean(predicted_labels == self.test_labels[:self.num_data_points_to_predict])
        expected_accuracy = 0.3
        self.assertTrue(accuracy >= expected_accuracy, "Accuracy {} less than {}".format(accuracy, expected_accuracy))

    def _save_and_load_repurposer(self, gp_repurposer):
        file_path = 'test_serialisation'
        RepurposerTestUtils._remove_files_with_prefix(file_path)
        assert not os.path.isfile(file_path + '.json')
        gp_repurposer.save_repurposer(model_name=file_path, save_source_model=False)
        assert os.path.isfile(file_path + '.json')
        loaded_repurposer = load(file_path, source_model=gp_repurposer.source_model)
        RepurposerTestUtils._remove_files_with_prefix(file_path)
        return loaded_repurposer

    def _compare_gp_repurposers(self, repurposer1, repurposer2):
        self.assertTrue(type(repurposer1) == type(repurposer2),
                        "Incorrect repurposer type. Expected: {} Actual: {}".format(type(repurposer1),
                                                                                    type(repurposer2)))
        self.assertTrue(isinstance(repurposer2.target_model, type(repurposer1.target_model)),
                        "Incorrect target_model type. Expected: {} Actual: {}".format(type(repurposer1.target_model),
                                                                                      type(repurposer2.target_model)))
        self.assertTrue(len(repurposer1.target_model) == len(repurposer2.target_model),
                        "Incorrect number of target_models. Expected:{} Actual:{}"
                        .format(len(repurposer1.target_model), len(repurposer2.target_model)))
        for model_id, target_model in enumerate(repurposer1.target_model):
            self.assertTrue(isinstance(repurposer2.target_model[model_id], type(target_model)),
                            "Incorrect GP model type. Expected:{} Actual:{}"
                            .format(type(repurposer1.target_model[model_id]), type(repurposer2.target_model[model_id])))
        RepurposerTestUtils._assert_common_attributes_equal(repurposer1, repurposer2)

    def test_binary_classification(self):
        train_features = np.array([[0.0286274, 0.41107054, 0.30557073], [0.18646135, 0.71026038, 0.87030804],
                                   [0.46904668, 0.96190886, 0.85772885], [0.40327128, 0.5739354, 0.21895921],
                                   [0.53548, 0.9645708, 0.56493308], [0.80917639, 0.78891976, 0.96257564],
                                   [0.10951679, 0.75733494, 0.10935291]])
        train_labels = np.array([0, 0, 0, 1, 0, 0, 1])
        gp_repurposer = GpRepurposer(self.source_model, self.source_model_layers)
        gp_repurposer.target_model = gp_repurposer._train_model_from_features(train_features, train_labels,
                                                                              {'l1': np.arange(0, 3)})
        self.assertTrue(len(gp_repurposer.target_model) == 1,
                        "Number of GP models expected: 1. Got: {}".format(len(gp_repurposer.target_model)))

        # Validate predicted probabilities
        test_features = np.array([[0.63747595, 0.86516482, 0.21255967],
                                  [0.33403457, 0.43162212, 0.77119909],
                                  [0.1678248, 0.41870605, 0.37232554]])
        test_labels = np.array([1, 0, 0])
        expected_probabilities = np.array([[0.48597323, 0.51402677],
                                           [0.67488224, 0.32511776],
                                           [0.55386502, 0.44613498]])
        predicted_probabilities = gp_repurposer._predict_probability_from_features(test_features)
        self.assertTrue(np.allclose(predicted_probabilities, expected_probabilities))

        # Validate predicted labels
        predicted_labels = gp_repurposer._predict_label_from_features(test_features)
        self.assertTrue(np.array_equal(predicted_labels, test_labels))
