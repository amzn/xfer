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
import numpy as np
import sklearn
from unittest.mock import patch

from xfer import LrRepurposer
from .test_meta_model_repurposer import MetaModelRepurposerTestCase
from ..repurposer_test_utils import RepurposerTestUtils


class LrRepurposerTestCase(MetaModelRepurposerTestCase):

    def setUp(self):
        super(LrRepurposerTestCase, self).setUp()

        # Override base repurpose_class with 'LrRepurposer' to run base tests with instance of Lr Repurposer
        self.repurposer_class = LrRepurposer

        self.expected_model_coef_ = np.loadtxt(self._test_data_dir+'LRmodel.coef_.out')
        self.expected_model_intercept_ = [0., 0., 0., 0.]
        self.expected_accuracy = 0.52857142857142858
        self.target_model_path = self._test_data_dir + 'lr_model.sav'

    def test_train_model_from_features(self):
        lr_repurposer = LrRepurposer(self.source_model, self.source_model_layers)
        lr_model = lr_repurposer._train_model_from_features(self.train_features, self.train_labels)
        self._validate_trained_model(lr_model)

    def test_predict_probability_from_features(self):
        self._test_predict_from_features(test_predict_probability=True, expected_accuracy=self.expected_accuracy)

    def test_predict_label_from_features(self):
        self._test_predict_from_features(test_predict_probability=False, expected_accuracy=self.expected_accuracy)

    @patch(RepurposerTestUtils.META_MODEL_REPURPOSER_MODEL_HANDLER_CLASS)
    def test_repurpose(self, mock_model_handler):
        # Patch model_handler and then create lr_repurposer
        mock_model_handler.return_value = RepurposerTestUtils.get_mock_model_handler_object()
        mock_model_handler.return_value.get_layer_output.return_value = self.train_feature_dict, self.train_labels
        self._test_repurpose(n_jobs=-1)  # Use all cores
        self._test_repurpose(n_jobs=1)  # Use single core

    def _test_repurpose(self, n_jobs=-1):
        lr_repurposer = LrRepurposer(self.source_model, self.source_model_layers, n_jobs=n_jobs)
        self._run_common_repurposer_tests(lr_repurposer)

    @patch.object(LrRepurposer, RepurposerTestUtils.VALIDATE_REPURPOSE_METHOD_NAME)
    @patch(RepurposerTestUtils.META_MODEL_REPURPOSER_MODEL_HANDLER_CLASS)
    def test_repurpose_calls_validate(self, mock_model_handler, mock_validate_method):
        self._test_repurpose_calls_validate(mock_model_handler, mock_validate_method)

    @patch.object(LrRepurposer, RepurposerTestUtils.VALIDATE_PREDICT_METHOD_NAME)
    @patch(RepurposerTestUtils.META_MODEL_REPURPOSER_MODEL_HANDLER_CLASS)
    def test_predict_probability(self, mock_model_handler, validate_method):
        self._test_predict(mock_model_handler, validate_method, test_predict_probability=True,
                           expected_accuracy=self.expected_accuracy)

    @patch.object(LrRepurposer, RepurposerTestUtils.VALIDATE_PREDICT_METHOD_NAME)
    @patch(RepurposerTestUtils.META_MODEL_REPURPOSER_MODEL_HANDLER_CLASS)
    def test_predict_label(self, mock_model_handler, validate_method):
        self._test_predict(mock_model_handler, validate_method, test_predict_probability=False,
                           expected_accuracy=self.expected_accuracy)

    def _validate_trained_model(self, model):
        # Validate type of model
        expected_type = sklearn.linear_model.LogisticRegression
        actual_type = type(model)
        self.assertTrue(actual_type == expected_type,
                        "Expected model of type: {}. Instead got: {}".format(expected_type, actual_type))

        # Validate properties of model
        self.assertTrue(np.isclose(model.intercept_, self.expected_model_intercept_).all(),
                        "LR model intercept is incorrect")
        self.assertTrue(np.isclose(model.coef_, self.expected_model_coef_).all(), "LR model co-efficient is incorrect")

    def test_get_params(self):
        repurposer = LrRepurposer(self.source_model, self.source_model_layers)

        params = repurposer.get_params()
        expected_params = {
            'context_function': 'cpu',
            'num_devices': 1,
            'feature_layer_names': ['fc1', 'fc2'],
            'tol': 0.0001,
            'c': 1.0,
            'n_jobs': -1
        }

        assert params == expected_params
