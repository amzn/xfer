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

from sklearn.svm import SVC
import numpy as np

from xfer import SvmRepurposer
from .test_meta_model_repurposer import MetaModelRepurposerTestCase
from ..repurposer_test_utils import RepurposerTestUtils


class SvmRepurposerTestCase(MetaModelRepurposerTestCase):
    def setUp(self):
        super(SvmRepurposerTestCase, self).setUp()

        # Override base repurpose_class with 'SvmRepurposer' to run base tests with SVM repurposer
        self.repurposer_class = SvmRepurposer
        self.target_model_path = self._test_data_dir + 'svm_model_probability.sav'
        self.expected_accuracy = 0.6571428571428571

    def test_train_model_from_features(self):
        svm_repurposer = SvmRepurposer(self.source_model, self.source_model_layers)
        model = svm_repurposer._train_model_from_features(self.train_features[:10], self.train_labels[:10])
        self._validate_trained_model(model)

    def test_predict_label_from_features(self):
        self._test_predict_from_features(test_predict_probability=False, expected_accuracy=0.657142857143)

    def test_predict_probability_from_features(self):
        self._test_predict_from_features(test_predict_probability=True, expected_accuracy=0.6)

    @patch.object(SvmRepurposer, RepurposerTestUtils.VALIDATE_PREDICT_METHOD_NAME)
    @patch(RepurposerTestUtils.META_MODEL_REPURPOSER_MODEL_HANDLER_CLASS)
    def test_predict_label(self, mock_model_handler, validate_method):
        """ Test predict_label wrapper in meta model base class using svm_repurposer object"""
        self._test_predict(mock_model_handler, validate_method, test_predict_probability=False,
                           expected_accuracy=0.657142857143)

    @patch.object(SvmRepurposer, RepurposerTestUtils.VALIDATE_PREDICT_METHOD_NAME)
    @patch(RepurposerTestUtils.META_MODEL_REPURPOSER_MODEL_HANDLER_CLASS)
    def test_predict_probability(self, mock_model_handler, validate_method):
        """ Test predict_probability wrapper in meta model base class using svm_repurposer object"""
        self._test_predict(mock_model_handler, validate_method, test_predict_probability=True, expected_accuracy=0.6)

    @patch(RepurposerTestUtils.META_MODEL_REPURPOSER_MODEL_HANDLER_CLASS)
    def test_repurpose(self, mock_model_handler):
        """ Test Repurpose wrapper in meta model base class using svm repurposer object"""
        mock_model_handler.return_value = RepurposerTestUtils.get_mock_model_handler_object()
        N = 10
        train_feature_dict_subset = {k: v[:N] for k, v in self.train_feature_dict.items()}
        mock_model_handler.return_value.get_layer_output.return_value = train_feature_dict_subset, self.train_labels[:N]
        repurposer = SvmRepurposer(self.source_model, self.source_model_layers)
        self._run_common_repurposer_tests(repurposer)

    @patch.object(SvmRepurposer, RepurposerTestUtils.VALIDATE_REPURPOSE_METHOD_NAME)
    @patch(RepurposerTestUtils.META_MODEL_REPURPOSER_MODEL_HANDLER_CLASS)
    def test_repurpose_calls_validate(self, mock_model_handler, mock_validate_method):
        self._test_repurpose_calls_validate(mock_model_handler, mock_validate_method)

    def _validate_trained_model(self, model):
        # Validate type of model
        expected_type = SVC
        actual_type = type(model)
        self.assertTrue(actual_type == expected_type,
                        "Expected model of type: {}. Instead got: {}".format(expected_type, actual_type))

        # Validate model properties
        expected_model_intercept = np.loadtxt(self._test_data_dir + 'SVMmodel.intercept_.out')
        expected_model_dual_coef = np.loadtxt(self._test_data_dir + 'SVMmodel.dual_coef_.out')
        expected_model_n_support = np.loadtxt(self._test_data_dir + 'SVMmodel.n_support_.out')
        expected_model_support = np.loadtxt(self._test_data_dir + 'SVMmodel.support_.out')
        expected_model_support_vectors = np.loadtxt(self._test_data_dir + 'SVMmodel.support_vectors_.out')

        self.assertTrue(np.isclose(model.intercept_, expected_model_intercept).all())
        self.assertTrue(np.isclose(model.dual_coef_, expected_model_dual_coef).all())
        self.assertTrue(np.isclose(model.n_support_, expected_model_n_support).all())
        self.assertTrue(np.isclose(model.support_, expected_model_support).all())
        self.assertTrue(np.isclose(model.support_vectors_, expected_model_support_vectors).all())

    def test_get_params(self):
        svm_repurposer = SvmRepurposer(self.source_model, self.source_model_layers)

        params = svm_repurposer.get_params()
        expected_params = {
            'context_function': 'cpu',
            'num_devices': 1,
            'feature_layer_names': ['fc1', 'fc2'],
            'c': 1.0,
            'kernel': 'linear',
            'gamma': 'auto',
            'enable_probability_estimates': False
        }

        assert params == expected_params
