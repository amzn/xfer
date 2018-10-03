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
import random
from unittest.mock import patch

import mxnet as mx
from mxnet import gluon

from xfer import BnnRepurposer
from xfer.bnn_classifier import BnnClassifier
from .test_meta_model_repurposer import MetaModelRepurposerTestCase
from ..repurposer_test_utils import RepurposerTestUtils


class BnnRepurposerTestCase(MetaModelRepurposerTestCase):

    def setUp(self):
        np.random.seed(1)
        random.seed(1)
        mx.random.seed(1)

        super(BnnRepurposerTestCase, self).setUp()

        N = 10
        self.train_features = self.train_features[:N]
        self.train_labels = self.train_labels[:N]

        # Override base repurpose_class with 'BnnRepurposer' to run base tests with instance of BNN Repurposer
        self.repurposer_class = BnnRepurposer

        # Minimum expected performance
        self.minimum_expected_accuracy = 0.2

    def test_train_model_from_features(self):
        bnn_repurposer = BnnRepurposer(self.source_model, self.source_model_layers, num_epochs=3)
        bnn_model = bnn_repurposer._train_model_from_features(self.train_features, self.train_labels)
        self._validate_trained_model(bnn_model)

    def test_predict_probability_from_features(self):
        self._test_predict_from_features(test_predict_probability=True,
                                         expected_accuracy=self.minimum_expected_accuracy)

    def test_predict_label_from_features(self):
        self._test_predict_from_features(test_predict_probability=False,
                                         expected_accuracy=self.minimum_expected_accuracy)

    # TODO: This overrides the method in _test_predict_from_features due to the lack of serialization functionality for
    # the bnn repurposer. Once the serialization is implemented this method will be deleted.
    def _test_predict_from_features(self, test_predict_probability, expected_accuracy):
        """ Used to test 'predict_from_features' implementation in derived classes """
        # Create repurposer
        repurposer = self.repurposer_class(self.source_model, self.source_model_layers, num_samples_mc_prediction=5,
                                           num_epochs=2, num_samples_mc=1)

        repurposer.target_model = repurposer._train_model_from_features(self.train_features, self.train_labels)

        if test_predict_probability:
            results = repurposer._predict_probability_from_features(self.test_features)
        else:
            results = repurposer._predict_label_from_features(self.test_features)

        self._validate_prediction_results(results, test_predict_probability, expected_accuracy)

    @patch.object(BnnRepurposer, RepurposerTestUtils.VALIDATE_PREDICT_METHOD_NAME)
    @patch(RepurposerTestUtils.META_MODEL_REPURPOSER_MODEL_HANDLER_CLASS)
    def test_predict_probability(self, mock_model_handler, validate_method):
        self._test_predict(mock_model_handler, validate_method, test_predict_probability=True,
                           expected_accuracy=self.minimum_expected_accuracy)

    @patch.object(BnnRepurposer, RepurposerTestUtils.VALIDATE_PREDICT_METHOD_NAME)
    @patch(RepurposerTestUtils.META_MODEL_REPURPOSER_MODEL_HANDLER_CLASS)
    def test_predict_label(self, mock_model_handler, validate_method):
        self._test_predict(mock_model_handler, validate_method, test_predict_probability=False,
                           expected_accuracy=self.minimum_expected_accuracy)

    # TODO: This overrides the method in test_meta_model_repurposer due to the lack of serialization fucntionality for
    # the bnn repurposer. Once the serialization is implemented this method will be deleted.
    def _test_predict(self, mock_model_handler, mock_validate_method, test_predict_probability, expected_accuracy):
        """ Test for predict wrapper in meta model base class """
        # Patch model_handler and then create repurposer object
        mock_model_handler.return_value = RepurposerTestUtils.get_mock_model_handler_object()

        # Create repurposer
        repurposer = self.repurposer_class(self.source_model, self.source_model_layers, num_samples_mc_prediction=5,
                                           num_epochs=2)

        # Identify which predict function to test
        if test_predict_probability:
            predict_function = repurposer.predict_probability
        else:
            predict_function = repurposer.predict_label

        # Load target model from file
        model = repurposer._train_model_from_features(self.train_features, self.train_labels)
        repurposer.target_model = model

        # Call predict method and get prediction results
        mock_model_handler.return_value.get_layer_output.return_value = self.test_feature_dict, self.test_labels
        mock_validate_method.reset_mock()
        # Mocking iterator because get_layer_output is patched
        results = predict_function(test_iterator=self.mock_object)

        # Check if predict called validate
        self.assertTrue(mock_validate_method.call_count == 1,
                        "Predict expected to called {} once. Found {} calls".
                        format(RepurposerTestUtils.VALIDATE_PREDICT_METHOD_NAME, mock_validate_method.call_count))

        self._validate_prediction_results(results, test_predict_probability, expected_accuracy)

    def _validate_trained_model(self, model):
        # Validate type of model
        expected_type = BnnClassifier
        actual_type = type(model)
        self.assertTrue(actual_type == expected_type,
                        "Expected model of type: {}. Instead got: {}".format(expected_type, actual_type))

        shapes_model = [x.shape for x in model.model.collect_params().values()]
        shapes_posterior = model.var_posterior.shapes
        for shape_model, shape_posterior in zip(shapes_model, shapes_posterior):
            self.assertTrue(shape_model == shape_posterior, "Shapes of the model and the variational posterior do not \
             match. {} vs {}".format(shape_model, shape_posterior))

    @patch(RepurposerTestUtils.META_MODEL_REPURPOSER_MODEL_HANDLER_CLASS)
    def test_repurpose(self, mock_model_handler):
        # Patch model_handler and then create bnn_repurposer
        mock_model_handler.return_value = RepurposerTestUtils.get_mock_model_handler_object()
        N = 5
        train_feature_dict_subset = {k: v[:N] for k, v in self.train_feature_dict.items()}
        mock_model_handler.return_value.get_layer_output.return_value = train_feature_dict_subset, self.train_labels[:N]
        self._test_repurpose(n_jobs=-1)  # Use all cores
        self._test_repurpose(n_jobs=1)  # Use single core

    def _test_repurpose(self, n_jobs=-1):
        bnn_repurposer = BnnRepurposer(self.source_model, self.source_model_layers, num_epochs=2)

        # Target model is not initialized yet
        self.assertTrue(bnn_repurposer.target_model is None, "Target model not expected to be initialized at this \
        point")

        # Call repurpose
        bnn_repurposer.repurpose(self.mock_object)

        # Validate target model is now set
        self.assertTrue(bnn_repurposer.target_model is not None, "Repurpose failed to set target model")

        # Validate trained model
        self._validate_trained_model(bnn_repurposer.target_model)

    def test_build_nn(self):
        bnn_repurposer = BnnRepurposer(self.source_model, self.source_model_layers)
        x_tr = self.train_features.astype(np.dtype(np.float32))
        y_tr = self.train_labels.astype(np.dtype(np.float32))
        train_data = gluon.data.DataLoader(gluon.data.ArrayDataset(x_tr, y_tr),
                                           batch_size=bnn_repurposer.batch_size, shuffle=True)

        neural_network, shapes = bnn_repurposer._build_nn(train_data, len(np.unique(y_tr)))
        for layer in neural_network:
            assert isinstance(layer, gluon.nn.Dense)

        # Input layer: Fully connected layer with 2 parameters (affine transformation plus bias)
        assert shapes[0] == (bnn_repurposer.n_hidden, self.train_features.shape[1])
        assert shapes[1] == (bnn_repurposer.n_hidden, )

        # Intermediate layers: Each of them is a Fully connected layer with 2 parameters (affine transformation plus
        # bias)
        for ss in range(2, len(shapes)-2):
            if ss % 2 == 0:
                assert shapes[ss] == (bnn_repurposer.n_hidden, bnn_repurposer.n_hidden)
            else:
                assert shapes[ss] == (bnn_repurposer.n_hidden, )

        # Output layer: Fully connected layer with 2 parameters (affine transformation plus bias)
        assert shapes[len(shapes)-2] == (self.n_classes, bnn_repurposer.n_hidden)
        assert shapes[len(shapes)-1] == (self.n_classes, )

    def _validate_prediction_results(self, results, test_predict_probability, expected_minimum_accuracy):
        # Validate type of prediction results
        self.assertTrue(type(results) == np.ndarray,
                        "Prediction results expected to be numpy array. Instead got: {}".format(type(results)))

        # Validate shape of prediction results
        if test_predict_probability:
            expected_shape = (self.n_test_instances, self.n_classes)
        else:
            expected_shape = (self.n_test_instances,)
        self.assertTrue(results.shape == expected_shape,
                        "Prediction results shape is incorrect. Expected: {}. Got: {}".format(expected_shape,
                                                                                              results.shape))

        # Validate if prediction probabilities sum to 1
        if test_predict_probability:
            probability_sum = np.sum(results, axis=1)
            array_of_ones = np.ones(shape=(self.n_test_instances,))
            self.assertTrue(np.allclose(probability_sum, array_of_ones), "Sum of predicted probabilities is not 1")

        # Validate accuracy of prediction results
        labels = np.argmax(results, axis=1) if test_predict_probability else results

        self.assertTrue(np.mean(labels == self.test_labels) >= expected_minimum_accuracy, "Prediction accuracy is " +
                        "incorrect. Minimum accuracy expected: {}. Actual accuracy: {}".format(
                        expected_minimum_accuracy,
                        np.mean(labels == self.test_labels)))

    def _assert_target_model_equal(self, model1, model2):
        assert model1.normalizer.__dict__ == model2.normalizer.__dict__

        assert model1.var_posterior.shapes == model2.var_posterior.shapes
        assert model1.var_posterior.ctx == model2.var_posterior.ctx

        for key in model1.var_posterior.raw_params.keys():
            for count, _ in enumerate(model1.var_posterior.raw_params[key]):
                assert np.array_equal(model1.var_posterior.raw_params[key][count].asnumpy(),
                                      model2.var_posterior.raw_params[key][count].asnumpy())

        for key in model1.var_posterior.params.keys():
            for count, value in enumerate(model1.var_posterior.params[key]):
                assert np.array_equal(value.data(), model2.var_posterior.params[key][count].data())
                assert value.grad_req == model2.var_posterior.params[key][count].grad_req
                assert value.name == model2.var_posterior.params[key][count].name

    def _assert_attributes_equal(self, repurposer1, repurposer2):
        super()._assert_attributes_equal(repurposer1, repurposer2)
        assert repurposer1.annealing_weight == repurposer2.annealing_weight

    def test_get_params(self):
        repurposer = BnnRepurposer(self.source_model, self.source_model_layers)

        params = repurposer.get_params()
        expected_params = {
            'context_function': 'cpu',
            'num_devices': 1,
            'feature_layer_names': ['fc1', 'fc2'],
            'bnn_context_function': 'cpu',
            'sigma': 100.0,
            'num_layers': 1,
            'n_hidden': 10,
            'num_samples_mc': 3,
            'learning_rate': 0.001,
            'batch_size': 20,
            'num_epochs': 200,
            'start_annealing': 20,
            'end_annealing': 40,
            'num_samples_mc_prediction': 100,
            'verbose': 0
        }

        assert params == expected_params

    def test_verbose(self):
        bnn_repurposer = BnnRepurposer(self.source_model, self.source_model_layers, num_epochs=3, verbose=True,
                                       num_samples_mc=1)
        N = 2
        with self.assertLogs() as cm:
            bnn_repurposer._train_model_from_features(self.train_features[:N], self.train_labels[:N])
        print(cm.output)
        assert len(cm.output) == 3

        bnn_repurposer = BnnRepurposer(self.source_model, self.source_model_layers, num_epochs=3, verbose=False,
                                       num_samples_mc=1)
        with self.assertRaises(AssertionError):
            with self.assertLogs():
                bnn_repurposer._train_model_from_features(self.train_features[:N], self.train_labels[:N])
