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
from unittest.mock import patch, Mock

import os
import numpy as np
import mxnet as mx

from xfer import NeuralNetworkRepurposer, load
from ..repurposer_test_utils import RepurposerTestUtils


class NeuralNetworkRepurposerTestCase(TestCase):

    def setUp(self):
        self.repurposer_class = NeuralNetworkRepurposer
        self.mxnet_model = RepurposerTestUtils.create_mxnet_module()
        self.dropout_model_path_prefix = 'tests/data/test_model_with_dropout'
        self.source_layers = RepurposerTestUtils.ALL_LAYERS

    def test_validate_before_repurpose(self):
        # Test invalid inputs
        neural_network_repurposer = NeuralNetworkRepurposer(source_model=None)
        self.assertRaisesRegex(TypeError, "Cannot repurpose because source_model is not an `mxnet.mod.Module` object",
                               neural_network_repurposer._validate_before_repurpose)

        neural_network_repurposer = NeuralNetworkRepurposer(source_model='')
        self.assertRaisesRegex(TypeError, "Cannot repurpose because source_model is not an `mxnet.mod.Module` object",
                               neural_network_repurposer._validate_before_repurpose)

        # Test valid input
        neural_network_repurposer = NeuralNetworkRepurposer(source_model=self.mxnet_model)
        neural_network_repurposer._validate_before_repurpose()

    def test_validate_before_predict(self):
        # Test invalid inputs
        neural_network_repurposer = NeuralNetworkRepurposer(source_model=self.mxnet_model)

        # Target model is neither created through repurpose nor explicitly assigned
        self.assertRaisesRegex(TypeError, "Cannot predict because target_model is not an `mxnet.mod.Module` object",
                               neural_network_repurposer._validate_before_predict)

        neural_network_repurposer.target_model = {}
        self.assertRaisesRegex(TypeError, "Cannot predict because target_model is not an `mxnet.mod.Module` object",
                               neural_network_repurposer._validate_before_predict)

        # Assert validate raises error for mxnet module that is not trained yet
        neural_network_repurposer.target_model = self.mxnet_model
        self.assertRaisesRegex(ValueError,
                               "target_model params aren't initialized. Ensure model is trained before calling predict",
                               neural_network_repurposer._validate_before_predict)

        # Test valid input
        neural_network_repurposer.target_model = mx.module.Module.load(prefix=self.dropout_model_path_prefix, epoch=0)
        neural_network_repurposer._validate_before_predict()

    @patch.object(NeuralNetworkRepurposer, RepurposerTestUtils.VALIDATE_PREDICT_METHOD_NAME)
    def test_predict_label(self, mock_validate_method):
        self._test_predict(mock_validate_method, test_predict_probability=False)

    @patch.object(NeuralNetworkRepurposer, RepurposerTestUtils.VALIDATE_PREDICT_METHOD_NAME)
    def test_predict_probability(self, mock_validate_method):
        self._test_predict(mock_validate_method, test_predict_probability=True)

    def _test_predict(self, mock_validate_method, test_predict_probability):
        neural_network_repurposer = NeuralNetworkRepurposer(source_model=None)
        neural_network_repurposer.target_model = mx.module.Module.load(
            prefix=RepurposerTestUtils.MNIST_MODEL_PATH_PREFIX, epoch=10, label_names=None)
        test_iterator = RepurposerTestUtils.create_mnist_test_iterator()
        mock_validate_method.reset_mock()

        if test_predict_probability:
            labels = np.argmax(neural_network_repurposer.predict_probability(test_iterator), axis=1)
        else:
            labels = neural_network_repurposer.predict_label(test_iterator)

        # Check if predict called validate
        self.assertTrue(mock_validate_method.call_count == 1,
                        "Predict expected to called {} once. Found {} calls".
                        format(RepurposerTestUtils.VALIDATE_PREDICT_METHOD_NAME, mock_validate_method.call_count))

        expected_accuracy = 0.96985
        accuracy = np.mean(labels == RepurposerTestUtils.get_labels(test_iterator))
        self.assertTrue(np.isclose(accuracy, expected_accuracy, rtol=1e-3),
                        "Prediction accuracy is incorrect. Expected:{}. Got:{}".format(expected_accuracy, accuracy))

    @patch.object(NeuralNetworkRepurposer, RepurposerTestUtils.VALIDATE_REPURPOSE_METHOD_NAME)
    @patch.object(NeuralNetworkRepurposer, '_create_target_module')
    def test_repurpose_calls_validate(self, mock_create_target_module, mock_validate_method):
        neural_network_repurposer = NeuralNetworkRepurposer(source_model=self.mxnet_model)
        neural_network_repurposer.target_model = mx.module.Module.load(prefix=self.dropout_model_path_prefix, epoch=0)

        mock_validate_method.reset_mock()
        neural_network_repurposer.repurpose(Mock())
        self.assertTrue(mock_validate_method.call_count == 1,
                        "Repurpose expected to called {} once. Found {} calls".
                        format(RepurposerTestUtils.VALIDATE_REPURPOSE_METHOD_NAME, mock_validate_method.call_count))

    def test_prediction_consistency(self):
        """ Test if predict method returns consistent predictions using the same model and test data """
        if self.repurposer_class != NeuralNetworkRepurposer:
            return
        # Create test data iterator to run predictions on
        test_iterator = RepurposerTestUtils.get_image_iterator()
        # Load a pre-trained model to predict. The model has a dropout layer used for training.
        # This test is to ensure that dropout doesn't happen during prediction.
        target_model = mx.module.Module.load(prefix=self.dropout_model_path_prefix, epoch=0, label_names=None)

        # Create repurposer and set the target model loaded from file
        repurposer = NeuralNetworkRepurposer(source_model=None)
        repurposer.target_model = target_model

        # Ensure prediction results are consistent
        self._predict_and_compare_results(repurposer, test_iterator, test_predict_probability=True)
        self._predict_and_compare_results(repurposer, test_iterator, test_predict_probability=False)

    def _predict_and_compare_results(self, repurposer, test_iterator, test_predict_probability):
        # Identify predict method to test
        predict_function = repurposer.predict_probability if test_predict_probability else repurposer.predict_label

        # Call predict method multiple times and check if the predictions are consistent
        current_prediction = predict_function(test_iterator=test_iterator)
        for i in range(1, 10):
            previous_prediction = current_prediction
            current_prediction = predict_function(test_iterator=test_iterator)
            self.assertTrue(np.array_equal(previous_prediction, current_prediction), 'Predictions are inconsistent')

    def test_serialisation(self):
        if self.repurposer_class == NeuralNetworkRepurposer:
            return
        self.data_name = 'data'
        self.imglist = [[0, 'accordion/image_0001.jpg'], [0, 'accordion/image_0002.jpg'], [1, 'ant/image_0001.jpg'],
                        [1, 'ant/image_0002.jpg'], [2, 'anchor/image_0001.jpg'], [2, 'anchor/image_0002.jpg']]
        self.train_iter = mx.image.ImageIter(2, (3, 224, 224), imglist=self.imglist, path_root='tests/data/test_images',
                                             label_name='softmaxoutput1_label', data_name=self.data_name)
        self.imglist = [[0, 'accordion/image_0003.jpg'], [0, 'accordion/image_0004.jpg'], [1, 'ant/image_0003.jpg'],
                        [1, 'ant/image_0004.jpg'], [2, 'anchor/image_0003.jpg'], [2, 'anchor/image_0004.jpg']]
        self.test_iter = mx.image.ImageIter(2, (3, 224, 224), imglist=self.imglist, path_root='tests/data/test_images',
                                            label_name='softmaxoutput1_label', data_name=self.data_name)

        self._test_data_dir = 'tests/data/meta_model_repurposer_data/'
        self.labels = np.loadtxt(self._test_data_dir + '_labels.out')
        self._test_indices = np.loadtxt(self._test_data_dir + '_test_indices.out').astype(int)
        self.test_labels = self.labels[self._test_indices]

        self._test_save_load_repurposed_model(save_source=True)
        self._test_save_load_repurposed_model(save_source=False)

    def _test_save_load_repurposed_model(self, save_source):
        file_path = 'test_serialisation'
        RepurposerTestUtils._remove_files_with_prefix(file_path)
        source_model = mx.module.Module.load('tests/data/testnetv1', 0, label_names=['softmaxoutput1_label'],
                                             data_names=('data',))
        repurposer = self._get_repurposer(source_model)
        repurposer.repurpose(self.train_iter)
        results = repurposer.predict_label(test_iterator=self.test_iter)
        assert not os.path.isfile(file_path + '.json')
        repurposer.save_repurposer(model_name=file_path, save_source_model=save_source)
        assert os.path.isfile(file_path + '.json')
        if save_source:
            loaded_repurposer = load(file_path)
        else:
            loaded_repurposer = load(file_path, source_model=repurposer.source_model)
        results_loaded = loaded_repurposer.predict_label(test_iterator=self.test_iter)

        assert type(repurposer) == type(loaded_repurposer)
        accuracy1 = np.mean(results == self.test_labels)
        accuracy2 = np.mean(results_loaded == self.test_labels)
        assert abs(accuracy1 - accuracy2) < 0.05

        assert repurposer.get_params() == loaded_repurposer.get_params()
        self._assert_attributes_equal(repurposer, loaded_repurposer)
        RepurposerTestUtils._remove_files_with_prefix(file_path)

    def _get_repurposer(self, source_model):
        pass

    def _assert_attributes_equal(self, repurposer1, repurposer2):
        RepurposerTestUtils._assert_common_attributes_equal(repurposer1, repurposer2)

    @patch('mxnet.mod.Module.fit')
    def test_iterator_reset(self, mock_func):
        if self.repurposer_class == NeuralNetworkRepurposer:
            return
        data_name = 'data'
        batch_size = 2
        imglist = [[0, 'accordion/image_0001.jpg'], [0, 'accordion/image_0002.jpg'], [1, 'ant/image_0001.jpg'],
                   [1, 'ant/image_0002.jpg'], [2, 'anchor/image_0001.jpg'], [2, 'anchor/image_0002.jpg']]
        train_iter = mx.image.ImageIter(batch_size, (3, 224, 224), imglist=imglist, path_root='tests/data/test_images',
                                        label_name='softmaxoutput1_label', data_name=data_name)

        source_model = mx.module.Module.load('tests/data/testnetv1', 0, label_names=['softmaxoutput1_label'],
                                             data_names=('data',))

        repurposer = self._get_repurposer(source_model)

        # The iterator cursor should start at 0 and be 6
        assert train_iter.cur == 0
        # iterate through every batch of iterator
        for _ in range(int(len(imglist)/batch_size)):
            train_iter.next()
        assert train_iter.cur == 6

        repurposer.repurpose(train_iter)

        # This is the iterator that Module.fit() is called with
        fit_iter = mock_func.call_args_list[0][0][0]

        assert fit_iter.cur == 0
