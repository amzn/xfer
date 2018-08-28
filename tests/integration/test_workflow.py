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
from abc import ABCMeta, abstractmethod

import mxnet as mx
import random
import numpy as np

import xfer
from ..repurposer_test_utils import RepurposerTestUtils


class WorkflowTestCase(TestCase):
    __metaclass__ = ABCMeta

    def setUp(self):
        np.random.seed(1)
        random.seed(1)
        mx.random.seed(1)

        self.label_name = 'prob_label'
        self.save_name = self.__class__.__name__
        self.meta_model_feature_layer_name = ['fc8']
        self.pre_saved_prefix = 'tests/data/pre_saved_repurposers/pre_saved_'
        self.expected_accuracy = None  # Overridden in derived classes

        RepurposerTestUtils.download_vgg19()
        RepurposerTestUtils.unzip_mnist_sample()

        # Load source model
        self.source_model = mx.module.Module.load('vgg19', 0, label_names=[self.label_name], data_names=('data',))
        # Create train and test data iterators
        self.train_iter = RepurposerTestUtils.create_img_iter('mnist_sample/train', 20, self.label_name)
        self.test_iter = RepurposerTestUtils.create_img_iter('mnist_sample/test', 20, self.label_name)
        self.test_labels = RepurposerTestUtils.get_labels(self.test_iter)

    @abstractmethod
    def get_repurposer(self, source_model):
        pass

    def test_workflow(self):
        """
        Test workflow

        Instantiate repurposer(1), repurpose(2), predict(3), save & load with source model(4),
        predict(5), repurpose(6), predict(7), save & load without model(8), predict(9)
        """
        if self.__class__ == WorkflowTestCase:  # base class
            return
        if self.__class__ == GpWorkflowTestCase:  # Remove after release of GPy 1.9.3
            return
        # remove any old saved repurposer files
        RepurposerTestUtils._remove_files_with_prefix(self.save_name)

        # instantiate repurposer (1)
        rep = self.get_repurposer(self.source_model)

        for save_source_model in [True, False]:
            # (2/6) repurpose
            # random seeds are set before repurposing to ensure training is the same
            np.random.seed(1)
            random.seed(1)
            mx.random.seed(1)
            rep.repurpose(self.train_iter)
            # (3/7) predict
            results = rep.predict_label(self.test_iter)
            accuracy = np.mean(results == self.test_labels)
            self.assert_accuracy(accuracy)
            # (4/8) serialise
            rep.save_repurposer(self.save_name, save_source_model=save_source_model)
            del rep
            if save_source_model:
                rep = xfer.load(self.save_name)
            else:
                rep = xfer.load(self.save_name, source_model=self.source_model)
            RepurposerTestUtils._remove_files_with_prefix(self.save_name)
            # (5/9) predict
            results = rep.predict_label(self.test_iter)
            accuracy = np.mean(results == self.test_labels)
            self.assert_accuracy(accuracy)

    def test_load_pre_saved_repurposer(self):
        """ Test case to check for backward compatibility of deserialization """
        if self.__class__ in [WorkflowTestCase, NnftWorkflowTestCase, NnrfWorkflowTestCase]:
            # Skipping base class and
            # NN repurposer (because nn deserialization is done by mxnet and pre-saved nn models are large)
            return
        # Load pre-saved repurposer from file
        repurposer_file_prefix = self.pre_saved_prefix + self.__class__.__name__
        repurposer = xfer.load(repurposer_file_prefix, source_model=self.source_model)
        # Validate accuracy of predictions
        predicted_labels = repurposer.predict_label(self.test_iter)
        accuracy = np.mean(predicted_labels == self.test_labels)
        self.assert_accuracy(accuracy)

    def assert_accuracy(self, accuracy):
        self.assertTrue(accuracy == self.expected_accuracy,
                        'accuracy: {}, expected: {}'.format(accuracy, self.expected_accuracy))


class LrWorkflowTestCase(WorkflowTestCase):
    def setUp(self):
        super().setUp()
        self.expected_accuracy = 0.94

    def get_repurposer(self, source_model):
        return xfer.LrRepurposer(source_model, self.meta_model_feature_layer_name)


class SvmWorkflowTestCase(WorkflowTestCase):
    def setUp(self):
        super().setUp()
        self.expected_accuracy = 0.92

    def get_repurposer(self, source_model):
        return xfer.SvmRepurposer(source_model, self.meta_model_feature_layer_name)


class GpWorkflowTestCase(WorkflowTestCase):
    def setUp(self):
        super().setUp()
        self.expected_accuracy = 0.94

    def get_repurposer(self, source_model):
        return xfer.GpRepurposer(source_model, self.meta_model_feature_layer_name, apply_l2_norm=True)


class BnnWorkflowTestCase(WorkflowTestCase):
    def setUp(self):
        super().setUp()
        self.min_expected_accuracy = 0.90

    def get_repurposer(self, source_model):
        return xfer.BnnRepurposer(source_model, self.meta_model_feature_layer_name)

    def assert_accuracy(self, accuracy):
        self.assertTrue(accuracy > self.min_expected_accuracy,
                        'accuracy: {}, minimum expected: {}'.format(accuracy, self.min_expected_accuracy))


class NnftWorkflowTestCase(WorkflowTestCase):
    def setUp(self):
        super().setUp()
        self.min_accuracy = 0.80
        self.prev_accuracy = None

    def get_repurposer(self, source_model):
        return xfer.NeuralNetworkFineTuneRepurposer(source_model, transfer_layer_name='fc8', target_class_count=5)

    def assert_accuracy(self, accuracy):
        assert accuracy > self.min_accuracy, 'accuracy: {}, min expected: {}'.format(accuracy, self.min_accuracy)
        if self.prev_accuracy is None:
            self.prev_accuracy = accuracy
        else:
            assert accuracy == self.prev_accuracy, 'accuracy: {}, previous: {}'.format(accuracy, self.prev_accuracy)


class NnrfWorkflowTestCase(WorkflowTestCase):
    def setUp(self):
        super().setUp()
        self.min_accuracy = 0.80
        self.prev_accuracy = None

    def get_repurposer(self, source_model):
        fixed_layers = ['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2', 'conv3_1', 'conv3_2', 'conv3_3', 'conv3_4',
                        'conv4_1', 'conv4_2', 'conv4_3', 'conv4_4', 'conv5_1', 'conv5_2', 'conv5_3']
        random_layers = ['fc6', 'fc7']
        return xfer.NeuralNetworkRandomFreezeRepurposer(source_model, target_class_count=5, fixed_layers=fixed_layers,
                                                        random_layers=random_layers)

    def assert_accuracy(self, accuracy):
        assert accuracy > self.min_accuracy, 'accuracy: {}, min expected: {}'.format(accuracy, self.min_accuracy)
        if self.prev_accuracy is None:
            self.prev_accuracy = accuracy
        else:
            assert accuracy == self.prev_accuracy, 'accuracy: {}, previous: {}'.format(accuracy, self.prev_accuracy)
