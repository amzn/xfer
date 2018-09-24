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
import pytest

import mxnet as mx
import random
import numpy as np

import xfer
from ..repurposer_test_utils import RepurposerTestUtils


@pytest.mark.integration
class WorkflowTestCase(TestCase):
    __metaclass__ = ABCMeta

    def setUp(self):
        np.random.seed(1)
        random.seed(1)
        mx.random.seed(1)

        self.label_name = 'prob_label'
        self.save_name = self.__class__.__name__
        self.meta_model_feature_layer_name = ['flatten']
        self.pre_saved_prefix = 'tests/data/pre_saved_repurposers/pre_saved_'
        self.expected_accuracy = None  # Overridden in derived classes

        RepurposerTestUtils.download_squeezenet()
        RepurposerTestUtils.unzip_mnist_sample()

        # Load source model
        self.source_model = mx.module.Module.load('squeezenet_v1.1', 0, label_names=[self.label_name],
                                                  data_names=('data',))
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
        if self.__class__ == WorkflowTestCase:  # base class
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
        self.expected_accuracy = 0.95

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
        self.expected_accuracy = 0.88

    def get_repurposer(self, source_model):
        rep = xfer.GpRepurposer(source_model, self.meta_model_feature_layer_name, apply_l2_norm=True)
        rep.NUM_INDUCING_SPARSE_GP = 5
        return rep


class BnnWorkflowTestCase(WorkflowTestCase):
    def setUp(self):
        super().setUp()
        self.min_expected_accuracy = 0.59

    def get_repurposer(self, source_model):
        return xfer.BnnRepurposer(source_model, self.meta_model_feature_layer_name, num_samples_mc_prediction=10,
                                  num_epochs=200, num_samples_mc=5)

    def assert_accuracy(self, accuracy):
        self.assertTrue(accuracy >= self.min_expected_accuracy,
                        'accuracy: {}, minimum expected: {}'.format(accuracy, self.min_expected_accuracy))


class NnftWorkflowTestCase(WorkflowTestCase):
    def setUp(self):
        super().setUp()
        self.min_accuracy = 0.61
        self.prev_accuracy = None

    def get_repurposer(self, source_model):
        return xfer.NeuralNetworkFineTuneRepurposer(source_model, transfer_layer_name='flatten', target_class_count=5,
                                                    num_epochs=5)

    def assert_accuracy(self, accuracy):
        assert accuracy >= self.min_accuracy, 'accuracy: {}, min expected: {}'.format(accuracy, self.min_accuracy)
        if self.prev_accuracy is None:
            self.prev_accuracy = accuracy
        else:
            assert accuracy == self.prev_accuracy, 'accuracy: {}, previous: {}'.format(accuracy, self.prev_accuracy)


class NnrfWorkflowTestCase(WorkflowTestCase):
    def setUp(self):
        super().setUp()
        self.min_accuracy = 0.58
        self.prev_accuracy = None

    def get_repurposer(self, source_model):
        fixed_layers = ['conv1', 'fire2_squeeze1x1', 'fire2_expand1x1', 'fire2_expand3x3', 'fire3_squeeze1x1',
                        'fire3_expand1x1', 'fire3_expand3x3', 'fire4_squeeze1x1', 'fire4_expand1x1', 'fire4_expand3x3',
                        'fire5_squeeze1x1', 'fire5_expand1x1', 'fire5_expand3x3']
        random_layers = ['conv10']
        return xfer.NeuralNetworkRandomFreezeRepurposer(source_model, target_class_count=5, fixed_layers=fixed_layers,
                                                        random_layers=random_layers, num_epochs=5)

    def assert_accuracy(self, accuracy):
        assert accuracy >= self.min_accuracy, 'accuracy: {}, min expected: {}'.format(accuracy, self.min_accuracy)
        if self.prev_accuracy is None:
            self.prev_accuracy = accuracy
        else:
            assert accuracy == self.prev_accuracy, 'accuracy: {}, previous: {}'.format(accuracy, self.prev_accuracy)
