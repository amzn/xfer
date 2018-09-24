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
from unittest import TestCase
import pytest
from collections import OrderedDict

from xfer import MetaModelRepurposer
from ..repurposer_test_utils import RepurposerTestUtils


@pytest.mark.integration
class MetaModelRepurposerIntegrationTest(TestCase):
    """
    Test interaction of MetaModelRepurposer with ModelHandler
    """

    EXPECTED_FEATURE_INDICES_PER_LAYER = 'expected_feature_indices_per_layer'
    EXPECTED_LABELS = 'expected_labels'
    EXPECTED_FEATURE_SHAPE = 'expected_feature_shape'

    def setUp(self):
        self.source_model = RepurposerTestUtils.create_mxnet_module()
        self.repurposer_class = MetaModelRepurposer

    def test_instantiation_valid_input(self):
        feature_layer_names_in_source_model = [RepurposerTestUtils.LAYER_FC1]
        repurposer = self.repurposer_class(self.source_model, feature_layer_names_in_source_model)
        self.assertTrue(repurposer.source_model == self.source_model)
        self.assertTrue(repurposer.feature_layer_names == feature_layer_names_in_source_model)
        self.assertTrue(repurposer.source_model_handler.layer_names == RepurposerTestUtils.ALL_LAYERS)

    def test_instantiation_feature_layer_names_not_in_source_model(self):
        # Some feature_layer_names not found in source_model
        feature_layer_names_some_not_in_source_model = [RepurposerTestUtils.LAYER_FC1, 'phantom_layer_2']
        self.assertRaisesRegex(ValueError, "feature_layer_name 'phantom_layer_2' is not found in source_model",
                               self.repurposer_class, self.source_model, feature_layer_names_some_not_in_source_model)

        # All feature_layer_names not found in source_model
        feature_layer_names_all_not_in_source_model = ['phantom_layer_1', 'phantom_layer_2']
        self.assertRaisesRegex(ValueError, "feature_layer_name 'phantom_layer_1' is not found in source_model",
                               self.repurposer_class, self.source_model, feature_layer_names_all_not_in_source_model)

    def test_get_features_from_source_model_single_layer(self):
        # Test with one feature layer
        feature_layer_names = ['fullyconnected1']
        expected_feature_indices = OrderedDict()
        expected_feature_indices['fullyconnected1'] = np.arange(0, 64)
        expected_outputs = {self.EXPECTED_FEATURE_SHAPE: (9984, 64),
                            self.EXPECTED_FEATURE_INDICES_PER_LAYER: expected_feature_indices}
        self._test_get_features_from_source_model(feature_layer_names, expected_outputs)

    def test_get_features_from_source_model_multiple_layers(self):
        # Test with two feature layers
        feature_layer_names = ['fullyconnected1', 'fullyconnected2']
        expected_feature_indices = OrderedDict()
        expected_feature_indices['fullyconnected1'] = np.arange(0, 64)
        expected_feature_indices['fullyconnected2'] = np.arange(64, 74)
        expected_outputs = {self.EXPECTED_FEATURE_SHAPE: (9984, 74),
                            self.EXPECTED_FEATURE_INDICES_PER_LAYER: expected_feature_indices}
        self._test_get_features_from_source_model(feature_layer_names, expected_outputs)

        feature_layer_names = ['fullyconnected2', 'fullyconnected1']
        expected_feature_indices = OrderedDict()
        expected_feature_indices['fullyconnected2'] = np.arange(0, 10)
        expected_feature_indices['fullyconnected1'] = np.arange(10, 74)
        expected_outputs = {self.EXPECTED_FEATURE_SHAPE: (9984, 74),
                            self.EXPECTED_FEATURE_INDICES_PER_LAYER: expected_feature_indices}
        self._test_get_features_from_source_model(feature_layer_names, expected_outputs)

    def _test_get_features_from_source_model(self, feature_layer_names, expected_outputs):
        # Create repurposer
        source_model = mx.module.Module.load(prefix=RepurposerTestUtils.MNIST_MODEL_PATH_PREFIX, epoch=10)
        repurposer = self.repurposer_class(source_model=source_model, feature_layer_names=feature_layer_names)

        # Create data iterator to extract features from source model
        data_iterator = RepurposerTestUtils.create_mnist_test_iterator()
        meta_model_data = repurposer.get_features_from_source_model(data_iterator=data_iterator)

        # Compare with expected outputs
        self.assertTrue(meta_model_data.features.shape == expected_outputs[self.EXPECTED_FEATURE_SHAPE])
        self.assertTrue(np.array_equal(meta_model_data.labels, RepurposerTestUtils.get_labels(data_iterator)))
        expected_feature_indices_per_layer = expected_outputs[self.EXPECTED_FEATURE_INDICES_PER_LAYER]
        RepurposerTestUtils.assert_feature_indices_equal(expected_feature_indices_per_layer,
                                                         meta_model_data.feature_indices_per_layer)
