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

import mxnet as mx

from xfer import utils


class UtilsTestCase(TestCase):
    def setUp(self):
        pass

    def test_serialize_ctx_fn(self):
        op = utils.serialize_ctx_fn(mx.cpu)
        assert op == 'cpu'

        op = utils.serialize_ctx_fn(mx.gpu)
        assert op == 'gpu'

        with self.assertRaises(ValueError):
            utils.serialize_ctx_fn('cpu')

    def test_deserialize_ctx_fn(self):
        op = utils.deserialize_ctx_fn('cpu')
        assert op == mx.cpu
        assert op == mx.context.cpu

        op = utils.deserialize_ctx_fn('gpu')
        assert op == mx.gpu
        assert op == mx.context.gpu

        with self.assertRaises(ValueError):
            utils.deserialize_ctx_fn(mx.cpu)

        with self.assertRaises(ValueError):
            utils.deserialize_ctx_fn(5)

    def test_assert_repurposer_file_exists(self):
        with self.assertRaises(NameError):
            utils._assert_repurposer_file_exists(['madeupfile'])

        with self.assertRaises(NameError):
            utils._assert_repurposer_file_exists([3])
