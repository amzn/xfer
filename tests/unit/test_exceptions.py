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

from xfer.model_handler import exceptions


class CustomError(Exception):
    """Exception type for errors caused by invalid model actions."""
    pass


class TestExceptions(TestCase):
    def test_handle_mxnet_error(self):
        # Assert ModelArchitectureError is raised
        with open('tests/data/exceptions/model_architecture_error.txt', 'r') as tf:
            string = tf.read()
        error = CustomError(string)
        with self.assertRaises(exceptions.ModelArchitectureError):
            exceptions._handle_mxnet_error(error)

    def test_handle_mxnet_error_140(self):
        # Assert ModelArchitectureError is raised
        with open('tests/data/exceptions/model_architecture_error_140.txt', 'r') as tf:
            string = tf.read()
        error = CustomError(string)
        with self.assertRaises(exceptions.ModelArchitectureError):
            exceptions._handle_mxnet_error(error)

    def test_handle_mxnet_error_150(self):
        # Assert ModelArchitectureError is raised
        with open('tests/data/exceptions/model_architecture_error_150.txt', 'r') as tf:
            string = tf.read()
        error = CustomError(string)
        with self.assertRaises(exceptions.ModelArchitectureError):
            exceptions._handle_mxnet_error(error)

    def test_handle_mxnet_error_index_fail(self):
        # Assert original error is raised when a ValueError is raised because string indexing fails
        with open('tests/data/exceptions/model_architecture_error_fail.txt', 'r') as tf:
            string = tf.read()
        error = CustomError(string)
        with self.assertRaises(CustomError):
            exceptions._handle_mxnet_error(error)

    def test_handle_mxnet_error_random_string(self):
        # Assert original error is raised when string does not match expected error string because error not related to
        # weight shape mismatch
        string = 'cvbcbcbvb'
        error = CustomError(string)
        with self.assertRaises(CustomError):
            exceptions._handle_mxnet_error(error)
