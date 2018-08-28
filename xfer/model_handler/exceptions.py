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
"""
Exceptions for Model Handler
"""


class ModelError(Exception):
    """Exception type for errors caused by invalid model actions."""
    pass


class ModelArchitectureError(Exception):
    """Exception type for errors caused when model architecture is incorrect or mismatched"""
    pass


def _handle_mxnet_error(error):
    error_string = error.__str__()
    try:
        if 'Check failed: assign(&dattr, (*vec)[i]) Incompatible attr in node' in error_string:
            start_str1 = 'expected ['
            start_index1 = error_string.index(start_str1) + len(start_str1)
            end_str1 = '], got'
            end_index1 = error_string.index(end_str1)

            start_str2 = '], got ['
            start_index2 = error_string.index(start_str2) + len(start_str2)
            end_str2 = ']\n\nStack trace returned '
            end_index2 = error_string.index(end_str2)

            correct_shape = error_string[start_index1:end_index1]
            actual_shape = error_string[start_index2:end_index2]
            raise ModelArchitectureError('Weight shape mismatch: Expected shape=({}), Actual shape=({}). This can be '
                                         'caused by incorrect layer shapes or incorrect input data shapes.'.format(
                                          correct_shape, actual_shape))
    except ValueError:
        raise error
    raise error
