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
from abc import ABCMeta, abstractmethod

# TODO: Update all docstrings


class MetaRepurposer:
    """
    Base Class for repurposers that do meta-learning

    :param input_model: Source neural network to use for meta-learning.
    :type input_model: :class:`mxnet.gluon.Block`
    """
    __metaclass__ = ABCMeta

    def __init__(self, input_model):
        self.input_model = input_model
        self.output_model = None

    @abstractmethod
    def repurpose(self, train_tasks, **kwargs):
        """
        Meta-learn parameters of the input model such that they are optimized for the provided train_tasks.

        :param train_tasks: List of training tasks to use for meta-learning.
        :type train_tasks: List of gluon data loaders #TODO: update syntax
        """
        pass
