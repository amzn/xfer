# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
from gpytorch.kernels.kernel import Kernel
from torch import diag

from .ntk_lazytensor import NeuralTangent
from collections import OrderedDict


class NTK(Kernel):
    def __init__(self, model, use_linearstrategy=False, **kwargs):
        r"""
        NTK is a kernel with the Jacobian matrix defined by a neural network (model).
        Formally, K(x, y) = J(x)^T J(y), where J(.) is the Jacobian matrix of a neural
        network. To keep computation under control, we compute matrix vector products 
        iteratively, via `z = J(y)v` and then `J(x)^T z`. Noting that gpytorch can 
        handle different prediction strategies, we include an option as to perform 
        inference in parameter space via a new linear prediction strategy
        (`use_linearstrategy=True`). This kernel can be batched by setting kwargs in
        gpytorch style.
        
        model: model with output parameters to compute Jacobians from

        use_linearstrategy: whether to expand into parameter space
        """
        super(NTK, self).__init__(**kwargs)
        self.model = model

        # register constraints for botorch if you want to call botorch.fit
        for param in self.model.children():
            param.__setattr__("_constraints", OrderedDict())

        self.use_linearstrategy = use_linearstrategy

        self.kwargs = kwargs

    def diag(self):
        output = self.evaluate()
        return diag(output)

    def get_root(self, x, **kwargs):
        kernel = NeuralTangent(
            model=self.model, data=x, cross_data=None, **self.kwargs, **kwargs
        )
        return kernel.get_root()

    def prediction_strategy(
        self, train_inputs, train_prior_dist, train_labels, likelihood
    ):
        if self.use_linearstrategy:
            from ..strategies import LinearPredictionStrategy

            return LinearPredictionStrategy(
                train_inputs, train_prior_dist, train_labels, likelihood
            )
        else:
            from gpytorch.models.exact_prediction_strategies import (
                DefaultPredictionStrategy,
            )

            return DefaultPredictionStrategy(
                train_inputs, train_prior_dist, train_labels, likelihood
            )

    def forward(self, x1, x2=None, diag=False, last_dim_is_batch=False, **kwargs):
        x1_ = x1

        # the following performs standard gpytorch based reshaping and batching
        if last_dim_is_batch:
            x1_ = x1_.transpose(-1, -2).unsqueeze(-1)

        if x2 is not None:
            if last_dim_is_batch:
                x2_ = x2.transpose(-1, -2).unsqueeze(-1)
            else:
                x2_ = x2

            # we define a NeuralTangent lazy tensor that defines the products of the
            # Jacobians properly. Here, we include x2_.
            output = NeuralTangent(
                model=self.model, data=x1_, cross_data=x2_, **self.kwargs, **kwargs
            )
        else:
            # we define a NeuralTangent lazy tensor that defines the products of the
            # Jacobians properly. We do not need to include x2_ if we are only computing
            # K(x, x).
            output = NeuralTangent(
                model=self.model, data=x1_, cross_data=None, **self.kwargs, **kwargs
            )

        # sadly, we currently do not support fast versions of kernel diagonals.
        if diag:
            output = output.diag()

        return output
