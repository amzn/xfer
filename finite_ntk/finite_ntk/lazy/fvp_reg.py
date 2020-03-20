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
import torch
import gpytorch
from gpytorch.lazy import LazyTensor
import copy

from .utils import flatten


class FVPR_FD(LazyTensor):
    def __init__(self, model, data, epsilon=1e-4, dummy=None):
        r"""
        FVPR_FD is a class representing a Fisher matrix of a model on a set of data, given
        that the probability model for the data is
        p(y | model(data)) = Normal(model(data), \sigma^2). Rather than
        forming the entire Fisher information matrix, we only access the Fisher via 
        Fisher vector products found using finite differences of the KL divergence.
        Note that for the homoscedastic regression problem, we don't actually need \sigma^2.
        model: model class
        data: data that the Fisher information is to be calulated on
        epsilon: hyper-parameter
        dummy: for gpytorch semantics
        This lazy tensor should not support batch dimensions.
        """

        # required for gpytorch semantics to work properly
        dummy = gpytorch.lazy.NonLazyTensor(
            torch.tensor([[]], device=data.device, dtype=data.dtype)
        )

        super(FVPR_FD, self).__init__(
            dummy=dummy, model=model, data=data, epsilon=epsilon
        )

        self.dummy = dummy

        self.model = model
        self.data = data
        self.epsilon = epsilon

        # compute number of paraemters
        self.num_params = 0
        for p in self.model.parameters():
            self.num_params += p.numel()

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def device(self):
        return self.data.device

    def _transpose_nonbatch(self):
        # Fisher matrix is symmetric so return self
        return self

    def _size(self, val=None):
        if val == 0 or val == 1:
            return self.num_params
        else:
            return torch.Size([self.num_params, self.num_params])

    def _matmul(self, rhs):
        """
        _matmul is the meat of the fisher vector product.
        rhs: tensor as input
        We loop through all dimensions of rhs and compute finite differences 
        Fisher vector products with the regression loss using the KL divergence.
        \nabla_\theta' KL(p(y | \theta) || p(y | \theta')) |\theta' = \theta + \epsilon v
        is approximately F v * \epsilon.
        """
        rhs_norm = torch.norm(rhs, dim=0, keepdim=True)
        vec = (rhs / rhs_norm).t()  # transpose and normalize
        vec[torch.isnan(vec)] = 0.0

        # check if all norms are zeros and return a zero matrix if so
        if torch.norm(rhs, dim=0).eq(0.0).all():
            return torch.zeros(
                self.num_params, rhs.size(1), device=rhs.device, dtype=rhs.dtype
            )

        grad_list = []

        # forwards pass with current parameters
        with torch.no_grad():
            output = self.model(self.data).detach()

        # copy model state dict
        model_state_dict = copy.deepcopy(self.model.state_dict())

        for v in vec:
            # update model with \theta + \epsilon v
            i = 0
            for param_val in self.model.parameters():
                n = param_val.numel()
                param_val.data.add_(self.epsilon * v[i : i + n].view_as(param_val))
                i += n

            with torch.autograd.enable_grad():
                # y_i: response, x_i: data in below
                # forwards pass with updated parameters
                # N(y_i | f(x_i; \theta + \epsilon v), \sigma^2)
                output_prime = self.model(self.data)

                # compute kl divergence loss
                # KL(p(y_i|\theta, x_i) || p(y_i|\theta + \epsilon v, x_i))
                # assumes fixed sigma
                kl = 0.5 * (output.double() - output_prime.double()) ** 2
                kl = kl.mean(0)

                kl = kl.type(self.data.dtype).to(self.data.device)

                # compute gradient of kl divergence loss
                kl_grad = torch.autograd.grad(
                    kl, self.model.parameters(), retain_graph=True
                )
                grad_i = flatten(kl_grad)
            grad_list.append(grad_i)

            # restore model dict now -> model(\theta)
            self.model.load_state_dict(model_state_dict)

        # stack vector and turn divide by epsilon
        res = torch.stack(grad_list) / self.epsilon
        return res.t() * rhs_norm  # de-normalize at the end

    def _approx_diag(self):
        # approximate diagonal - useful for preconditioning
        grad_vec = self._matmul(
            torch.ones(self.shape[0], 1, device=self.device, dtype=self.dtype)
        )
        grad_vec = grad_vec.squeeze(-1)
        print(grad_vec.shape)
        return grad_vec.pow(2.0)

    def __getitem__(self, index):
        print("index in __getitem__ is: ", index)
        # Will not do anything except get a single row correctly
        row_id = index[0].item()
        e_i = torch.zeros(self.size(0), 1, device=list(self.model.parameters())[0].device)
        e_i[row_id] = 1
        return self._matmul(e_i).squeeze()
