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
from gpytorch.lazy import LazyTensor

from .utils import flatten, unflatten_like


class FVP_AG(LazyTensor):
    def __init__(self, model, data, **kwargs):
        r"""
        FVP_AG is a class representing a Fisher matrix of a model on a set of data, given
        that the probability model for the data is
        p(y | model(data)) = Categorical(model(data)). Rather than
        forming the entire Fisher information matrix, we compute it with matrix vector products
        using second order autograd (hence the AG name).
        model: model class
        data: data that the Fisher information is to be calulated on
        epsilon: hyper-parameter
        """
        super(FVP_AG, self).__init__(data)
        self.model = model
        self.data = data

        # compute number of paraemters
        self.num_params = 0
        for p in self.model.parameters():
            self.num_params += p.numel()

    def _size(self, val=None):
        if val == 0 or val == 1:
            return self.num_params
        else:
            return (self.num_params, self.num_params)

    def _transpose_nonbatch(self):
        return self

    # A loss whose 2nd derivative is the Fisher information matrix
    # Thanks for Marc Finzi for deriving this loss fn.
    def detached_entropy(self, logits, y=None):
        # -1*\frac{1}{m}\sum_{i,k} [f_k(x_i)] \log f_k(x_i), where [] is detach
        log_probs = torch.nn.LogSoftmax(dim=1)(logits)
        probs = torch.nn.Softmax(dim=1)(logits)
        return -1 * (probs.detach() * log_probs).sum(1).mean(0)

    def _matmul(self, rhs):
        orig_dtype = rhs.dtype
        rhs = rhs.float()
        vec = rhs.t()  # transpose

        # check if all norms are zeros and return a zero matrix otherwise
        if torch.norm(vec, dim=0).eq(0.0).all():
            return torch.zeros(
                self.num_params, rhs.size(1), device=rhs.device, dtype=rhs.dtype
            )

        # form list of all vectors
        with torch.autograd.no_grad():
            vec_list = []
            for v in vec:
                vec_list.append(unflatten_like(v, self.model.parameters()))

        with torch.autograd.enable_grad():
            # compute batch loss with detached entropy
            batch_loss = self.detached_entropy(self.model(self.data))

            # first gradient wrt parameters
            grad_bl_list = torch.autograd.grad(
                batch_loss, self.model.parameters(), create_graph=True, only_inputs=True
            )

            res = []
            for vec_sublist in vec_list:
                deriv = 0
                for vec_part, grad_part in zip(vec_sublist, grad_bl_list):
                    deriv += torch.sum(vec_part.detach().double() * grad_part.double())

                # fast implicit hvp product
                hvp_list = torch.autograd.grad(
                    deriv.float(),
                    self.model.parameters(),
                    only_inputs=True,
                    retain_graph=True,
                )

                res.append(flatten(hvp_list))

        res_matrix = torch.stack(res).detach()
        return res_matrix.t().type(orig_dtype)
