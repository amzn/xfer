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
import copy

# link: https://github.com/wjmaddox/drbayes/blob/master/experiments/hessian_eigs/fvp.py
from .utils import flatten


class FVP_FD(LazyTensor):
    def __init__(self, model, data, epsilon=1e-4):
        r"""
        FVP_FD is a class representing a Fisher matrix of a model on a set of data, given
        that the probability model for the data is
        p(y | model(data)) = Categorical(model(data)). Rather than
        forming the entire Fisher information matrix, we only access the Fisher via 
        Fisher vector products found using finite differences of the KL divergence.
        Hence the FD tag at the end of the class name.
        model: model class
        data: data that the Fisher information is to be calulated on
        epsilon: hyper-parameter
        """
        super(FVP_FD, self).__init__(model=model, data=data, epsilon=epsilon)

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
        return self

    def _size(self, val=None):
        if val == 0 or val == 1:
            return self.num_params
        else:
            return torch.Size([self.num_params, self.num_params])

    def KL_logits(self, p_logits, q_logits):
        # computes KL divergence between two tensors of logits
        # KL(p || q) = \sum p log(p/q) \propto -\sum p log(q)
        # when we differentiate this, we really only need the cross-entropy
        # but include the other bits for enhanced numerical stability

        # this is the standard version (float/double independent)
        SM = torch.nn.Softmax(dim=1)
        p = SM(p_logits)

        # \sum_{i=1}^N \left(\sum_{k=1}^K p(y = k|x_i, \theta) (logit_p(y=k| x_i, \theta) - logit_q(y=k| x_i, \theta))\right)
        part1 = (p * (p_logits - q_logits)).sum(1).mean(0)
        # normalization constants

        # \log{ \sum_{k=1}^K exp{logit_p(y=k| x_i, \theta)} }
        # apparently implementations of LogSumExp are slower?
        r1 = torch.log(torch.exp(q_logits).sum(1))
        r2 = torch.log(torch.exp(p_logits).sum(1))
        # mean of difference of normalization constants
        part2 = (r1 - r2).mean(0)
        kl = part1 + part2
        return kl

    def _matmul(self, rhs):
        """
        _matmul is the meat of the fisher vector product.
        rhs: tensor as input
        We loop through all dimensions of rhs and compute finite differences 
        Fisher vector products with the cross entropy loss using the KL divergence.
        \nabla_\theta' KL(p(y | \theta) || p(y | \theta')) |\theta' = \theta + \epsilon v
        is approximately F v * \epsilon.
        """
        rhs_norm = torch.norm(rhs, dim=0, keepdim=True)
        vec = (rhs / rhs_norm).t()  # transpose and normalize

        # check if all norms are zeros and return a zero matrix if so
        if torch.norm(rhs, dim=0).eq(0.0).all():
            return torch.zeros(
                self.num_params, rhs.size(1), device=rhs.device, dtype=rhs.dtype
            )

        grad_list = []

        # forwards pass with current parameters
        # return logit(y_k | \theta, x_i)
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
                # forwards pass with updated parameters
                # logit(y_k | \theta + \epsilon v, x_i)
                output_prime = self.model(self.data)

                # compute kl divergence loss
                # KL(p(y|\theta, x_i) || p(y|\theta + \epsilon v, x_i))
                kl = self.KL_logits(output.double(), output_prime.double())
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
        # we use the empirical fisher for the approximate diagonal
        # empirical fisher version of F
        # diag(F) \approx (\nabla_\theta \log{p(y_i | x_i, \theta)})^2
        lst = []
        for param in self.model.parameters():
            if param.grad is not None:
                lst.append(param.grad)
            else:
                lst.append(torch.zeros_like(param))
        grad_vec = flatten(lst)
        return grad_vec.pow(2.0)

    def __getitem__(self, index):
        # Will not do anything except get a single row correctly
        row_id = index[0].item()
        e_i = torch.zeros(self.size(0), 1, device=list(self.model.parameters())[0].device)
        e_i[row_id] = 1
        return self._matmul(e_i).squeeze()
