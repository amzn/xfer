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

from gpytorch.lazy import LazyTensor, NonLazyTensor

from .utils import Jacvec, Rop, flatten, unflatten_like


class TransposedLT(LazyTensor):
    def __init__(self, base, **kwargs):
        r"""
        TransposedLT is a lazy tensor class that enables efficient transpositions.
        Currently, only two dimensional transpositions are implemented.

        base (gpytorch.lazy.LazyTensor): base lazy tensor
        kwargs: for other lazy tensor options (e.g. batch shaping)
        """
        super(TransposedLT, self).__init__(base, **kwargs)
        self.base = base

    def _matmul(self, rhs):
        return self.base._t_matmul(rhs)

    def _t_matmul(self, lhs):
        return self.base._matmul(lhs)

    def _transpose_nonbatch(self):
        return self.base

    def _size(self, val=None):
        base_size = self.base.size()
        if val is not None:
            return base_size[val == 0]
        else:
            return torch.Size([base_size[1], base_size[0]])

    def _get_indices(self, row_index, col_index, *batch_indices):
        res = self.base._get_indices(col_index, row_index, *batch_indices)
        return res


class Jacobian(LazyTensor):
    def __init__(
        self,
        model,
        data,
        target=None,
        num_outputs=1,
        cross_data=None,
        cross_target=None,
        dummy=None,
        **kwargs
    ):
        r"""
        Jacobian is a lazy tensor class to store Jacobian matrices and only access via
        either Jacobian vector products (_matmul) or Jacobian transpose vecotr products (_t_matmul)
        model (torch.nn): model with output parameters to compute Jacobians from
        data (torch.tensor): dataset that the Jacobian of the model is evaluated on
        target (torch.tensor): label for the data (unused - check if deprectated)
        num_ouputs (int): number of outputs of the model (default: 1)
        cross_data (torch.tensor): dataset if we are computing a cross product J_x'J_y
            (default: None which implies J_x'J_x)
        cross_target (torch.tensor): label for the cross data (unused - check if deprecated)
        dummy (None): used solely for gpytorch semantics & initialization
        used_dims (int): if model has multiple outputs; which dimension to use for the Jacobian
            (default: None which implies all dimensions)
        """
        # required for gpytorch semantics to work properly
        dummy = NonLazyTensor(torch.tensor([[]], device=data.device, dtype=data.dtype))

        super(Jacobian, self).__init__(dummy=dummy, model=model, data=data, **kwargs)

        self.model = model
        self.data = data
        self.target = target
        self.num_outputs = num_outputs

        self.loss = None

        self.num_params = 0
        for param in self.model.parameters():
            self.num_params += param.numel()

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def device(self):
        return self.data.device

    def _transpose_nonbatch(self):
        return TransposedLT(self)

    def _size(self, val=None):
        if val == 0:
            return self.num_params
        elif val == 1:
            return self.data.size(0)
        elif val == None:
            return torch.Size([self.num_params, self.data.size(0)])

    # TODO: explore forwards/backwards passes in jit
    def _matmul(self, rhs):
        rhs_norm = torch.norm(rhs, dim=0, keepdim=True)

        zero_norms = (rhs_norm == 0).squeeze(0)

        vec = (rhs / rhs_norm).t()

        vec[zero_norms, ...] = 0.0

        if torch.norm(rhs, dim=0).eq(0.0).all():
            return torch.zeros(
                self.data.size(0), rhs.size(1), device=rhs.device, dtype=rhs.dtype
            )

        prod_list = []
        for v in vec:
            self.model.zero_grad()

            if self.loss is None:
                with torch.set_grad_enabled(True):
                    self.loss = self.model(self.data)

            z_current = Jacvec(
                self.loss, self.model.parameters(), v.view(-1, self.num_outputs)
            )

            prod_list.append(flatten(z_current))

        res = torch.stack(prod_list)
        return res.t() * rhs_norm

    # TODO: explore forwards/backwards passes in jit
    def _t_matmul(self, lhs):
        lhs_norm = torch.norm(lhs, dim=0, keepdim=True)
        zero_norms = (lhs_norm == 0).squeeze(0)

        vec = (lhs / lhs_norm).t()
        vec[zero_norms, ...] = 0.0

        if torch.norm(lhs, dim=0).eq(0.0).all():
            return torch.zeros(
                self.data.size(0), lhs.size(1), device=lhs.device, dtype=lhs.dtype
            )

        # self.model.train()

        prod_list = []
        for v in vec:
            v_list = unflatten_like(v, likeTensorList=list(self.model.parameters()))
            self.model.zero_grad()
            with torch.set_grad_enabled(True):
                if self.loss is None:
                    self.loss = self.model(self.data)

                prod_current = Rop(self.loss, self.model.parameters(), v_list)
            prod_list.append(prod_current[0].view(-1))

        res = torch.stack(prod_list)
        return res.t() * lhs_norm

    def _get_cols(self, col_index, *args, **kwargs):
        subset_data = self.data[col_index.view(-1), :]
        return Jacobian(self.model, subset_data)

    def _get_indices(self, row_index, col_index, *batch_indices):
        row_jacobian = self._get_cols(col_index)
        id_i = torch.eye(row_jacobian.shape[-1], device=self.device, dtype=self.dtype)
        res = row_jacobian.matmul(id_i).t()
        return res[..., row_index[0]]
