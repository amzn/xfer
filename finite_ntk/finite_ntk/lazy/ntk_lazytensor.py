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
from .utils import Rop, Jacvec, flatten
from .jacobian import Jacobian


class NeuralTangent(LazyTensor):
    def __init__(
        self,
        model,
        data,
        target=None,
        num_outputs=1,
        cross_data=None,
        cross_target=None,
        dummy=None,
        used_dims=None,
        keep_outputs=False,
        **kwargs
    ):
        r"""
        NTK is a kernel with the Jacobian matrix defined by a neural network (model)
        currently is only tested for single outputs.
        
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
        keep_outputs (int): whether to use batch mode or to squeeze the features (default: None)
        """
        # required for gpytorch semantics to work properly
        dummy = NonLazyTensor(torch.tensor([[]], device=data.device, dtype=data.dtype))

        super(NeuralTangent, self).__init__(
            dummy=dummy,
            model=model,
            data=data,
            target=target,
            cross_data=cross_data,
            cross_target=cross_target,
            num_outputs=num_outputs,
            used_dims=used_dims,
            **kwargs
        )

        self.model = model
        self.data = data
        self.target = target
        self.num_outputs = num_outputs
        self.used_dims = used_dims
        self.keep_outputs = keep_outputs

        # construct the criterion
        if used_dims is not None:
            criterion = lambda x, y: x[:, used_dims]
        else:
            criterion = lambda x, y: x

        if num_outputs == 1:
            self.criterion = lambda x, y: criterion(x, y).reshape(-1, 1)
        else:
            self.criterion = criterion

        if cross_data is not None:
            self.cross_data = cross_data
            self.cross_target = cross_target
            self.use_cross = True
        else:
            self.use_cross = False

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def device(self):
        return self.data.device

    def detach_(self):
        # given that ntk operates solely on derivatives, do nothing
        pass

    def _transpose_nonbatch(self):
        if self.use_cross:
            return NeuralTangent(
                model=self.model,
                data=self.cross_data,
                target=self.cross_target,
                num_outputs=self.num_outputs,
                cross_data=self.data,
                cross_target=self.target,
                used_dims=self.used_dims,
            )
        else:
            # symmetric matrix
            return self

    def _size(self, val=None):
        size_list = []

        if self.num_outputs > 1 or self.keep_outputs:
            size_list.append(self.num_outputs)

        size_list.append(self.data.size(0))

        if self.use_cross:
            size_list.append(self.cross_data.size(0))
        else:
            size_list.append(self.data.size(0))

        if val is not None:
            return size_list[val]
        else:
            return torch.Size(size_list)

    def _matmul(self, lhs):
        r"""
        lhs: vector of shape num_output x n_2 x r
        """

        lhs_norm = torch.norm(lhs, dim=-2, keepdim=True)

        # check if all norms are zeros and return a zero matrix if so
        if lhs_norm.eq(0.0).all():
            if self.num_outputs == 1:
                return torch.zeros(
                    self.data.size(0), lhs.size(-1), device=lhs.device, dtype=lhs.dtype
                )
            else:
                return torch.zeros(
                    self.num_outputs,
                    self.data.size(0),
                    lhs.size(-1),
                    device=lhs.device,
                    dtype=lhs.dtype,
                )

        # vec is now r x n_2 x num_output
        vec = (lhs / lhs_norm).transpose(0, -1)  # transpose and normalize

        # self.model.train()

        prod_list = []
        for v in vec:
            # ensure that v has two dimensions so we can slice it below
            if len(v.shape) == 1:
                v = v.unsqueeze(1)

            # zero gradients incase weirdness is occurring
            self.model.zero_grad()

            with torch.enable_grad():
                if self.use_cross:
                    loss = self.criterion(self.model(self.cross_data), self.cross_target)
                else:
                    loss = self.criterion(self.model(self.data), self.target)

                curr_prod = []
                for dim in range(self.num_outputs):
                    # compute jacobian vector product
                    z_current = Jacvec(
                        loss[..., dim], self.model.parameters(), v[..., dim]
                    )

                    # compute second loss
                    if self.use_cross:
                        cross_loss = self.criterion(self.model(self.data), self.target)
                    else:
                        cross_loss = loss

                    # now perform r op
                    prod_current = Rop(
                        cross_loss[..., dim], self.model.parameters(), z_current
                    )[0]
                    curr_prod.append(prod_current)

                prod_list.append(torch.stack(curr_prod))

        res = torch.stack(prod_list, dim=-2)
        output = res.transpose(-2, -1) * lhs_norm

        if not self.keep_outputs or self.num_outputs == 1:
            output = output.squeeze(0)

        output[torch.isnan(output)] = 0.0

        return output

    def _get_indices(self, row_index, col_index, *batch_indices):
        return self._getitem(row_index, col_index, *batch_indices)

    def _getitem(self, row_index, col_index, *batch_indices):
        # if batch indices is none, do nothing
        if len(batch_indices) is 0:
            num_outputs = self.num_outputs
            used_dims = self.used_dims
        # if batch indices is a None slice, do nothing
        else:
            used_dims = batch_indices[0]
            # compute length of range derived from slicing...
            if type(batch_indices[0]) is slice:
                num_outputs = len(list(range(*used_dims.indices(self.num_outputs))))
            else:
                num_outputs = len(batch_indices)

        row_data = self.data[row_index, ...]
        col_data = self.data[col_index, ...]

        if self.target is not None:
            row_target = self.target[row_index].squeeze(0)
        else:
            row_target = None

        if self.use_cross and self.cross_target is not None:
            col_target = self.cross_target[col_index].squeeze(0)
        else:
            col_target = None

        return NeuralTangent(
            model=self.model,
            data=row_data,
            target=row_target,
            num_outputs=num_outputs,
            cross_data=col_data,
            cross_target=col_target,
            used_dims=used_dims,
        )

    def diag(self):
        """
        computes an exact diagonal
        This method scales linearly in the data x output calls
        """
        if self.num_outputs > 1:
            diag_vec = torch.zeros(
                self.num_outputs,
                self.data.size(0),
                device=self.data.device,
                dtype=self.data.dtype,
            )
        else:
            diag_vec = torch.zeros(
                self.data.size(0), device=self.data.device, dtype=self.data.dtype
            )

        for i, curr_data in enumerate(self.data):
            self.model.zero_grad()
            output = self.model(curr_data.unsqueeze(0)).squeeze()

            if self.num_outputs > 1:
                for j in range(self.num_outputs):
                    gradval = flatten(
                        torch.autograd.grad(
                            output[j], self.model.parameters(), retain_graph=True
                        )
                    )

                diag_vec[j, i] = gradval.norm() ** 2
            else:
                gradval = flatten(
                    torch.autograd.grad(
                        output, self.model.parameters(), retain_graph=True
                    )
                )

                diag_vec[i] = gradval.norm() ** 2

        return diag_vec

    def _approx_diag(self):
        """
        Computes an approximate diagonal, useful for preconditioners
        """
        if self.size(-2) != self.size(-1):
            raise NotImplementedError(
                "diag does not make sense when matrix is not square"
            )

        # calling approx diag
        with torch.set_grad_enabled(True):
            loss = self.criterion(self.model(self.data), self.target)

        ones_list = []
        for param in self.model.parameters():
            ones_list.append(torch.ones_like(param))

        # this may not strictly be an upper bound because J^T 1 may not
        # be a good approximator of \sum_p |df/d\theta_p|
        # J^T 1 = \sum_j J_{ij} (returns a n dimensional vector)
        jac_sum_by_point = Rop(loss, self.model.parameters(), ones_list)[0]

        if self.num_outputs == 1:
            jac_sum_by_point = jac_sum_by_point.squeeze(-1)
        elif len(jac_sum_by_point.shape) > 1:
            jac_sum_by_point = jac_sum_by_point.t()

        # squares the n dimensional vector
        return jac_sum_by_point.pow(2.0)

    def get_root(self, dim=-2):
        if self.use_cross and dim == -1:
            return Jacobian(
                self.model,
                self.cross_data,
                self.cross_target,
                num_outputs=self.num_outputs,
            )
        else:
            return Jacobian(
                self.model, self.data, self.target, num_outputs=self.num_outputs
            )

    def get_expansion(self, **kwargs):
        from .fvp_reg import FVPR_FD

        # we want N * F because F is a mean over the data pts
        return self.data.shape[0] * FVPR_FD(self.model, self.data, **kwargs)

    def _unsqueeze_batch(self, dim):
        if dim == 0:
            return NeuralTangent(
                model=self.model,
                data=self.data,
                target=self.target,
                cross_data=self.cross_data,
                cross_target=self.cross_target,
                num_outputs=self.num_outputs,
                used_dims=self.used_dims,
                keep_outputs=True,
            )
        else:
            return super()._unsqueeze_batch(dim)
