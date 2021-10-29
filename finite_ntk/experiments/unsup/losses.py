# Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

from gpytorch.utils.lanczos import lanczos_tridiag

from finite_ntk.lazy import Jacobian, FVP_FD, flatten, unflatten_like, Rop


def map_crossentropy(
    model,
    num_classes=10,
    bias=True,
    wd=1,
    current_pars=None,
    num_data=1,
    eval_mode=False,
):
    r"""
    constructor for MAP estimation, returns a function
    model (nn.module): torch model
    num_classes (int): number of classes
    bias (bool): whether to include the bias parameters list in the loss.
    current_pars (list/iterable): parameter list, only used when bias=True
    num_data (int): number of data points
    eval_mode (bool): whether to include the regularizer in the model definition. overloaded
    to more generally mean if the loss function is in train (regularizer included) or eval_mode mode
    (regularizer not included for test LL computation)
    """
    if bias and current_pars is None:
        Warning("Nothing will be returned because current_pars is none")

    #model_pars = flatten(model.parameters())
    #rand_proj = torch.randn(512, 10).cuda() 

    def criterion(current_pars, input_data, target, return_predictions=True):
        r"""
        Loss function for MAP

        current_pars (list/iterable): parameter list
        input_data (tensor): input data for model
        target (tensor): response
        return_predictions (bool):if predictions should be returned as well as loss
        """
        rhs = current_pars[0] - flatten(model.parameters()).view(-1,1)

        #def functional_model(*params):
        #    unflattened_params = unflatten_like(params[0], model.parameters())
        #    for param, new_param in zip(model.parameters(), unflattened_params):
        #        param.data.mul_(0.).add_(new_param)
        #    return model(input_data)

        #features = torch.autograd.functional.jvp(functional_model, 
        #    flatten(model.parameters()), v=flatten(model.parameters()).detach())
        with torch.enable_grad():
            features = Rop(model(input_data), model.parameters(), unflatten_like(rhs, model.parameters()))[0]

        if bias:
            features = wd * features + model(input_data)

        #if rand_proj is None:
        #    nc = target.max() + 1
        #    if nc > 10:
        #        nc = 100
        #    else:
        #        nc = 10
        #
        #    rand_proj = torch.randn(features.shape[-1], nc)
        #    print('i just created a new random projection!!!')

        predictions = features @ current_pars[1]

        #rhs = current_pars[0]
        #if bias:
        #    rhs = current_pars[0] - model_pars.view(-1, 1)

        # compute J^T \theta
        #predictions = Jacobian(model=model, data=input_data, num_outputs=1)._t_matmul(rhs)
        #print('shape of the predictions', predictions.shape)
        #predictions_reshaped = predictions.reshape(target.shape[0], num_classes)

        #if bias:
        #    #print('adding into the bias term??')
        #    predictions_reshaped = predictions_reshaped + model(input_data)

        loss = (
            torch.nn.functional.cross_entropy(predictions, target)
            * target.shape[0]
        )
        #regularizer = current_pars[0].norm() * wd
        
        if eval_mode:
            output = loss
        else:
            #output = num_data * loss + regularizer
            output = loss

        return output, predictions 

    return criterion


def laplace_crossentropy(
    model,
    num_classes=10,
    bias=True,
    wd=1e-4,
    current_pars=None,
    num_data=1,
    eval_mode=False,
):
    r"""
    constructor for Laplace approximation, returns a loss function
    model (nn.module): torch model
    num_classes (int): number of classes
    bias (bool): whether to include the bias parameters list in the loss.
    current_pars (list/iterable): parameter list, only used when bias=True
    num_data (int): number of data points
    eval_mode: whether to include the regularizer in the model definition. overloaded
    to more generally mean if the loss function is in train (regularizer included) or eval_mode mode
    (regularizer not included and we perform sampling)
    """
    model_pars = flatten(model.parameters())

    def criterion(current_pars, input_data, target, return_predictions=True):
        r"""
        Loss function for Laplace

        current_pars (list/iterable): parameter list
        input_data (tensor): input data for model
        target (tensor): response
        return_predictions (bool):if predictions should be returned as well as loss
        """
        if eval_mode:
            # this means prediction time
            # so do a Fisher vector product + jitter, take the tmatrix invert the cholesky decomp and sample
            # F \approx Q T Q' => F^{-1} \approx Q T^{-1} Q'
            # F^{-1/2} \approx Q T^{-1/2}
            fvp = ((num_data / input_data.shape[0]) * FVP_FD(model, input_data)).add_jitter(1.0)
            qmat, tmat = lanczos_tridiag(
                fvp.matmul,
                30,
                dtype=current_pars[0].dtype,
                device=current_pars[0].device,
                init_vecs=None,
                matrix_shape=[current_pars[0].shape[0], current_pars[0].shape[0]],
            )

            eigs, evecs = torch.symeig(tmat, eigenvectors=True)

            # only consider the top half of the eigenvalues bc they're reliable
            eigs_gt_zero = torch.sort(eigs)[1][-int(tmat.shape[0] / 2) :]

            # update the eigendecomposition
            # note that @ is a matmul
            updated_evecs = (qmat @ evecs)[:, eigs_gt_zero]

            z = torch.randn(
                eigs_gt_zero.shape[0], 1, device=tmat.device, dtype=tmat.dtype
            )
            approx_lz = updated_evecs @ torch.diag(1.0 / eigs[eigs_gt_zero].pow(0.5)) @ z
            sample = current_pars[0] + approx_lz
        else:
            sample = current_pars[0]

        rhs = sample
        if bias:
            rhs = sample - model_pars.view(-1, 1)

        predictions = Jacobian(model=model, data=input_data, num_outputs=1)._t_matmul(rhs)
        predictions_reshaped = predictions.reshape(target.shape[0], num_classes)

        if bias:
            predictions_reshaped = predictions_reshaped + model(input_data)

        loss = (
            torch.nn.functional.cross_entropy(predictions_reshaped, target)
            * target.shape[0]
        )
        regularizer = current_pars[0].norm() * wd

        if eval_mode:
            output = loss
        else:
            output = num_data * loss + regularizer

        return output, predictions_reshaped

    return criterion


def vi_crossentropy(
    model,
    num_classes=10,
    bias=True,
    wd=1e-4,
    current_pars=None,
    num_data=1,
    eval_mode=False,
):
    r"""
    constructor for SVI approximation, returns a loss function
    model (nn.module): torch model
    num_classes (int): number of classes
    bias (bool): whether to include the bias parameters list in the loss.
    current_pars (list/iterable): parameter list, only used when bias=True
    num_data (int): number of data points
    eval_mode: whether to include the regularizer in the model definition. overloaded
    to more generally mean if the loss function is in train (regularizer included) or eval_mode mode
    (regularizer not included and we perform sampling)
    """
    model_pars = flatten(model.parameters())

    def criterion(current_pars, input_data, target, return_predictions=True):
        r"""
        Loss function for SVI

        current_pars (list/iterable): parameter list
        input_data (tensor): input data for model
        target (tensor): response
        return_predictions (bool):if predictions should be returned as well as loss
        """
        current_dist = torch.distributions.Normal(
            current_pars[0], torch.nn.functional.softplus(current_pars[1])
        )
        prior_dist = torch.distributions.Normal(
            torch.zeros_like(current_pars[0]), 1 / wd * torch.ones_like(current_pars[1])
        )

        if not eval_mode:
            sample = current_dist.rsample()
        else:
            sample = current_pars[0]

        rhs = sample
        if bias:
            rhs = sample - model_pars.view_as(sample)

        # compute J^T \theta
        predictions = Jacobian(model=model, data=input_data, num_outputs=1)._t_matmul(rhs)

        predictions_reshaped = predictions.reshape(target.shape[0], num_classes)
        if bias:
            predictions_reshaped = predictions_reshaped + model(input_data)

        loss = (
            torch.nn.functional.cross_entropy(predictions_reshaped, target)
            * target.shape[0]
        )

        regularizer = (
            torch.distributions.kl_divergence(current_dist, prior_dist).sum() / num_data
        )

        if eval_mode:
            output = loss
        else:
            output = loss + regularizer

        return output, predictions_reshaped

    return criterion
