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


def train_fullds(
    net,
    train_x,
    train_y,
    lr=1e-3,
    iterations=100,
    criterion=torch.nn.MSELoss(),
    num_batches_per_epoch=3,
    **kwargs
):
    r"""training routine for training over a full dataset
    net: torch.nn model
    train_x: tensor of inputs
    train_y: tensor of response
    lr: learning rate specified
    iterations: number of epochs
    criterion: loss function (default MSE Loss)
    num_batches_per_epochs
    """

    td = torch.utils.data.TensorDataset(train_x, train_y)
    loader = torch.utils.data.DataLoader(
        td, shuffle=True, batch_size=max(1, int(train_x.shape[0] / num_batches_per_epoch))
    )

    optimizer = torch.optim.SGD(net.parameters(), lr=lr, **kwargs)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=max(1, int(iterations / 5)), gamma=0.1
    )

    for i in range(iterations):
        for input, label in loader:

            def closure():
                net.zero_grad()
                loss = criterion(net(input), label)
                loss.backward()
                return loss

            loss = optimizer.step(closure)
        scheduler.step()
        if i % max(1, int(iterations / 10)) is 0:
            print(loss)


def ablr_compute_loss(data, target, model, pars):
    r"""specialized loss function for ABLR
    http://papers.nips.cc/paper/7917-scalable-hyperparameter-transfer-learning
    """
    softplus_inv_pars = 1.0 / torch.nn.functional.softplus(pars)

    features = model(data)
    features_lt = gpytorch.lazy.RootLazyTensor(features) * softplus_inv_pars[0]
    marginal_likelihood_dist = gpytorch.distributions.MultivariateNormal(
        torch.zeros_like(target), features_lt.add_diag(softplus_inv_pars[1])
    )
    return -marginal_likelihood_dist.log_prob(target)


def ablr_compute_predictions(test_data, model, pars, train_data, train_response):
    r"""
    specialized prediction function for ablr
    http://papers.nips.cc/paper/7917-scalable-hyperparameter-transfer-learning
    """
    softplus_pars = torch.nn.functional.softplus(pars)

    scale = softplus_pars[1] / softplus_pars[0]
    features = model(train_data)
    features_lt = gpytorch.lazy.RootLazyTensor(features.t()) * scale

    phi_y = features.t() @ train_response
    pred_mean_cache = features_lt.add_jitter(1.0).inv_matmul(phi_y)
    pred_y = scale * model(test_data) @ pred_mean_cache

    return pred_y
