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

# This script contains an implemention of finite NTKs in the transfer learning
# setting for regression. We use exact GPs as for most MLPs the networks are pretty
# small.

import torch
import gpytorch
import matplotlib.pyplot as plt
import argparse
import time
import pickle

import finite_ntk
import finite_ntk.utils as utils
from utils import parse_and_generate_tasks


class ExactGPModel(gpytorch.models.ExactGP):
    # exact Gaussian process class
    def __init__(self, train_x, train_y, likelihood, model, use_linearstrategy=False):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = finite_ntk.lazy.NTK(
            model=model, use_linearstrategy=use_linearstrategy
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


# parse arguments and generate tasks
task_dataset = parse_and_generate_tasks()
train_x, train_y, train_parameters = (
    task_dataset[0][0],
    task_dataset[0][1],
    task_dataset[0][2],
)

# generate model
model = torch.nn.Sequential(
    torch.nn.Linear(1, 40),
    torch.nn.Tanh(),
    torch.nn.Linear(40, 40),
    torch.nn.Tanh(),
    torch.nn.Linear(40, 1),
)

# train model
utils.train_fullds(model, train_x, train_y, iterations=2500, lr=1e-3, momentum=0.9)

# construct likelihood and gp model
gplh = gpytorch.likelihoods.GaussianLikelihood()
gpmodel = ExactGPModel(
    train_x, train_y.squeeze(), gplh, model, use_linearstrategy=args.fisher
)

# set noise to be smaller
print("residual error: ", torch.mean((model(train_x) - train_y) ** 2))
with torch.no_grad():
    gplh.noise = torch.max(
        1e-3 * torch.ones(1), torch.mean((model(train_x) - train_y) ** 2)
    )
    print("noise is: ", gplh.noise)

fig, axnl = plt.subplots(2, 2)
ax = [item for sublist in axnl for item in sublist]

plotting_data_dict = {
    "train": {
        "x": train_x.data.numpy(),
        "y": train_y.data.numpy(),
        "pred": model(train_x).data.numpy(),
        "true_parameters": train_parameters,
    }
}

# we now loop through each task computing the posterior predictive distribution
# transfer learning occurs when we call the predictive mean (everything is lazy
# so there is effectively no precomputation)
for task in range(1, tasks):
    with torch.no_grad():
        ax[task - 1].plot(
            train_x.numpy(), model(train_x).numpy(), color="blue", label="Train Pred"
        )
        ax[task - 1].scatter(train_x, train_y, color="red", label="Train")
    transfer_x, transfer_y, task_pars = task_dataset[task]

    # here, we reset the training data to ensure that all caches are cleansed
    gpmodel.set_train_data(transfer_x, transfer_y.squeeze(), strict=False)

    start = time.time()
    gpmodel.train()
    # compute the loss if so desired
    loss = gplh(gpmodel(transfer_x)).log_prob(transfer_y.squeeze())

    # compute the predictive distribution
    with gpytorch.settings.fast_pred_var():
        gpmodel.eval()
        interp_x = torch.linspace(
            torch.min(transfer_x) - 2.0, torch.max(transfer_x) + 2.0, 1000
        )
        predictive_dist = gpmodel(interp_x)

    pmean = predictive_dist.mean.data
    lower, upper = predictive_dist.confidence_region()
    lower = lower.detach()
    upper = upper.detach()

    end = time.time() - start

    # we now plot the results
    ax[task - 1].scatter(
        transfer_x.numpy(), transfer_y.numpy(), color="black", label="Test"
    )
    ax[task - 1].plot(interp_x.numpy(), pmean.numpy(), color="magenta", label="Test Pred")
    ax[task - 1].fill_between(
        interp_x.view(-1).numpy(),
        lower.data.numpy(),
        upper.data.numpy(),
        color="magenta",
        alpha=0.5,
    )
    print("Time for task ", task, " is: ", end)

    plotting_data_dict["task" + str(task)] = {
        "x": transfer_x.numpy(),
        "y": transfer_y.numpy(),
        "interp_x": interp_x.numpy(),
        "interp_pred": pmean.numpy(),
        "lower": lower.data.numpy(),
        "upper": upper.data.numpy(),
        "true_parameters": task_pars,
    }

ttle = "Posterior Predictive:"
if args.fisher:
    ttle = ttle + " Weight Space"
    plotting_data_dict["fisher"] = True
else:
    ttle = ttle + " Function Space"
    plotting_data_dict["fisher"] = False
fig.suptitle(ttle)

with open(args.output_file, "wb") as handle:
    pickle.dump(plotting_data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

plt.show()
