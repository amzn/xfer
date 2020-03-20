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

# This experiment is our script for training both NNs and the NTK on the malaria
# dataset. We use a relatively wide three layer NN with a heteroscedastic likelihood
# (we tried homoscedastic but observed considerably worse results).

import torch
import gpytorch
import sys
import argparse
import finite_ntk

import data

from finite_ntk.utils import train_fullds


def compute_mse(pred, actual):
    return (pred.view_as(actual) - actual).norm().pow(2) / actual.size(-1)


def gaussian_loglikelihood(input, target, eps=1e-5):
    r"""
    heteroscedastic Gaussian likelihood where we parameterize the variance
    with the 1e-5 + softplus(network)
    input: tensor (batch + two-d, presumed to be output from model)
    target: tensor
    eps (1e-5): a nugget style term to ensure that the variance doesnt go to 0
    """
    dist = torch.distributions.Normal(
        input[:, 0], torch.nn.functional.softplus(input[:, 1]) + eps
    )
    res = -dist.log_prob(target.view(-1))
    return res.mean()


parser = argparse.ArgumentParser()
parser.add_argument(
    "--nx", help="(int) number of data points for simulated data", default=10, type=int
)
parser.add_argument(
    "--ntrain", help="(int) number of training data points", default=2000, type=int
)
parser.add_argument(
    "--ntest", help="(int) number of testing points", default=5000, type=int
)
parser.add_argument(
    "--fisher", action="store_true", help="fisher basis usage flag (default: off)"
)
parser.add_argument("--seed", help="random seed", type=int, default=10)
parser.add_argument("--output_file", type=str)
args = parser.parse_args()

torch.random.manual_seed(args.seed)

nn_model = torch.nn.Sequential(
    torch.nn.Linear(2, 500),
    torch.nn.Tanh(),
    torch.nn.Linear(500, 500),
    torch.nn.Tanh(),
    torch.nn.Linear(500, 2),
).cuda()

train_x, train_y, _, _, _, _ = data.generate_data(
    nsamples=args.ntrain, train_year=2012, seed=args.seed
)
val_x, val_y, test_x, test_y, inside, extent = data.generate_data(
    nsamples=args.nx, train_year=2016, test_year=2016, seed=args.seed
)

keep = torch.randint(test_y.shape[0], torch.Size((args.ntest,)))
test_x = test_x[keep, :]
test_y = test_y[keep]
# train nn model on data from 2012
train_fullds(
    nn_model,
    train_x.cuda(),
    train_y.unsqueeze(-1).cuda(),
    num_batches_per_epoch=10,
    iterations=500,
    lr=1e-3,
    criterion=gaussian_loglikelihood,
)

nn_predictions = nn_model(test_x.cuda()).cpu().data[:, 0]
nn_mse = compute_mse(nn_predictions, test_y)
print("MSE of NN: ", nn_mse)

# ### ntk
class NTKGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, model):
        super(NTKGP, self).__init__(train_x, train_y, likelihood)

        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = finite_ntk.lazy.NTK(
            model=model, use_linearstrategy=args.fisher, used_dims=0
        )

    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)


likelihood = gpytorch.likelihoods.GaussianLikelihood().cuda()
model = NTKGP(val_x, val_y, likelihood, nn_model).cuda()

model.eval()
pred_dist = model(test_x.cuda())
ntk_predictions = pred_dist.mean.data.cpu()
ntk_mse = compute_mse(ntk_predictions, test_y)
print("MSE of NTK:", ntk_mse)

### retrain last layer
parlength = len(list(nn_model.named_parameters()))
for i, (n, p) in enumerate(nn_model.named_parameters()):
    if i < (parlength - 2):
        p.requires_grad = False
    else:
        # print out the name of the layer to verify we are only using the last layer
        print(n)

train_fullds(
    nn_model,
    val_x.cuda(),
    val_y.unsqueeze(-1).cuda(),
    num_batches_per_epoch=10,
    iterations=100,
    lr=1e-3,
    criterion=gaussian_loglikelihood,
)
nn_predictions = nn_model(test_x.cuda()).cpu().data[:, 0]
rnn_mse = compute_mse(nn_predictions, test_y)
print("MSE of retrained NN: ", rnn_mse)

if args.output_file is not None:
    import pickle

    with open(args.output_file, "wb") as handle:
        pickle.dump(
            {
                "test_x": test_x,
                "test_y": test_y,
                "nn": nn_predictions,
                "ntk": ntk_predictions,
                "val_x": val_x,
                "val_y": val_y,
                "train_x": train_y,
                "train_x": train_x,
                "nn_mse": nn_mse,
                "ntk_mse": ntk_mse,
                "retrained_mse": rnn_mse,
            },
            handle,
            pickle.HIGHEST_PROTOCOL,
        )
