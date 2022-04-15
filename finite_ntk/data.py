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

import pandas as pd
import numpy as np
import torch


def subsample(train_inputs, train_targets, train_targets_var, n_samp, seed=110):
    """Selects n_samp random rows from training data
    train_inputs (tensor): full input data
    train_targets (tensor): full response data
    train_targets_var (tensor): full response variability data
    n_samp (int): size of response
    seed (int): random seed (numpy)
    """
    np.random.seed(seed)

    idx = np.random.permutation(range(len(train_inputs)))[:n_samp]
    return train_inputs[idx], train_targets[idx], train_targets_var[idx]


def unitize(x):
    """Puts design space on a unit cube"""
    x1 = x - x.min()
    return x1 / x1.max()


def generate_data(
    nsamples=2000, train_year=2012, test_year=2016, grid_size=200, seed=110, hdf_loc=None
):
    r"""
    generates subsampled dataset from the hdf_location given years, grids, etc.
    nsamples (int): dataset size
    train_year (int): year to use from dataset
    test_year (int): year to test on from dataset
    grid_size (int): size of grid
    seed (int): random seed for subsampling
    hdf_loc (str): location of dataset hdf5 file
    """
    
    df = pd.read_hdf(hdf_loc, "full")

    is_train_year = torch.from_numpy((df["year"] == train_year).values)
    is_ng = torch.from_numpy(df["is_ng"].values).bool()
    is_test = torch.from_numpy((df["year"] == test_year).values)

    all_x = torch.from_numpy(df[["longitude", "latitude", "year"]].values).float()
    lon_lims = (all_x[:, 0].min().item(), all_x[:, 0].max().item())
    lat_lims = (all_x[:, 1].min().item(), all_x[:, 1].max().item())
    extent = lon_lims + lat_lims

    all_x[:, 0] = unitize(all_x[:, 0])
    all_x[:, 1] = unitize(all_x[:, 1])

    # don't include year
    all_x = all_x[:, :-1]

    all_y = torch.from_numpy(df["mean"].values).float()
    all_y_var = torch.from_numpy(df["std_dev"].values).pow(2).float()

    train_inputs, train_targets, train_targets_var = (
        all_x[is_train_year],
        all_y[is_train_year],
        all_y_var[is_train_year],
    )
    test_x, test_y, test_y_var = all_x[is_test], all_y[is_test], all_y_var[is_test]

    # Generate nxn grid of test points spaced on a grid of size 1/(n-1) in [0,1]x[0,1] for evaluation
    n = grid_size
    L = torch.linspace(0, 1, n)
    X, Y = torch.meshgrid(L, L)
    grid_x = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=-1)

    # let's start with a small set of samples
    train_inputs, train_targets, train_targets_var = subsample(
        train_inputs, train_targets, train_targets_var, nsamples, seed=seed
    )

    # mark nigeria - not great but works reasonably well
    ng_coords = (n * all_x[:, :2]).round().long()[is_ng]
    sparse_ng = torch.sparse.LongTensor(
        ng_coords.transpose(0, 1),
        torch.ones(ng_coords.size(0)).long(),
        torch.Size([n, n]),
    )
    inside = sparse_ng.to_dense().reshape(-1) > 0

    return train_inputs, train_targets, test_x, test_y, inside, extent
