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

# This script contains a basic implementation of ABLR as described in Perrone, et
# al, 2018 (http://papers.nips.cc/paper/7917-scalable-hyperparameter-transfer-learning.pdf)
# for the sinusoidal curves that we test.

import torch
import gpytorch

import pickle

from finite_ntk.utils import ablr_compute_loss, ablr_compute_predictions
from utils import parse_and_generate_tasks

# generate model (3 layer MLP with 100 outputs for the features)
model = torch.nn.Sequential(
    torch.nn.Linear(1, 40),
    torch.nn.Tanh(),
    torch.nn.Linear(40, 40),
    torch.nn.Tanh(),
    torch.nn.Linear(40, 100),
)

# parse arguments and generate tasks
task_dataset = parse_and_generate_tasks()

nuisance_pars = -5.0 * torch.ones(len(task_dataset), 2)
nuisance_pars.requires_grad = True

model_optimizer = torch.optim.Adam(list(model.parameters()) + [nuisance_pars], lr=1e-2)

# # warm starting
for id, task in enumerate(task_dataset):
    if id == 0:
        num_steps = 2500
    else:
        num_steps = 150

    for step in range(num_steps):

        model_optimizer.zero_grad()

        loss = ablr_compute_loss(task[0], task[1], model, nuisance_pars[id])
        loss.backward()

        model_optimizer.step()

        if step % 10 is 0:
            print("At end of training steps: ", loss.item())

pred_y_list = [
    ablr_compute_predictions(
        torch.linspace(-6.5, 6.5, 200).view(-1, 1),
        model,
        nuisance_pars[i],
        task_dataset[i][0],
        task_dataset[i][1],
    )
    for i in range(len(task_dataset))
]

with open("ablr_transfer_output.pkl", "wb") as handle:
    pickle.dump(pred_y_list, handle, pickle.HIGHEST_PROTOCOL)
