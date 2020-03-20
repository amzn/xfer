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

# This experiment trains a linear model with the Jacobian as the features
# for multi-class classification. We use this script for both fast adaptation
# and for training linearized models. Three different methods are used (MAP
# which is just SGD for optimization, Laplace which computes the MAP estimate and
# then at test time tests a Laplace approximation, and VI which runs SVI using SGD.)


import argparse
import numpy as np
import torch
import torchvision
import os

import finite_ntk

from swag import models
from utils import train_epoch, eval, predict

import losses
from parser import parser
from data import generate_data

def main():
    args = parser()

    torch.backends.cudnn.benchmark = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    ######
    # prepare model and dataset
    ######

    print("Using model %s" % args.model)
    model_cfg = getattr(models, args.model)

    loaders, num_classes, num_data = generate_data(args, model_cfg)
    train_loader = loaders["train"]
    test_loader = loaders["test"]

    print("Preparing model")
    model = model_cfg.base(*model_cfg.args, num_classes=num_classes, **model_cfg.kwargs)
    model.cuda()

    ## please note that this code will only work if the checkpoints are saved as cuda tensors
    print("Loading Model")
    checkpoint = torch.load(args.resume)
    start_epoch = checkpoint["epoch"]
    model.load_state_dict(checkpoint["net"])

    #######
    # prepare linearized model by cloning parameters
    current_pars = finite_ntk.lazy.utils.flatten(model.parameters())
    # but dont initialize to zero so we add a little bit of noise
    eps = 1e-6
    pars = torch.clone(current_pars.data) + eps
    pars = pars.detach_().view(-1, 1)
    pars.requires_grad = True

    if args.inference == "vi":
        sigma_pars = -5.0 * torch.ones_like(pars)

        pars = [pars, sigma_pars]
    else:
        pars = [pars]

    optimizer = torch.optim.Adam(pars, lr=args.lr_init, amsgrad=True)

    # set model in eval mode to freeze batch norm and dropout
    model.eval()

    loss_args = [model, num_classes, args.bias, args.wd, current_pars, num_data]
    loss_instances = {
        "map": losses.map_crossentropy,
        "laplace": losses.laplace_crossentropy,
        "vi": losses.vi_crossentropy,
    }

    try:
        loss_func = loss_instances[args.inference]
        criterion = loss_func(*loss_args)
        eval_criterion = loss_func(*loss_args, eval=True)
    except:
        raise ValueError("Inference method not found")

    if args.epochs == 0:
        eval_dict = eval(
            loader=test_loader, model=pars, criterion=criterion, verbose=True
        )
        print("Eval loss: {} Eval acc: {}".format(eval_dict["loss"], eval_dict["accuracy"]))

    for epoch in range(args.epochs):
        train_epoch(
            loader=train_loader,
            model=pars,
            criterion=criterion,
            optimizer=optimizer,
            verbose=True,
        )
        if epoch % args.eval_freq == 0:
            eval_dict = eval(
                loader=test_loader, model=pars, criterion=eval_criterion, verbose=True
            )
            print("Eval loss: {} Eval acc: {}".format(eval_dict["loss"], eval_dict["accuracy"]))

    if args.save_path is not None:
        print("Saving predictions to ", args.save_path)
        predictions_dict = predict(
            loader=test_loader, model=pars, criterion=eval_criterion, verbose=True
        )
        np.savez(
            args.save_path,
            weights=pars[0].detach().cpu().numpy(),
            predictions=predictions_dict["predictions"],
            targets=predictions_dict["targets"],
        )

if __name__ == '__main__':
    main()
