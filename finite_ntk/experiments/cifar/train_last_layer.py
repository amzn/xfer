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

# This experiment trains the last layer of a pre-trained model (e.g. for fast
# adaptation). Specifically, we freeze all but the logistic classification layer
# (including the batch norm statistics).

import argparse
import numpy as np
import torch
import torchvision
import os

import finite_ntk

from swag import models
from swag.losses import cross_entropy

from utils import train_epoch, eval, predict

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
    if args.resume is not None:
        print("Loading Model")
        checkpoint = torch.load(args.resume)
        start_epoch = checkpoint["epoch"]
        model.load_state_dict(checkpoint["state_dict"])

    model.eval()
    pars = []
    param_name_list = list(model.named_parameters())
    for i, (n, p) in enumerate(param_name_list):
        if i < len(param_name_list) - 2:
            p.requires_grad = False
        else:
            pars.append(p)

    optimizer = torch.optim.Adam(pars, lr=args.lr_init, amsgrad=True)

    criterion = cross_entropy

    if args.epochs == 0:
        eval_dict = eval(
            loader=test_loader, model=model, criterion=criterion, verbose=True
        )
        print("Eval loss: {} Eval acc: {}".format(eval_dict["loss"], eval_dict["accuracy"]))

    for epoch in range(args.epochs):
        train_epoch(
            loader=train_loader,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            verbose=True,
        )
        eval_dict = eval(
            loader=test_loader, model=model, criterion=criterion, verbose=True
        )
        print("Eval loss: {} Eval acc: {}".format(eval_dict["loss"], eval_dict["accuracy"]))

    if args.save_path is not None:
        print("Saving predictions to ", args.save_path)
        predictions_dict = predict(
            loader=test_loader, model=model, criterion=criterion, verbose=True
        )
        np.savez(
            args.save_path,
            predictions=predictions_dict["predictions"],
            targets=predictions_dict["targets"],
        )

if __name__ == '__main__':
    main()