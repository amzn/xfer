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

# This file trains a model (following the wide residual network architecture) from scratch, without any knowledge
# transfer method applied. The trained model can later be used as a teacher for the student model trained on
# train_with_transfer.py file.

import os
import argparse
import torch

from functools import partial

from ignite.engine import Engine, Events

from cifar10 import get_cifar10_loaders
from wide_residual_network import (
    BATCH_SIZE,
    LEARNING_RATE,
    MOMENTUM,
    WEIGHT_DECAY,
    LEARNING_RATE_DECAY_MILESTONES,
    LEARNING_RATE_DECAY_FACTOR,
    WideResidualNetwork,
)
from util import (
    BatchUpdaterWithoutTransfer,
    BatchEvaluator,
    prepare_batch,
    LearningRateUpdater,
    MetricLogger,
    attach_pbar_and_metrics,
)


def main(width, depth, max_epochs, state_dict_path, device, data_dir, num_workers):
    """
    This function constructs and trains a model from scratch, without any knowledge transfer method applied. 

    :param int depth: factor for controlling the depth of the model.
    :param int width: factor for controlling the width of the model.
    :param int max_epochs: maximum number of epochs for training the student model.
    :param string state_dict_path: path to save the trained model.
    :param int device: device to use for training the model.
    :param string data_dir: directory to save and load the dataset.
    :param int num_workers: number of workers to use for loading the dataset.
    """

    # Define the device for training the model.
    device = torch.device(device)

    # Get data loaders for the CIFAR-10 dataset.
    train_loader, validation_loader, test_loader = get_cifar10_loaders(
        data_dir, batch_size=BATCH_SIZE, num_workers=num_workers
    )

    # Construct the model to be trained.
    model = WideResidualNetwork(depth=depth, width=width)
    model = model.to(device)

    # Define optimizer and learning rate scheduler.
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=LEARNING_RATE_DECAY_MILESTONES, gamma=LEARNING_RATE_DECAY_FACTOR
    )

    # Construct the loss function to be used for training.
    criterion = torch.nn.CrossEntropyLoss()

    # Define the ignite engines for training and evaluation.
    batch_updater = BatchUpdaterWithoutTransfer(model=model, optimizer=optimizer, criterion=criterion, device=device)
    batch_evaluator = BatchEvaluator(model=model, device=device)
    trainer = Engine(batch_updater)
    evaluator = Engine(batch_evaluator)

    # Define and attach the progress bar, loss metric, and the accuracy metrics.
    attach_pbar_and_metrics(trainer, evaluator)

    # The training engine updates the learning rate schedule at end of each epoch.
    lr_updater = LearningRateUpdater(lr_scheduler=lr_scheduler)
    trainer.on(Events.EPOCH_COMPLETED(every=1))(lr_updater)

    # The training engine logs the training and the evaluation metrics at end of each epoch.
    metric_logger = MetricLogger(evaluator=evaluator, eval_loader=validation_loader)
    trainer.on(Events.EPOCH_COMPLETED(every=1))(metric_logger)

    # Train the model
    trainer.run(train_loader, max_epochs=max_epochs)

    # Save the model to pre-defined path. We move the model to CPU which is desirable as the default device
    # for loading the model.
    model.cpu()
    state_dict_dir = "/".join(state_dict_path.split("/")[:-1])
    os.makedirs(state_dict_dir, exist_ok=True)
    torch.save(model.state_dict(), state_dict_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth", type=int, default=16, help="factor for controlling the depth of the model.")
    parser.add_argument("--width", type=int, default=1, help="factor for controlling the width of the model.")
    parser.add_argument(
        "--max-epochs", type=int, default=200, help="maximum number of epochs for training the trained model."
    )
    parser.add_argument(
        "--state-dict-path", type=str, default="./state_dict/teacher.th", help="path to save the trained model."
    )
    parser.add_argument("--device", type=int, default=0, help="device to use for training the model.")
    parser.add_argument("--data-dir", type=str, default="./data/", help="directory to save and load the dataset.")
    parser.add_argument("--num-workers", type=int, default=4, help="number of workers to use for loading the dataset.")
    args = parser.parse_args()
    main(
        depth=args.depth,
        width=args.width,
        max_epochs=args.max_epochs,
        state_dict_path=args.state_dict_path,
        device=args.device,
        data_dir=args.data_dir,
        num_workers=args.num_workers,
    )
