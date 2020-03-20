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
# This code is a small data loader function for our cifar experiments.

import os
import numpy as np
import torch
import torchvision
import enum


class DatasetTypes(enum.Enum):
    stl10 = 'STL10'

def generate_data(args, model_cfg):
    print("Loading dataset {} from {}".format(args.dataset, args.data_path))
    dataset = getattr(torchvision.datasets, args.dataset)
    path = os.path.join(args.data_path, args.dataset.lower())

    if args.dataset == DatasetTypes.stl10:
        train_set = dataset(
            root=path, split="train", download=False, transform=model_cfg.transform_train
        )
        num_classes = 10
        # this is a manual mapping of STL10 classes to CIFAR10 classes
        # CIFAR10 classes: {automobile, bird, cat, deer, dog, frog, horse, ship}
        # STL10 classes: {airplane, bird, car, cat, deer, dog, horse, monkey, ship, truck}
        cls_mapping = np.array([0, 2, 1, 3, 4, 5, 7, 6, 8, 9])
        train_set.labels = cls_mapping[train_set.labels]

        test_set = dataset(
            root=path, split="test", download=False, transform=model_cfg.transform_test
        )
        test_set.labels = cls_mapping[test_set.labels]
    else:
        train_set = dataset(
            root=path, train=True, download=False, transform=model_cfg.transform_train
        )
        # zero indexing so the max target is one less than the number of classes
        num_classes = max(train_set.targets) + 1

        test_set = dataset(
            root=path, train=False, download=False, transform=model_cfg.transform_test
        )

    loaders = {
        "train": torch.utils.data.DataLoader(
            train_set,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
        ),
        "test": torch.utils.data.DataLoader(
            test_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        ),
    }
    num_data = len(loaders["train"].dataset)
    print("Number of data points: ", num_data)
    return loaders, num_classes, num_data