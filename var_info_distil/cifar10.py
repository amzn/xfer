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
import numpy as np
import torch
import torchvision
import torchvision.transforms as T

CIFAR10_TRAIN_SIZE = 40000
CIFAR10_VALIDATION_SIZE = 10000
CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR10_AVERAGE = [0.247, 0.243, 0.261]
CIFAR10_PAD_SIZE = 4
CIFAR10_CROP_SIZE = 32

def get_cifar10_loaders(root, batch_size, num_workers):
    """
    This function prepares the DataLoader class for training, validation and testing of the CIFAR-10 dataset. The data 
    loaders will prepare the mini-batch of images and labels when training the neural networks. Especially, whenever the
    images are selected for the mini-batch, they are randomly transformed on-line by the pre-defined operations. 
    Specifically, data loaders are equipped with the image augmentation schemes, i.e., normalization, padding, random 
    flipping, and random cropping, that were used by Zagoruyko et al. (https://arxiv.org/abs/1605.07146) when 
    training the wide residual networks for the CIFAR-10 dataset. 
    
    :param string root: the root directory for saving and loading the CIFAR-10 dataset.
    :param int batch_size: the size of mini-batchs to sample from the data loaders when training the neural network.
    :param int num_workers: number of parallel CPU workers to use for loading the CIFAR-10 dataset.
    """
    # For evaluation, we normalize the mean and the variance of the dataset to be 0 and 1, respectively.
    # Specifically, (0.4914, 0.4822, 0.4465) and (0.247, 0.243, 0.261) are the mean and the standard deviation computed
    # from the raw CIFAR-10 dataset.
    eval_transform = T.Compose([T.ToTensor(), T.Normalize(np.array(CIFAR10_MEAN), np.array(CIFAR10_AVERAGE))])
    # For training, we (a) pad the images with four pixels on each side, (b) randomly flip the image in horizontal way,
    # (c) randomly crop the image to have size of 32x32 and (d) normalize images as in evaluation.
    train_transform = T.Compose(
        [
            T.Pad(CIFAR10_PAD_SIZE, padding_mode="reflect"),
            T.RandomHorizontalFlip(),
            T.RandomCrop(CIFAR10_CROP_SIZE),
            eval_transform,
        ]
    )

    # Make training dataset (for updating weight of neural networks) and validation dataset (for choosing the
    # hyper-parameters of training). To this end, we need to split the "train" dataset (pre-defined in torchvision)
    # into two parts, with sizes of CIFAR10_TRAIN_SIZE and CIFAR10_VALIDATION_SIZE for training and validataion dataset,
    # respectively.

    train_and_validation_dataset_with_train_transform = torchvision.datasets.CIFAR10(
        root=root, train=True, download=True, transform=train_transform
    )
    train_and_validation_dataset_with_eval_transform = torchvision.datasets.CIFAR10(
        root=root, train=True, download=False, transform=eval_transform
    )

    train_and_validation_indices = torch.randperm(len(train_and_validation_dataset_with_train_transform))
    train_indices = train_and_validation_indices[:CIFAR10_TRAIN_SIZE]
    validation_indices = train_and_validation_indices[CIFAR10_TRAIN_SIZE : CIFAR10_TRAIN_SIZE + CIFAR10_VALIDATION_SIZE]

    train_dataset = torch.utils.data.Subset(train_and_validation_dataset_with_train_transform, train_indices)
    validation_dataset = torch.utils.data.Subset(train_and_validation_dataset_with_eval_transform, validation_indices)

    # Make test dataset
    test_dataset = torchvision.datasets.CIFAR10(root=root, train=False, download=False, transform=eval_transform)

    # Make data loaders
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    validation_loader = torch.utils.data.DataLoader(
        dataset=validation_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, validation_loader, test_loader
