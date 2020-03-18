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
import torch.nn as nn
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

from enum import Enum

class DatasetsNames(Enum):
    cifar10  = lambda x: {"train" : True if x == "train" else False}
    cifar100 = lambda x: {"train" : True if x == "train" else False}
    svhn     = lambda x: x
    stl10    = lambda x: x


def load_model(device, modelname, pretrained=True):

    r"""

    Load an ImageNet model in PyTorch.


    Parameters
    --------
    device : str
        the chosen hardware backend for the model
    pretrained : boolean
        whether to load a pretrained model or not

    Returns
    --------
    loader : PyTorch Model instance
        The returned instance is an PyTorch model.

    Notes
    --------
    The current version supports resnet-based models,
    including resnet, wide_resnet, and resnext models.

    """

    model = getattr(models, modelname)(pretrained=pretrained).to(device)

    return model


def load_dataset(name, split, batchsize, datapath, imgsize):

    r"""

    Produce a DataLoader instance in PyTorch.


    Parameters
    --------
    name : str
        the name of the dataset
    train : boolean
        whether the split required is the training or the test
    transform : PyTorch transforms instance
        process of preprocessing data

    Returns
    --------
    loader : PyTorch DataLoader instance
        The returned instance is an iterable PyTorch DataLoader instance
        for the given dataset and the split.

    Notes
    --------
    The current version supports cifar10, cifar100, svhn and stl10.

    """

    # standard preprocessing steps for imagenet models

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(imgsize),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    transform_test = transforms.Compose([
        transforms.Resize((imgsize, imgsize)),
        transforms.ToTensor(),
        normalize,
    ])

    # transform = transform_train if train else transform_test
    # for our purpose, we only consider the testing case.
    transform = transform_test

    func_ = getattr(torchvision.datasets, name.upper())

    kws = {
            "root": datapath,
            "download": True,
            "transform": transform,
    }

    kws = {**kws, **getattr(DatasetsNames, name)(split)}

    dataset = func_(**kws)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batchsize, shuffle=False, num_workers=0)

    return loader
