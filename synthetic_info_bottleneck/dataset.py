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
import torchvision.transforms as transforms

def dataset_setting(dataset, nSupport):
    """
    Return dataset setting

    :param string dataset: name of dataset
    :param int nSupport: number of support examples
    """
    if dataset == 'miniImageNet':
        mean = [x/255.0 for x in [120.39586422,  115.59361427, 104.54012653]]
        std = [x/255.0 for x in [70.68188272,  68.27635443,  72.54505529]]
        normalize = transforms.Normalize(mean=mean, std=std)
        trainTransform = transforms.Compose([transforms.RandomCrop(80, padding=8),
                                             transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                                             transforms.RandomHorizontalFlip(),
                                             lambda x: np.asarray(x),
                                             transforms.ToTensor(),
                                             normalize
                                            ])

        valTransform = transforms.Compose([transforms.CenterCrop(80),
                                            lambda x: np.asarray(x),
                                            transforms.ToTensor(),
                                            normalize])

        inputW, inputH, nbCls = 80, 80, 64

        trainDir = './data/Mini-ImageNet/train/'
        valDir = './data/Mini-ImageNet/val/'
        testDir = './data/Mini-ImageNet/test/'
        episodeJson = './data/Mini-ImageNet/val1000Episode_5_way_1_shot.json' if nSupport == 1 \
                else './data/Mini-ImageNet/val1000Episode_5_way_5_shot.json'

    elif dataset == 'Cifar':
        mean = [x/255.0 for x in [129.37731888,  124.10583864, 112.47758569]]
        std = [x/255.0 for x in [68.20947949,  65.43124043,  70.45866994]]
        normalize = transforms.Normalize(mean=mean, std=std)
        trainTransform = transforms.Compose([
                                             transforms.RandomCrop(32, padding=4),
                                             transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                                             transforms.RandomHorizontalFlip(),
                                             lambda x: np.asarray(x),
                                             transforms.ToTensor(),
                                             normalize
                                            ])

        valTransform = transforms.Compose([lambda x: np.asarray(x),
                                           transforms.ToTensor(),
                                           normalize])
        inputW, inputH, nbCls = 32, 32, 64

        trainDir = './data/cifar-fs/train/'
        valDir = './data/cifar-fs/val/'
        testDir = './data/cifar-fs/test/'
        episodeJson = './data/cifar-fs/val1000Episode_5_way_1_shot.json' if nSupport == 1 \
                else './data/cifar-fs/val1000Episode_5_way_5_shot.json'

    else:
        raise ValueError('Do not support other datasets yet.')

    return trainTransform, valTransform, inputW, inputH, trainDir, valDir, testDir, episodeJson, nbCls
