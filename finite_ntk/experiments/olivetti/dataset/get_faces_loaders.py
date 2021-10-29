# Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
import numpy as np


def get_faces_loaders(batch_size=128, test=True, data_path="./data/"):
    """
    returns the train (and test if selected) loaders for the olivetti
    rotated faces dataset
    """

    dat = np.load(data_path + "rotated_faces_data.npz")
    train_images = torch.FloatTensor(dat['train_images'])
    train_targets = torch.FloatTensor(dat['train_angles'])

    traindata = torch.utils.data.TensorDataset(train_images, train_targets)
    trainloader = torch.utils.data.DataLoader(traindata, batch_size=batch_size,
                                              shuffle=True)

    if test:
        test_images = torch.FloatTensor(dat['test_images'])
        test_targets = torch.FloatTensor(dat['test_angles'])

        testdata = torch.utils.data.TensorDataset(test_images, test_targets)
        testloader = torch.utils.data.DataLoader(testdata, batch_size=batch_size)

        return trainloader, testloader

    return trainloader
