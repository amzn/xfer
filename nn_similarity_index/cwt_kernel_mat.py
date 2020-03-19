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

import os
os.environ["OMP_NUM_THREADS"] = "1"

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

import numpy as np
from abc import ABC
import os
import argparse
from sketched_kernels import SketchedKernels

from utils import *


if __name__ == "__main__":

    # Get arguments from the command line
    parser = argparse.ArgumentParser(description='PyTorch CWT sketching kernel matrices')

    parser.add_argument('--datapath', type=str,
                                help='absolute path to the dataset')
    parser.add_argument('--modelname', type=str,
                                help='model name')
    parser.add_argument('--pretrained', action='store_true',
                                help='whether to load a pretrained ImageNet model')

    parser.add_argument('--seed', default=0, type=int,
                                help='random seed for sketching')
    parser.add_argument('--task', default='cifar10', type=str, choices=['cifar10', 'cifar100', 'svhn', 'stl10'],
                                help='the name of the dataset, cifar10 or cifar100 or svhn or stl10')
    parser.add_argument('--split', default='train', type=str,
                                help='split of the dataset, train or test')
    parser.add_argument('--bsize', default=512, type=int,
                                help='batch size for computing the kernel')

    parser.add_argument('--M', '--num-buckets-sketching', default=512, type=int,
                                help='number of buckets in Sketching')
    parser.add_argument('--T', '--num-buckets-per-sample', default=1, type=int,
                                help='number of buckets each data sample is sketched to')

    parser.add_argument('--freq_print', default=10, type=int,
                                help='frequency for printing the progress')

    args = parser.parse_args()

    # Set the backend and the random seed for running our code
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(args.seed)
    if device == 'cuda':
        cudnn.benchmark = True
        torch.cuda.manual_seed(args.seed)

    # The size of images for training and testing ImageNet models
    imgsize = 224

    # Generate a dataloader that iteratively reads data
    # Load a model, either pretrained or not
    loader = load_dataset(args.task, args.split, args.bsize, args.datapath, imgsize)
    net = load_model(device, args.modelname, pretrained=True)

    # Set the model to be in the evaluation mode. VERY IMPORTANT!
    # This step to fix the running statistics in batchnorm layers,
    # and disable dropout layers
    net.eval()

    csm = SketchedKernels(net, loader, imgsize, device, args.M, args.T, args.freq_print)
    csm.compute_sketched_kernels()

    # Compute sketched kernel matrices for each layer
    for layer_id in range(len(csm.kernel_matrices)):
        nkme = (csm.kernel_matrices[layer_id].sum() ** 0.5) / csm.n_samples
        print("The norm of the kernel mean embedding of layer {:d} is {:.4f}".format(layer_id, nkme))

    del net, loader
    torch.cuda.empty_cache()

    # Save the sketched kernel matrices
    savepath = 'sketched_kernel_mat/'
    if not os.path.isdir(savepath):
        os.mkdir(savepath)

    save_filename = '{}_{}_{}_{}.npy'.format(args.modelname, args.split, args.task, args.seed)
    np.save(savepath + save_filename, csm.kernel_matrices)
