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

import numpy as np
import argparse
from sim_indices import SimIndex


if __name__ == "__main__":

    # Get arguments from the command line
    parser = argparse.ArgumentParser(description='PyTorch CWT sketching kernel matrices')

    parser.add_argument('--loadpath', type=str,
                                help='absolute path to the folder that contains the file')
    parser.add_argument('--filename1', type=str,
                                help='absolute path to the file that contains kernel matrices')
    parser.add_argument('--filename2', type=str, default=None,
                                help='absolute path to the file that contains kernel matrices')
    parser.add_argument('--simindex', type=str, choices=['euclidean', 'cka', 'nbs'], default='cka',
                                help='similarity index to use in computing the scores')

    args = parser.parse_args()

    # load the file that contains kernel matrices of individual residual blocks
    kernel_matrices_1 = np.load(args.loadpath + args.filename1, allow_pickle=True).item()
    kernel_matrices_2 = np.load(args.loadpath + args.filename2, allow_pickle=True).item() if args.filename2 else kernel_matrices_1

    n_resblocks_1 = len(kernel_matrices_1)
    n_resblocks_2 = len(kernel_matrices_2)
    sim_scores = np.zeros((n_resblocks_1, n_resblocks_2))

    simindices = SimIndex()
    func_ = getattr(simindices, args.simindex)

    for layer_id1 in range(n_resblocks_1):
        for layer_id2 in range(n_resblocks_2):
            sim_scores[layer_id1, layer_id2] = func_(kernel_matrices_1[layer_id1], kernel_matrices_2[layer_id2])

    np.save(args.loadpath + 'heatmap.npy', sim_scores)
