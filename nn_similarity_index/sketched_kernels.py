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

import numpy as np
from abc import ABC

class SketchedKernels(ABC):

    r"""

    Compute kernel matrices of individual residual blocks given the sketched feature matrices
    and the sketched gradient matrices.

    """

    def __init__(self, model, loader, imgsize, device, M, T, freq_print, beta=0.5):

        r"""
        Initialise variables
        """
        self.model = model
        self.loader = loader
        self.imgsize = imgsize
        self.device = device
        self.M = M
        self.T = T
        self.freq_print = freq_print
        self.beta = beta


        self.n_samples = 0 # number of data samples in the given dataset
        self.sketched_matrices = {}
        self.kernel_matrices = {}

        # allocate CPU memory for storing sketched feature matrices
        # and sketched gradient matrices
        self.allocate_memory()


    def allocate_memory(self):

        r"""

        Forward a random input to the given neural network to compute the size of the output at each residual block,
        and allocate memory for storing sketched feature matrices and sketched gradient matrices.

        Parameters
        --------
        None

        Returns
        --------
        None

        """

        # create a random input with the same size of the data samples
        # for ImageNet models in PyTorch, the input size is 3 x 224 x 224
        rand_inputs = torch.randn(1, 3, self.imgsize, self.imgsize).to(self.device)
        rand_inputs.requires_grad = True
        _, rand_feats = self.forward_with_gradient_hooks(rand_inputs)

        # allocate CPU memory for the sketched feature matrix and the gradient matrix for each layer
        # the size of the required memory for each layer is 2 x n_buckets x d_output,
        # since each of the sketched feature and gradient matrix requires n_buckets x d_output
        for i in range(len(rand_feats)):
            layer_sizes = np.prod(rand_feats[i].size())
            self.sketched_matrices[i] = torch.zeros(2, self.M, layer_sizes)
            rand_feats[i].data.zero_()

        # remove random features produced from the generated random input
        del rand_feats
        torch.cuda.empty_cache()


    def forward_with_gradient_hooks(self, input_features):

        r"""

        Compute feature vectors by forwarding the data x into a given neural network model,
        also register feature vectors to retain the gradient vectors w.r.t. individual ones.

        The function is adapted from the model definition file provided by PyTorch:
        https://pytorch.org/docs/stable/_modules/torchvision/models/resnet.html

        Parameters
        ---------
        model : PyTorch model instance
            the given neural network model
        input_features : (n_samples, n_channels, height, width) PyTorch tensor
            the input data


        Returns
        --------
        out   : (n_samples, d_output) PyTorch tensor
            the output of the top layer
        feats : (n_ResBlocks) list
            a dictionary that contains the feature vectors produced from individual Residual Blocks


        """
        feats = [input_features]
        out = self.model.conv1(input_features)
        feats.append(out)

        out = self.model.bn1(out)
        out = self.model.relu(out)
        out = self.model.maxpool(out)

        feats.append(out)

        # The residual blocks in a ResNet model are grouped into four stages
        for layer in [self.model.layer1, self.model.layer2, self.model.layer3, self.model.layer4]:
            for mod in layer:
                out = mod(out)
                feats.append(out)

        out = out.mean(dim=(2,3))
        feats.append(out)
        out = self.model.fc(out)

        for feat in feats:
            feat.retain_grad()

        return out, feats


    def cwt_matrix(self, n_rows, n_cols, T):

        r"""

        Generate a matrix S which represents a Clarkson-Woodruff transform in PyTorch according the following reference.

        Clarkson, Kenneth L., and David P. Woodruff. "Low-rank approximation and regression in input sparsity time." Journal of the ACM (JACM) 63.6 (2017): 1-45.


        Parameters
        --------
        n_rows : int
            Number of rows of S
        n_cols : int
            Number of columns of S
        T : int
            Number of nonzeros elements per column


        Returns
        --------
        S : (n_rows, n_cols) PyTorch sparse matrix
            The returned matrix has ``n_cols x T'' nonzeros entries.


        Notes
        --------
        The current version only generates the sparse matrix S with the CPU backend.


        """


        all_rows = []
        all_cols = []
        all_signs = []

        for t in range(T):

            chunk = int(n_rows / T)
            shift = int(t * chunk)

            rows = torch.randint(shift, shift+chunk, (1, n_cols))
            cols = torch.arange(n_cols).view(1,-1)
            signs = torch.randn(n_cols).sign().float()

            all_rows.append(rows)
            all_cols.append(cols)
            all_signs.append(signs)

        rows = torch.cat(all_rows, dim=1)
        cols = torch.cat(all_cols, dim=1)
        pos = torch.cat([rows.long(), cols.long()], dim=0)
        signs = torch.cat(all_signs, dim=0)
        cwt = torch.sparse.FloatTensor(pos, signs, torch.Size([n_rows, n_cols]))
        return cwt


    def cwt_sketching(self, X, S):

        r"""

        Sketch the input matrix X with the sparse random matrix S

        Parameters
        --------
        X : (n_samples, n_dim) dense matrix
            matrix to be sketched
        S : (n_sketches, n_samples) PyTorch sparse matrix
            sparse random matrix

        Returns
        --------
        sketched : (n_sketches, n_dim) dense matrix
            The returned matrix is the sketched matrix which is a summary of the input matrix X


        Notes
        --------
        The current version is only with the CPU backend.

        """
        sketched = torch.sparse.mm(S, X)
        return sketched


    def compute_sketched_mat(self):

        r"""

        Given a PreActResNet model and a dataset, the module computes the sketched matrices
        for feature and gradient vectors respectively generated from individual residual blocks.

        Parameters
        --------
        model : PyTorch model
            A PreActResNet model
        loader : DataLoader
            A iterable DataLoader instance in PyTorch

        Returns
        --------
        No returns

        Notes
        --------
        The module runs efficiently with GPU backend, also runs with CPU backend.

        """

        self.model.eval()
        total = 0

        # Iterate through the data loader in batches:
        for batch_idx, (data, target) in enumerate(self.loader):

            # load a batch of data samples
            data, target = data.to(self.device), target.to(self.device)
            data.requires_grad = True
            output, feats = self.forward_with_gradient_hooks(data)

            # calculate the log-likelihood and the predicted distribution over labels
            logp = torch.log_softmax(output, dim=-1)
            prob = torch.softmax(output, dim=-1)

            # calculate the q distribution which is a smoothed predicted distribution
            q_dist = (prob ** self.beta) / (prob ** self.beta).sum(dim=-1, keepdim=True)

            # zero gradients in the models and calculate the gradients w.r.t. feature maps
            self.model.zero_grad()
            torch.autograd.backward(logp, grad_tensors=q_dist)

            batchsize = data.size(dim=0)
            self.n_samples += batchsize

            with torch.no_grad():

                # generate a CWT matrix
                s = self.cwt_matrix(self.M, batchsize, self.T).to(self.device)

                # calculate the sketched feature vectors and gradient vectors for individual layers
                for layer_id in range(len(feats)):

                    # number of data points x dimension
                    batched_feats = feats[layer_id].data.view(batchsize, -1)
                    batched_grads = feats[layer_id].grad.data.view(batchsize, -1)

                    # sketch the feature vectors into buckets
                    # accumulate buckets on CPU backend
                    sketched_feats  = self.cwt_sketching(batched_feats, s) / (self.T ** 0.5)
                    self.sketched_matrices[layer_id][0] += sketched_feats.cpu()

                    # delete intermediate variables to create memory for the following matrix multiplication
                    del batched_feats
                    torch.cuda.empty_cache()

                    # sketch the gradient vectors into buckets
                    sketched_grads  = self.cwt_sketching(batched_grads, s) / (self.T ** 0.5)
                    self.sketched_matrices[layer_id][1] += sketched_grads.cpu()

                    # delete intermediate variables to create memory for the following matrix multiplication
                    del batched_grads
                    torch.cuda.empty_cache()

                    feats[layer_id].data.zero_()
                    feats[layer_id].grad.data.zero_()

            if batch_idx % self.freq_print == 0:
                print("finished {:d}/{:d}".format(batch_idx, len(self.loader)))


    def compute_sketched_kernels(self):

        r"""
        Compute the sketched feature matrices and sketched gradient matrices first,
        and then calculate the kernel matrices for individual layers.

        Parameters
        --------
        None

        Returns
        --------
        No returns

        """

        self.compute_sketched_mat()

        for layer_id in range(len(self.sketched_matrices)):
            temp = torch.bmm(self.sketched_matrices[layer_id], self.sketched_matrices[layer_id].transpose(1,2))
            self.kernel_matrices[layer_id] = (temp[0] * temp[1]).numpy()
            del self.sketched_matrices[layer_id]
            torch.cuda.empty_cache()
