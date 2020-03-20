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


def Rop(y, x, v):
    """Computes an Rop - J^T v

    Arguments:
    y (torch.tensor): output of differentiated function
    x (torch.tensor): differentiated input
    v (torch.tensor): vector to be multiplied with Jacobian from the right
    from: https://gist.github.com/apaszke/c7257ac04cb8debb82221764f6d117ad
    """
    w = torch.ones_like(y, requires_grad=True)
    return torch.autograd.grad(torch.autograd.grad(y, x, w, create_graph=True), w, v)


def Jacvec(y, x, v):
    """Computes a Jacobian vector product - J v

    Arguments:
    y (torch.tensor): output of differentiated function
    x (torch.tensor): differentiated input
    v (torch.tensor): vector to be multiplied with Jacobian from the left
    """
    return torch.autograd.grad(y, x, v, retain_graph=True)


def flatten(lst):
    """
    Flattens a list or iterable. Note that this chunk allocates more memory.

    Argument:
    lst (list or iteratble): input vector to be flattened

    Returns:
    one dimensional tensor with all elements of lst
    """
    tmp = [i.contiguous().view(-1, 1) for i in lst]
    return torch.cat(tmp).view(-1)


def unflatten_like(vector, likeTensorList):
    """
    Takes a flat torch.tensor and unflattens it to a list of torch.tensors
        shaped like likeTensorList
    Arguments:
    vector (torch.tensor): flat one dimensional tensor
    likeTensorList (list or iterable): list of tensors with same number of ele-
        ments as vector
    """
    outList = []
    i = 0
    for tensor in likeTensorList:
        n = tensor.numel()
        outList.append(vector[i : i + n].view(tensor.shape))
        i += n
    return outList
