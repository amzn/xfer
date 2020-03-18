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
from abc import ABC
import os
import argparse


class SimIndex(ABC):

    r"""

    The class that supports three similarity indices.

    Notes:

    Currently supports Euclidean distance, Centred Kernel Alignment
    and Normalised Bures Similarity between two kernel matrices.

    """

    def __init__(self):
        ...

    def centering(self, kmat):
        r"""
        Centering the kernel matrix
        """
        return kmat - kmat.mean(axis=0, keepdims=True) - kmat.mean(axis=1, keepdims=True) + kmat.mean()

    def euclidean(self, kmat_1, kmat_2):
        r"""
        Compute the Euclidean distance between two kernel matrices
        """
        return np.linalg.norm(kmat_1 - kmat_2)

    def cka(self, kmat_1, kmat_2):
        r"""
        Compute the Centred Kernel Alignment between two kernel matrices.
        \rho(K_1, K_2) = \Tr (K_1 @ K_2) / ||K_1||_F / ||K_2||_F
        """
        kmat_1 = self.centering(kmat_1)
        kmat_2 = self.centering(kmat_2)
        return np.trace(kmat_1 @ kmat_2) / np.linalg.norm(kmat_1) / np.linalg.norm(kmat_2)

    def nbs(self, kmat_1, kmat_2):
        r"""
        Compute the Normalised Bures Similarity between two kernel matrices.
        \rho(K_1, K_2) = \Tr( (K_1^{1/2} @ K_2 @ K_1^{1/2})^{1/2} ) / \Tr(K_1) /  \Tr(K_2)
        """
        kmat_1 = self.centering(kmat_1)
        kmat_2 = self.centering(kmat_2)
        return sum(np.real(np.linalg.eigvals(kmat_1 @ kmat_2)).clip(0.) ** 0.5) / ((np.trace(kmat_1) * np.trace(kmat_2)) ** 0.5)
