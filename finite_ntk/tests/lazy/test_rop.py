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
import unittest

import time

from finite_ntk.lazy import utils


class SimpleConvModel(torch.nn.Module):
    def __init__(self, linear=True, nout=1):
        super(SimpleConvModel, self).__init__()
        self.conv = torch.nn.Conv2d(3, 1, 1)
        if linear:
            self.linear = torch.nn.Linear(1024, nout)
        else:
            self.linear = lambda x: x

    def forward(self, x):
        out = self.conv(x).view(x.size(0), -1)
        return self.linear(out)


class TestRop(unittest.TestCase):
    def test_jacobian_vector_product(self, seed=2019):
        torch.random.manual_seed(seed)
        input = torch.randn(64, 3, 32, 32)
        model = SimpleConvModel(nout=2)

        model_pars = 0
        for p in model.parameters():
            model_pars += p.numel()

        loss = (model(input) - torch.randn(64, 2)) ** 2
        # loss = loss.sum(-2)

        # Jv returns a list that should have the same number of elements
        # as the number of parameters
        result = utils.Jacvec(loss, model.parameters(), torch.ones(64, 2))
        print(result[0].shape)
        result_elements = utils.flatten(result).numel()

        self.assertEqual(result_elements, model_pars)

    def test_rop(self, seed=2019):
        torch.random.manual_seed(seed)
        input = torch.randn(64, 3, 32, 32)
        model = SimpleConvModel(nout=2)

        model_pars = 0
        for p in model.parameters():
            model_pars += p.numel()

        loss = (model(input) - torch.randn(64, 2)) ** 2

        ones = torch.randn(model_pars, 1)
        vec = utils.unflatten_like(ones, model.parameters())
        # J^T v returns a vector
        result = utils.Rop(loss, model.parameters(), vec)
        self.assertEqual(result[0].shape, (64, 2))


if __name__ == "__main__":
    if torch.cuda.is_available():
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
    unittest.main()
