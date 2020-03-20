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

from finite_ntk.lazy import NeuralTangent
from finite_ntk.lazy.ntk_lazytensor import Jacvec, Rop


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


class TestNeuralTangent(unittest.TestCase):
    def test_matmul(self, seed=2019):
        torch.random.manual_seed(seed)

        model = torch.nn.Sequential(torch.nn.Linear(30, 1, bias=True))

        input_data = torch.randn(50, 30)

        output = model(input_data)

        rhs = torch.ones(50, 1)
        z = Jacvec(output, model.parameters(), rhs)

        ntk_vec = Rop(output, model.parameters(), z)[0]

        # this checks the size is what we'd expect from a Jv and a J^T v
        self.assertEqual(ntk_vec.size(), torch.Size((50, 1)))

        ntk = NeuralTangent(model, data=input_data)
        kernel_output = ntk.matmul(rhs)

        self.assertLess(torch.norm(kernel_output - ntk_vec) / torch.norm(ntk_vec), 1e-5)

    def test_get_item_2d(self, seed=2019):
        # first 2d setting
        model = torch.nn.Sequential(torch.nn.Linear(30, 1, bias=True))

        input_data = torch.randn(50, 30)

        ntk = NeuralTangent(model, data=input_data)

        # check shape is what we think it is
        self.assertEqual(ntk.shape, torch.Size([50, 50]))

        # check that _getitem returns outputs as expected for a single output
        self.assertEqual(ntk[10:12, 14:18].shape, torch.Size([2, 4]))

    def test_get_item_4d(self, seed=2019):
        model = SimpleConvModel()
        input_data = torch.randn(64, 3, 32, 32)  # cifar sized image

        ntk = NeuralTangent(model, data=input_data)

        self.assertEqual(ntk.shape, torch.Size([64, 64]))

        # check that _getitem returns outputs as expected for a single output
        self.assertEqual(ntk[10:12, 14:18].shape, torch.Size([2, 4]))

        model_multioutput = SimpleConvModel(linear=False)
        ntk_mo = NeuralTangent(model_multioutput, data=input_data, num_outputs=1024)
        self.assertEqual(ntk_mo.shape, torch.Size([1024, 64, 64]))

        # subset a small bit
        subset = ntk_mo[10:15, 10:12, 15:19]
        self.assertEqual(subset.shape, torch.Size([5, 2, 4]))

        # ensure that we can evaluate
        eval_subset = subset.evaluate()
        self.assertGreater(eval_subset.norm(), 0.0)

    def test_approx_diag_multi_output(self, seed=2019):
        input_data = torch.randn(64, 3, 32, 32)  # cifar sized image
        model_multioutput = SimpleConvModel(linear=False)
        ntk_mo = NeuralTangent(model_multioutput, data=input_data, num_outputs=1024)
        approx_diag = ntk_mo._approx_diag()

        true_diag = ntk_mo.diag()

        self.assertEqual(approx_diag.shape, true_diag.shape)

        # should be nearly an upper bound??
        approx_diag[approx_diag < 1e-3] = 0.0
        true_diag[true_diag < 1e-3] = 0.0
        self.assertGreater(
            torch.sum(approx_diag >= true_diag), 0.99 * approx_diag.numel()
        )

    def test_fourd_symmetry(self, seed=2019):
        input_data = torch.randn(4, 3, 32, 32)
        model = SimpleConvModel(linear=True, nout=3)

        ntk_mo = NeuralTangent(model, data=input_data, num_outputs=3)
        ntk_mat = ntk_mo.evaluate()

        ntk_mat_transposed = ntk_mat.permute(0, 2, 1)
        self.assertLess(torch.norm(ntk_mat - ntk_mat_transposed), 1e-5)

    def test_detachment_and_squeezing(self):
        input_data = torch.randn(4, 3, 32, 32)
        model = SimpleConvModel(linear=True, nout=3)

        ntk_mo = NeuralTangent(model, data=input_data, num_outputs=1, keep_outputs=True)

        self.assertEqual(ntk_mo.shape, torch.Size([1, 4, 4]))
        try:
            ntk_mo.detach_()
        except:
            self.assertAlmostEqual(1, 0)


if __name__ == "__main__":
    if torch.cuda.is_available():
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
    unittest.main()
