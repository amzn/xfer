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

from finite_ntk.lazy import FVPR_FD


class TestFisherRFD(unittest.TestCase):
    def test_fisher_regression_fd(self):
        model = torch.nn.Sequential(torch.nn.Linear(7500, 1, bias=False))

        data = torch.randn(50, 7500)

        # closed form fisher information for linear regression
        fim = data.t() @ data / data.size(0)

        eps_list = [1.0, 1e-1, 5e-2, 1e-2, 1e-3, 5e-3, 1e-4, 5e-4, 1e-5, 5e-6, 1e-6, 5e-6]

        for epsilon in eps_list:
            fvp = FVPR_FD(model, data, epsilon=epsilon)
            fvp_approx = fvp._matmul(torch.eye(data.shape[-1]))
            rel_error = torch.norm(fvp_approx - fim) / torch.norm(fim)

            # check to ensure that our relative error is never greater than 4%
            self.assertLess(rel_error.item(), 4e-2)

    def test_fisher_matrix_matrix_matmul(self):
        model = torch.nn.Sequential(
            torch.nn.Linear(1, 400),
            torch.nn.ELU(),
            torch.nn.Linear(400, 400),
            torch.nn.ELU(),
            torch.nn.Linear(400, 1),
        )

        data = torch.randn(1500, 1)

        fvp = FVPR_FD(model, data)

        numpars = 0
        for p in model.parameters():
            numpars += p.numel()

        orthmat, _ = torch.qr(torch.randn(numpars, 80))
        emat = 1e-2 * torch.randn(80, 2)

        full_matmul = fvp.matmul(orthmat @ emat)
        split_matmul = fvp.matmul(orthmat) @ emat

        # check that F (Vy) = FV y
        self.assertLess(
            torch.norm(full_matmul - split_matmul) / split_matmul.norm(), 1e-2
        )

        # check that matrix columns work
        self.assertLess(
            torch.norm(full_matmul[:, 0] - fvp.matmul(orthmat @ emat[:, 0])), 1e-5
        )


if __name__ == "__main__":
    # lock down the seed
    torch.random.manual_seed(2019)
    unittest.main()
