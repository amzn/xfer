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
import math


def gen_reg_task(numdata, seed=None, split=0.0, input_dict=None, extrapolate=False):
    r"""
    gen_reg_task generates data from the same data-generating process as for the 
    sine curves in Bayesian MAML (https://arxiv.org/pdf/1806.03836.pdf). we also
    include a couple of extra options for extrapolation and incorporation of linear
    regression functions (detailed below)
    
    numdata (int): number of data points
    seed (int): for reproducibility
    split (float between 0 and 1): determine the probability of generating a given 
                                linear function
    input_dict (dict - three value): dict for locking down parameters 
                                    (useful for regenerating plots)
    extrapolate (True/False): whether the train data should be U(-5,5) or on a grid
    [-6.5, 6.5] (and we should test extrapolation)
    """

    if not extrapolate:
        train_x = 10.0 * torch.rand(numdata) - 5.0  # U(-5.0, 5.0)
        train_x = torch.sort(train_x)[0]
    else:
        train_x = torch.linspace(-6.5, 6.5, numdata)

    if seed is not None:
        torch.random.manual_seed(seed)

    if torch.rand(1) > split:
        if input_dict is None:
            # same setup as in Bayesian MAML:
            A = 4.9 * torch.rand(1) + 0.1  # should be random on [0.1, 5.0]
            b = 2 * math.pi * torch.rand(1)  # U(0, 2 pi)
            w = 1.5 * torch.rand(1) + 0.5  # U(0.5, 2.0)

            train_y = A * torch.sin(w * (train_x) + b) + (0.01 * A) * torch.randn_like(
                train_x
            )
        else:
            A = input_dict["A"]
            b = input_dict["b"]
            w = input_dict["w"]

            train_y = A * torch.sin(w * (train_x) + b)

    else:
        if input_dict is None:
            A = 6 * torch.rand(1) - 3.0
            b = 6 * torch.rand(1) - 3.0
            w = None

            train_y = A * train_x + b + (0.3) * torch.randn_like(train_x)

        else:
            A = input_dict["A"]
            b = input_dict["b"]
            w = input_dict["w"]
            train_y = A * train_x + b

    return train_x.unsqueeze(-1), train_y.unsqueeze(-1), {"A": A, "b": b, "w": w}


def parse_and_generate_tasks():
    r"""
    this is the argparser and data generator for the simple regression scripts.
    we first parse arguments (note that fisher is unused for ablr). we then loop 
    through the required number of tasks generating datasets.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--nx",
        help="(int) number of data points for simulated data",
        default=10,
        type=int,
    )
    parser.add_argument("--ntasks", help="(int) number of tasks", default=5, type=int)
    parser.add_argument(
        "--fisher", action="store_true", help="fisher basis usage flag (default: off)"
    )
    parser.add_argument("--seed", help="random seed", type=int, default=10)
    parser.add_argument("--output_file", type=str)
    parser.add_argument("--model_output_file", type=str, default=None)
    parser.add_argument(
        "--transfer_fn",
        action="store_true",
        help="whether to transfer from linear to sine",
    )
    args = parser.parse_args()

    torch.random.manual_seed(args.seed)
    # task setup - for cl args
    tasks = args.ntasks
    datapoints = args.nx

    if args.transfer_fn:
        transfer_split = 0.5
    else:
        transfer_split = 0.0

    task_dataset = []
    for _ in range(args.ntasks):
        # generate data
        task_dataset.append(gen_reg_task(datapoints, split=transfer_split))

    return task_dataset
