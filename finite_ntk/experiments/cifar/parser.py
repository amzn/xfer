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

import argparse


def parser():
    parser = argparse.ArgumentParser(description="fast adaptation training")

    parser.add_argument(
        "--dataset", type=str, default="CIFAR10", help="dataset name (default: CIFAR10). Other options include CIFAR100 and STL10"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        required=True,
        metavar="path",
        help="path to datasets location (default: None)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        metavar="N",
        help="input batch size (default: 128)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        metavar="N",
        help="number of workers (default: 4)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="PreResNet56",
        required=True,
        metavar="MODEL",
        help="model name (default: PreResNet56)",
    )

    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        metavar="CKPT",
        help="checkpoint to resume training from (default: None)",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        metavar="N",
        help="number of epochs to train (default: 10)",
    )
    parser.add_argument(
        "--save_freq",
        type=int,
        default=25,
        metavar="N",
        help="save frequency (default: 25)",
    )
    parser.add_argument(
        "--eval_freq",
        type=int,
        default=5,
        metavar="N",
        help="evaluation frequency (default: 5)",
    )
    parser.add_argument(
        "--lr_init",
        type=float,
        default=0.1,
        metavar="LR",
        help="initial learning rate (default: 0.1)",
    )
    parser.add_argument(
        "--wd", type=float, default=1e-4, help="weight decay (default: 1e-4)"
    )
    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default=None,
        required=False,
        help="path to npz results file",
    )
    parser.add_argument(
        "--bias",
        action="store_true",
        help="whether to use bias terms in the loss (default: off)",
    )
    parser.add_argument(
        "--inference",
        type=str,
        choices=["laplace", "vi", "map"],
        required=True,
        default="map",
        help="inference choice to use",
    )
    args = parser.parse_args()

    return args
