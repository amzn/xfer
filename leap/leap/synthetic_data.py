# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
import random
import numpy as np
import matplotlib.pyplot as plt

from mxnet.gluon.data import ArrayDataset
import mxnet

from .data import MetaTaskDataContainer, TaskDataContainer
from .config import DEFAULT_CONFIG_SYNTHETIC


class MetaTaskSynthetic(MetaTaskDataContainer):
    def __init__(self, config=None, weights=None, bias=None, seed=1, context=None):

        """
        :param config: If None, DEFAULT_CONFIG_SYNTHETIC is loaded.
        :param weights: Tasks' weights matrix. Row k corresponds to the weight parameters of task k. If None, w is
            sampled from a N(0,1).
        :param bias: Tasks' biases vector. Row k corresponds to the bias parameters of task k. If None, w is sampled
            from a N(0,1).
        :param seed: Seed for random generator.
        """

        if config is None:
            config = DEFAULT_CONFIG_SYNTHETIC

        self.config = config
        self.weights = weights
        self.bias = bias

        if context is None:
            context = mxnet.cpu()
        self.context = context

        self.seed = seed
        random.seed(self.seed)

        num_tasks_train = config["num_tasks_train"]
        num_tasks_test = config["num_tasks_test"]
        num_tasks_val = config["num_tasks_val"]
        num_tasks = num_tasks_train + num_tasks_test + num_tasks_val

        self.num_tasks = num_tasks

        self._generate_parameters()
        self._validate_parameters()

        num_examples = config["num_examples_per_task"]
        std_x = config["std_x"]
        hold_out = config["hold_out"]
        noise = config["std_noise"]

        # Generate the training/test/val dataset.
        # Each dataset is a list of TaskSynthetic objects (one per task)
        data_train = [TaskSynthetic(self.weights[t, :], self.bias[t], num_examples, std_x, noise, hold_out,
                                    context=context)
                      for t in np.arange(0, num_tasks_train)]
        data_test = [TaskSynthetic(self.weights[t, :], self.bias[t], num_examples, std_x, noise, hold_out,
                                   context=context)
                     for t in np.arange(num_tasks_train, num_tasks_train + num_tasks_test)]
        data_val = [TaskSynthetic(self.weights[t, :], self.bias[t], num_examples, std_x, noise, hold_out,
                                  context=context)
                    for t in np.arange(num_tasks_train + num_tasks_test, num_tasks)]

        super().__init__(data_train, data_test, data_val, context=context)

    def plot_sample(self, root="./sample_synth"):

        """Plot N images from each alphabet and store the images in root."""

        if self.weights.shape[1] != 2:
            raise ValueError("Only 2D datasets can be plot.")

        if not os.path.exists(root):
            os.makedirs(root)

        fig_train = self._plot([dd._train_dataset for dd in self.train_tasks],
                               "Training Samples for Training Tasks")
        fig_train.savefig(os.path.join(root, "sample_train_train_tasks.png"))
        del fig_train
        fig_test = self._plot([dd._train_dataset for dd in self.test_tasks],
                              "Training Samples for Test Tasks")
        fig_test.savefig(os.path.join(root, "sample_train_test_tasks.png"))
        del fig_test
        fig_val = self._plot([dd._train_dataset for dd in self.val_tasks],
                             "Training Samples for Validation Tasks")
        fig_val.savefig(os.path.join(root, "sample_train_val_tasks.png"))
        del fig_val

        if self.config["hold_out"] > 0:
            fig_train = self._plot([dd._val_dataset for dd in self.train_tasks],
                                   "Validation Samples for Training Tasks")
            fig_train.savefig(os.path.join(root, "sample_val_train_tasks.png"))
            del fig_train
            fig_test = self._plot([dd._val_dataset for dd in self.test_tasks],
                                  "Validation Samples for Test Tasks")
            fig_test.savefig(os.path.join(root, "sample_val_test_tasks.png"))
            del fig_test
            fig_val = self._plot([dd._val_dataset for dd in self.val_tasks],
                                 "Validation Samples for Validation Tasks")
            fig_val.savefig(os.path.join(root, "sample_val_val_tasks.png"))
            del fig_val

    def _plot(self, data, title):

        """Helper function for plotting."""

        num_tasks = len(data)
        fig, ax = plt.subplots(1, num_tasks, figsize=(num_tasks*5, 5))
        fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5)
        for mm in range(num_tasks):
            X, y = data[mm][:]
            X = X.asnumpy()
            y = y.asnumpy()
            ax[mm].scatter(X[:, 0], X[:, 1], c=y.flatten())
        fig.suptitle(title, size=18)
        return fig

    def _validate_parameters(self):
        if self.weights.shape[0] != self.num_tasks:
            raise ValueError("Number of rows in w must be equal to the total number of tasks")

        if len(self.bias) != self.num_tasks:
            raise ValueError("Length of b must be equal to the total number of tasks")

    def _generate_parameters(self):
        if self.weights is None:
            dim = self.config["dim"]
            self.weights = self.config["global_bias"] + mxnet.nd.random_normal(shape=(self.num_tasks, dim),
                                                                               ctx=self.context)

        if self.bias is None:
            if self.config["task_bias"]:
                self.bias = mxnet.nd.random_normal(shape=self.num_tasks, ctx=self.context)
            else:
                self.bias = mxnet.nd.zeros(num_tasks, ctx=self.context)


class TaskSynthetic(TaskDataContainer):

    """
    Synthetic Task Container: Linear Regression.
    """

    def __init__(self, w, b, num_examples, std_x, noise, hold_out=None, seed=None, context=None):

        """
        :param w: Task's weights vector.
        :param b: Task's bias.
        :param num_examples: Total number of examples per task.
        :param std_x: The covariates are sampled from a zero mean normal distribution with
            standard deviation equal to std_x.
        :param hold_out: Number of examples to hold out for validation
        :param seed: seed for the random generator
        """

        self.w = w
        self.b = b
        self.num_examples = num_examples
        self.seed = seed

        if context is None:
            context = mxnet.cpu()
        self.context = context

        if seed:
            random.seed(seed)
        if hold_out and hold_out < num_examples:
            Xtr, Ytr = self._real_fn(std_x * mxnet.nd.random_normal(shape=(num_examples - hold_out, len(w)),
                                     ctx=context), noise)
            train_dataset = ArrayDataset(Xtr, Ytr)
            Xval, Yval = self._real_fn(std_x * mxnet.nd.random_normal(shape=(hold_out, len(w)), ctx=context), noise)
            val_dataset = ArrayDataset(Xval, Yval)
        else:
            Xtr, Ytr = self._real_fn(std_x * mxnet.nd.random_normal(shape=(num_examples, len(w)), ctx=context), noise)
            train_dataset = ArrayDataset(Xtr, Ytr)
            val_dataset = None

        super().__init__(train_dataset, val_dataset, context=context)

    def _real_fn(self, X, noise):
        y = mxnet.nd.dot(X, mxnet.nd.expand_dims(self.w, axis=1)) + self.b
        if noise > 0.0:
            y += mxnet.nd.expand_dims(noise * mxnet.nd.random_normal(shape=(X.shape[0],)), axis=1)
        return X, y


if __name__ == '__main__':

    s1 = MetaTaskSynthetic()
    s1.plot_sample()

    batch_size = 20
    train_tasks = s1.train_tasks

    assert len(s1.train_tasks) == 3
    for task in train_tasks:
        tr_iterator = task.get_train_iterator(batch_size)
        for data in tr_iterator:
            assert (data[0].shape == (batch_size, 2))
            assert (data[1].shape == (batch_size, 1))
            assert (data[1].asnumpy().dtype == np.float32)
            break
        val_iterator = task.get_val_iterator(batch_size)
        for data in val_iterator:
            assert (data[0].shape == (batch_size, 2))
            assert (data[1].shape == (batch_size, 1))
            assert (data[1].asnumpy().dtype == np.float32)
            break

    dim = 2
    num_tasks = 15
    w = mxnet.nd.random_normal(shape=(num_tasks, dim))
    b = mxnet.nd.random_normal(shape=num_tasks)

    s2 = MetaTaskSynthetic(weights=w, bias=b)
    s2.plot_sample(root="./sample_synth_w_b_given")

    batch_size = 20
    train_tasks = s2.train_tasks

    assert len(train_tasks) == 3
    for task in train_tasks:
        tr_iterator = task.get_train_iterator(batch_size)
        for data in tr_iterator:
            assert (data[0].shape == (batch_size, 2))
            assert (data[1].shape == (batch_size, 1))
            assert (data[1].asnumpy().dtype == np.float32)
            break
        val_iterator = task.get_val_iterator(batch_size)
        for data in val_iterator:
            assert (data[0].shape == (batch_size, 2))
            assert (data[1].shape == (batch_size, 1))
            assert (data[1].asnumpy().dtype == np.float32)
            break
