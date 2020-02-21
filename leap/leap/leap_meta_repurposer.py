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
import random

from mxnet import nd, autograd, gluon
import mxnet as mx

from .meta_repurposer import MetaRepurposer
from .leap import Leap
from .metalogger import MetaLogger

DEFAULT_META_OPTIMIZER_PARAMS = {'learning_rate': 0.0001}
DEFAULT_TASK_OPTIMIZER_PARAMS = {'learning_rate': 0.005}


class LeapMetaRepurposer(MetaRepurposer):
    """
    Repurposer to do meta-learning using Leap algorithm.

    Starting from the parameters of input_model as the initial point, the Leap algorithm is applied. It learns a final
    parametrization of input_model that can be quickly adapted to solve a test task sampled from the same underlying
    distribution that generated train_tasks.

    :param input_model: Source neural network to use for meta-learning.
    :type input_model: :class:`mxnet.gluon.Block`
    """

    def __init__(self, model, num_meta_steps, num_epochs, loss_class=gluon.loss.L2Loss,
                 meta_optimizer='sgd', meta_optimizer_params=None, task_optimizer='sgd', task_optimizer_params=None,
                 log_params=False, verbosity=1, context=None):
        self.model = model
        self.meta_logger = MetaLogger()

        self.num_meta_steps = num_meta_steps
        self.num_epochs = num_epochs

        self.context = context
        if self.context is None:
            self.context = mx.cpu()

        self.loss_class = loss_class
        self.meta_optimizer = meta_optimizer
        self.meta_optimizer_params = meta_optimizer_params
        if self.meta_optimizer_params is None:
            self.meta_optimizer_params = DEFAULT_META_OPTIMIZER_PARAMS
        self.task_optimizer = task_optimizer
        self.task_optimizer_params = task_optimizer_params
        if self.task_optimizer_params is None:
            self.task_optimizer_params = DEFAULT_TASK_OPTIMIZER_PARAMS

        self.log_params = log_params
        self.verbosity = verbosity

    @staticmethod
    def _task_sampler(train_tasks, meta_batch_size, with_replacement=False):
        """
        Tasks' meta-batch sampler

        :param train_tasks: List of enumerated training tasks
        :param meta_batch_size: Number of tasks per meta-batch
        :param with_replacement: If True, sample tasks with replacement.
        :return: List of sampled enumerated tasks
        """

        n_tasks = len(train_tasks)

        if n_tasks == 1:
            tasks = train_tasks * meta_batch_size
        elif with_replacement:
            tasks = [train_tasks[random.randint(0, n_tasks - 1)] for _ in range(meta_batch_size)]
        else:
            tasks = []
            ix_tasks = list(range(n_tasks))
            while True:
                random.shuffle(ix_tasks)
                tasks.extend([train_tasks[i] for i in ix_tasks])
                if len(tasks) >= meta_batch_size:
                    break
            tasks = tasks[:meta_batch_size]

        return tasks

    def repurpose(self, train_tasks, meta_batch_size=None, with_replacement=False):
        """
        Meta-learn parameters of the input model such that they are optimized for the provided train_tasks.

        :param train_tasks: List of all training tasks.
        :param meta_batch_size: Number of tasks per meta-batch (size of train_tasks as default value).
        :param with_replacement: If True, sample tasks with replacement.

        """

        loss_function = self.loss_class()
        loss_sequence = []

        net = self.model
        leap = Leap(net)
        meta_trainer = gluon.Trainer(list(leap.parameters()), self.meta_optimizer, self.meta_optimizer_params)

        self.meta_logger.reset()

        if meta_batch_size is None:
            meta_batch_size = len(train_tasks)

        train_tasks = list(enumerate(train_tasks))

        for ms in range(self.num_meta_steps):

            batch_tasks = LeapMetaRepurposer._task_sampler(train_tasks, meta_batch_size, with_replacement)
            for task, train_data in batch_tasks:

                leap.to(net)
                leap.init_task()

                trainer = gluon.Trainer(net.collect_params(), self.task_optimizer, self.task_optimizer_params)
                if self.log_params:
                    self.meta_logger.log_initial_params(ms, net)

                for e in range(self.num_epochs):
                    num_examples = 0
                    cumulative_loss = 0
                    # Inner loop for training model for task
                    for i, (data, label) in enumerate(train_data):
                        if i == 0:
                            batch_size = data.shape[0]
                        data = data.as_in_context(self.context)
                        label = label.as_in_context(self.context)
                        with autograd.record():
                            output = net(data)
                            loss = loss_function(output, label)
                        loss.backward()
                        leap.update(nd.mean(loss), net)

                        trainer.step(batch_size)
                        cumulative_loss += nd.mean(loss).asscalar()
                        num_examples += len(label)
                    loss_sequence.append(cumulative_loss)
                    self.meta_logger.log_loss(ms, task, e, cumulative_loss/num_examples)
                    if self.log_params:
                        self.meta_logger.log_params(ms, task, e, net)
                    if self.verbosity > 2:
                        self.meta_logger.report(end=self.meta_logger.EPOCH, hook=print)
                leap.accumulate()
                if self.verbosity > 1:
                    self.meta_logger.report(end=self.meta_logger.TASK, hook=print)

            if self.verbosity > 0:
                self.meta_logger.report(end=self.meta_logger.METASTEP, hook=print)
            leap.load()
            meta_trainer.step(1)  # 1 because we already normalised
            leap.zero()

        leap.to(net)
