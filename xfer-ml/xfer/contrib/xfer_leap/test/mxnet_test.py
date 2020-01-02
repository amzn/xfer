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
from __future__ import print_function
import mxnet as mx
from mxnet import nd, autograd, gluon

from ..leap import Leap
from ..metalogger import MetaLogger

data_ctx = mx.cpu()
model_ctx = mx.cpu()

num_inputs = 2
num_outputs = 1
num_examples = 1000
num_tasks = 4
batch_size = 4


def synthetic_task_data(N_tasks, num_examples, W, apply_noise=True, real_fn=None):
    def fn(X, W):
        assert len(W) >= 3, '3 weights required. Only {} given.'.format(len(W))
        return W[1] * X[:, 0] - W[2] * X[:, 1] + W[0]

    if real_fn is None:
        real_fn = fn

    tasks = []
    for i in range(N_tasks):
        X = nd.random_normal(shape=(num_examples, num_inputs))
        noise = 0.01 * nd.random_normal(shape=(num_examples,)) if apply_noise else 0
        y = real_fn(X, W[i]) + noise
        tasks.append((X, y))
    return tasks


W = [
     [0, 10, -2],
     [0, 12, -3.8],
     [0, 7, -7],
     [0, 3.9, -11.5]
    ]

tasks = synthetic_task_data(N_tasks=num_tasks, num_examples=num_examples, W=W)

train_data_all = [gluon.data.DataLoader(gluon.data.ArrayDataset(X, y),
                                        batch_size=batch_size,
                                        shuffle=True) for X, y in tasks]

net = gluon.nn.Dense(1, in_units=2, use_bias=True)

print(net.weight)
print(net.bias)

net.collect_params()
type(net.collect_params())

net.collect_params().initialize(mx.init.Normal(sigma=1.), ctx=model_ctx)

square_loss = gluon.loss.L2Loss()

epochs = 3
loss_sequence = []
num_batches = num_examples / batch_size

verbose = 1

##########
meta_steps = 10
leap = Leap(net)
meta_trainer = gluon.Trainer(list(leap.parameters()), 'sgd', {'learning_rate': 0.0001})
meta_logger = MetaLogger(num_tasks)
log_params = True
##########

for ms in range(meta_steps):
    for task in range(num_tasks):
        train_data = train_data_all[task]

        leap.to(net)
        leap.init_task()

        trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.0001})
        if log_params:
            meta_logger.log_initial_params(ms, net)

        for e in range(epochs):
            cumulative_loss = 0
            # inner loop
            for i, (data, label) in enumerate(train_data):
                data = data.as_in_context(model_ctx)
                label = label.as_in_context(model_ctx)
                with autograd.record():
                    output = net(data)
                    loss = square_loss(output, label)
                loss.backward()
                leap.update(nd.mean(loss), net)

                trainer.step(batch_size)
                cumulative_loss += nd.mean(loss).asscalar()
            loss_sequence.append(cumulative_loss)
            meta_logger.log_loss(ms, task, e, cumulative_loss/num_examples)
            if log_params:
                meta_logger.log_params(ms, task, e, net)
            if verbose > 2:
                meta_logger.report(end=meta_logger.EPOCH, hook=print)
        leap.accumulate()
        if verbose > 1:
            meta_logger.report(end=meta_logger.TASK, hook=print)

    if verbose > 0:
        meta_logger.report(end=meta_logger.METASTEP, hook=print)
    leap.load()
    meta_trainer.step(1)  # 1 because we already normalised
    leap.zero()
