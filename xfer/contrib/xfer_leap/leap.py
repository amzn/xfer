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
import mxnet as mx

from collections import OrderedDict
from .updaters import DefaultUpdater


class Leap(object):

    """
    Meta gradient path learner.

    MXNet implementation of the Leap algorithm: https://arxiv.org/abs/1812.01054.
    """

    def __init__(self, model, norm=True, regularizer=True, loss=True):
        self.norm = norm
        self.loss = loss
        self.regularizer = regularizer

        self.state = mx.gluon.ParameterDict()
        for n, p in model.collect_params().items():
            q = mx.gluon.Parameter(
                p.name,
                p.grad_req,
                p.shape,
                p.dtype,
                p.lr_mult,
                p.wd_mult,
                p._allow_deferred_init,
                p._differentiable,
                p._stype
            )
            q.initialize(mx.init.Normal(sigma=1.), ctx=p.data().context)
            q.set_data(p.data())
            self.state._params[n] = q

        self._acc_counter = None
        self._acc_state = None

        self.updater = DefaultUpdater(
            self.state,
            self.norm,
            self.regularizer,
            self.loss
        )

        self.zero()
        self.clear_acc()
        self.init_task()

    def zero(self):
        """Zero out gradients tensors in state"""
        for p in self.parameters():
            if getattr(p, 'grad', None) is not None:
                q = p.data()
                mx.nd.zeros(q.shape, q.context, q.dtype, q.stype).copyto(p.grad())

    def clear_acc(self):
        """Clear gradient accumulation state"""
        self._acc_counter = 0
        self._acc_state = OrderedDict([(n, None) for n in self.state.keys()])

    def to(self, model):
        """Transfer cloned state to model"""
        new_state = OrderedDict()
        for n, p in self.state.items():
            d = p.data().detach().copy()
            if getattr(p, 'grad', None) is not None:
                d.attach_grad()
            new_state[n] = d

        mstate = model.collect_params()
        for (n, p), (m, q) in zip(mstate.items(), new_state.items()):
            assert n == m, 'incorrect parameter order: {} != {}'.format(n, m)
            p.set_data(q.detach())

    def init_task(self):
        """Initialize running variables"""
        self.updater.initialize()

    def reset(self):
        """Reset all counters"""
        self.updater.reset()
        self.clear_acc()
        self.init_task()
        self.zero()

    def accumulate(self):
        """Accumulate meta gradients across tasks"""
        self._acc_counter += 1

        def _acc(n, p):
            g = self._acc_state[n]

            if g is None:
                q = p.data()
                g = self._acc_state[n] = mx.nd.zeros(
                    q.shape, q.context, q.dtype, q.stype)

            if getattr(p, 'grad', None) is not None:
                g += p.data().grad.detach()
            else:
                g += p.data().detach()

        with mx.autograd.pause():
            for n, p in self.state.items():
                _acc(n, p)

    def load(self):
        """Load accumulated meta gradients and buffers into aggregate meta gradient"""

        def _load(n, p):
            g = self._acc_state[n]
            g /= self._acc_counter
            if getattr(p, 'grad', None) is not None:
                g.copyto(p.data().grad)
            else:
                g.copyto(p.data())
            p.data()._fresh_grad = True

        with mx.autograd.pause():
            for n, p in self.state.items():
                _load(n, p)

    def update(self, loss, model, hook=None):
        """Increment meta gradient given current task"""
        curr_state = {n: p.data() for n, p in model.collect_params().items()}
        curr_loss = loss.detach().copy()
        self.updater(curr_loss, curr_state, hook=hook)

    def named_parameters(self):
        """Iterator named parameters"""
        for n, p in self.state.items():
            if getattr(p, 'grad', None) is not None:
                yield n, p

    def parameters(self):
        """Iterator over parameters"""
        for _, p in self.named_parameters():
            yield p
