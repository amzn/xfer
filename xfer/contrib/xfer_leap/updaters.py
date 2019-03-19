"""Meta learning by gradient path minimization

MXNet implementation of the Leap algorithm: https://arxiv.org/abs/1812.01054.

:author: Sebastian Flennerhag
"""
# pylint: disable=too-many-arguments, invalid-name, redefined-builtin, protected-access, too-many-locals, arguments-differ

import mxnet as mx

from collections import OrderedDict


class BaseUpdater(object):

    """Base class for update schemes"""

    # pylint: disable=too-many-instance-attributes

    def __init__(self, state):
        self.state = state
        self.prev_loss = None
        self.prev_state = None
        self.train_counter = 0
        self.initialize()

    def __call__(self, curr_loss, curr_state, *args, **kwargs):
        with mx.autograd.pause():
            self.update(curr_loss, curr_state, *args, **kwargs)
            self.memorize(curr_loss, curr_state)
            self.train_counter += 1

    def memorize(self, curr_loss, curr_state):
        """Update memory buffer"""
        self.prev_loss = curr_loss

        for n, p in curr_state.items():
            if hasattr(p, 'copy'):
                self.prev_state[n] = p.copy()
                if getattr(p, 'grad', None) is not None:
                    if self.prev_state[n].grad is None:
                        self.prev_state[n].attach_grad()
                    p.grad.copyto(self.prev_state[n].grad)
            else:
                self.prev_state[n] = p

    def initialize(self):
        """Initialize counters for new task"""
        self.train_counter = 0
        self.prev_loss = None
        self.prev_state = OrderedDict([(n, None) for n in self.state.keys()])

    def update(self, curr_loss, curr_state):
        """Add unconstrained gradient term"""
        raise NotImplementedError


class DefaultUpdater(BaseUpdater):

    """Default Leap updater scheme"""

    def __init__(self, state, norm=True, regularizer=None, loss=True):
        super(DefaultUpdater, self).__init__(state)
        self.norm = norm
        self.loss = loss
        self.regularizer = regularizer

    def update(self, curr_loss, curr_state, hook=None):
        """Accumulate gradients in a state dictionary"""
        prev_loss = self.prev_loss
        prev_state = self.prev_state

        if prev_loss is None:
            return

        d_loss = curr_loss - prev_loss

        if self.regularizer and d_loss > 0:
            d_loss = -d_loss

        if self.norm:
            norm = d_loss * d_loss if self.loss else 0
            for n, p in self.state.items():
                if getattr(p, 'grad', None) is None:
                    continue

                cp = curr_state[n].detach()
                pp = prev_state[n].detach()
                d = cp.reshape((-1,)) - pp.reshape((-1,))
                norm += mx.nd.dot(d, d)
            norm.sqrt(out=norm)

        for n, p in self.state.items():
            if getattr(p, 'grad', None) is None:
                continue

            cp = curr_state[n].detach()
            pp = prev_state[n].detach()
            pg = prev_state[n].grad.detach()

            if self.loss:
                add = -d_loss * pg + pp - cp
            elif self.regularizer and curr_loss > prev_loss:
                add = cp - pp
            else:
                add = pp - cp

            if self.norm:
                add /= norm

            if hook is not None:
                hook(add)

            mx.nd.elemwise_add(p.data().grad, add, out=p.data().grad)
