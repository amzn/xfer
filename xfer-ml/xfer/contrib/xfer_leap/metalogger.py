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
import matplotlib.pyplot as plt
import numpy as np
import logging
import mxnet as mx


ALL = 'all'


class MetaLogger():
    """
    Class for holding the parameters and losses for a MetaRepurposer and plotting those values.
    """
    # TODO: Add support for logging loss/parameters after each batch rather than after every epoch
    def __init__(self, alpha_plot=0.1):
        self._losses = {}
        self._parameters = {}

        self.alpha_plot = alpha_plot

        self.EPOCH = 'epoch'
        self.TASK = 'task'
        self.METASTEP = 'metastep'

    def reset(self):
        self._losses = {}
        self._parameters = {}

    @property
    def num_tasks(self):
        all_tasks = []
        for ms in self._parameters.keys():
            all_tasks.append([k for k in self._parameters[ms].keys() if isinstance(k, int)])
        return len(np.unique(all_tasks))

    def report(self, end, hook=None):
        """
        Report results at end of epoch/task/metastep using hook function.
        """
        if hook is None:
            hook = logging.info

        reporter = {self.EPOCH: self._report_epoch,
                    self.TASK: self._report_task,
                    self.METASTEP: self._report_metastep}

        reporter[end](hook)

    def _report_epoch(self, hook):
        hook('\t\tMetastep: {}, Task: {}, Epoch: {}, Loss: {:.3f}'.format(
            self.latest_metastep, self.latest_task,
            len(self._losses[self.latest_metastep][self.latest_task]),
            self._losses[self.latest_metastep][self.latest_task][-1]))

    def _report_task(self, hook):
        initial_loss = self._losses[self.latest_metastep][self.latest_task][0]
        final_loss = self._losses[self.latest_metastep][self.latest_task][-1]
        hook('\tMetastep: {}, Task: {}, Initial Loss: {:.3f}, Final Loss: {:.3f}, Loss delta: {:.3f}'.format(
            self.latest_metastep, self.latest_task,
            initial_loss, final_loss, final_loss - initial_loss))

    def _report_metastep(self, hook):
        loss_total = 0
        for task_loss in self._losses[self.latest_metastep].values():
            loss_total += task_loss[-1]
        num_tasks = len(self._losses[self.latest_metastep].keys())
        mean_loss = loss_total / num_tasks
        hook('Metastep: {}, Num tasks: {}, Mean Loss: {:.3f}'.format(self.latest_metastep, num_tasks, mean_loss))

    @property
    def latest_metastep(self):
        return max(self._losses.keys())

    @property
    def latest_task(self):
        return max(self._losses[self.latest_metastep].keys())

    def log_loss(self, metastep, task, epoch, loss):
        """
        Append loss to dictionary.
        """
        if metastep not in self._losses.keys():
            self._losses[metastep] = {}
        if task not in self._losses[metastep].keys():
            self._losses[metastep][task] = []
        self._losses[metastep][task].append(loss)

    def log_params(self, metastep, task, epoch, net):
        """
        Append parameters to dictionary.
        """
        parameters = {}
        for k, v in net.params.items():
            parameters[k] = v.data().copy().asnumpy()

        if metastep not in self._parameters.keys():
            self._parameters[metastep] = {}
        if task not in self._parameters[metastep].keys():
            self._parameters[metastep][task] = []
        self._parameters[metastep][task].append(parameters)

    def log_initial_params(self, ms, net):
        """
        Log parameters before any updates made.
        """
        if ms in self._parameters.keys():
            return
        self.log_params(ms, ALL, -1, net)

    def plot_losses(self, add_label=True, figsize=(20, 4)):
        """
        Plot the logged losses.
        """
        if self._losses == {}:
            raise ValueError('No losses logged.')
        fig, axes = plt.subplots(ncols=self.num_tasks, figsize=figsize)
        fig.suptitle('Losses', fontsize=30, y=1.08)
        for task in range(self.num_tasks):
            axes[task].set_title('Task {}'.format(task))
            axes[task].set_xlabel('epoch')
            axes[task].set_ylabel('loss')
        for ms in self._losses.keys():
            for task in range(self.num_tasks):
                if task in self._losses[ms].keys():
                    alpha = 1 if ms == max(self._losses.keys()) else self.alpha_plot
                    axes[task].plot(self._losses[ms][task], 'o-', alpha=alpha)
                    if add_label:
                        axes[task].text(x=0.05, y=self._losses[ms][task][0], s=ms)

    def plot_params(self, param, W, loss_fn, figsize=(20, 6), gridsize=(100, 100), a=0.2, loss_samples=100):
        """
        Plot the logged parameters.
        """
        if self._parameters == {}:
            raise ValueError('No parameters logged.')
        fig, axes = plt.subplots(ncols=self.num_tasks, figsize=figsize)
        for surface in range(self.num_tasks):
            for ms in sorted(self._parameters.keys()):
                for task in range(self.num_tasks):
                    if task in self._parameters[ms].keys() or ms == max(self._parameters.keys()):
                        temp_ms = ms
                        while task not in self._parameters[temp_ms].keys():
                            temp_ms -= 1
                        x = np.concatenate([p[param] for p in self._parameters[temp_ms][task]])
                        x = np.concatenate([self._parameters[temp_ms]['all'][0][param], x]).T
                        initial_point = self._parameters[temp_ms]['all'][0][param].T

                        assert x.shape[0] == 2, 'Dimension of parameter must be 2.'

                        label = task if ms == max(self._parameters.keys()) else None
                        alpha = 1 if ms == max(self._parameters.keys()) else self.alpha_plot
                        color = 'r' if surface == task else 'k'
                        axes[surface].plot(x[0], x[1], 'o-', color=color, label=label, alpha=alpha)
                        axes[surface].plot(initial_point[0], initial_point[1], 'o-', color='tab:pink', alpha=alpha)
            axes[surface].legend()
            axes[surface].set_title('Loss surface for Task {}'.format(surface))
            # Plot loss surface
            extent = axes[surface].get_xlim() + axes[surface].get_ylim()
            grid = np.zeros(gridsize)
            for i, w1 in enumerate(np.linspace(extent[0], extent[1], gridsize[0])):
                for j, w2 in enumerate(np.linspace(extent[2], extent[3], gridsize[1])):
                    grid[j][i] = loss_fn(mx.nd.array([w1, w2]), W[surface], loss_samples)
            axes[surface].imshow(grid, extent=extent, origin='lower')
            # Set labels
            axes[surface].set_xlabel(param + ' 1')
            axes[surface].set_ylabel(param + ' 2')
        fig.suptitle('Parameters', fontsize=30, y=0.9)
