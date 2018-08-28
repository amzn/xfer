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
import mxnet.ndarray as nd

from .prob_base import Probability
from .utils import log_gaussian


class Prior(Probability):
    def __init__(self, shapes, ctx):
        super(Prior, self).__init__(shapes, ctx)


class GaussianPrior(Prior):

    """
    Gaussian Prior (iid) over a set of variables [W_1, ..., W_n] where W_i is an mxnet.ndarray

    :param params: Dictionary containing the parameters that characterize the probability distribution
      , i.e. mean and variance of each W_i.
    :type params: dict(str, :class:`mxnet.gluon.Parameter`)
    :param raw_params: Dictionary containing the values of the parameters stored in params
    :type raw_params: dict(str, list[:class:`mxnet.ndarray`]}
    :param shapes: Shape of the variables over which the probability is defined.
    :type shapes: list(tuple(int, int))
    :param ctx: MXNet context
    :type ctx: :class:`mxnet.Context`
    """

    def __init__(self, means, sigmas, shapes, ctx=None, fix_means=True, fix_sigmas=True):
        """
        Default constructor

        :param means: Mean of the prior. Can be:
                [np.array]: Mean shared across all variables.
                [np.array_1, ..., np.array_n]: Different mean for each variable W_i
        :type means: list[:class:`np.ndarray`]
        :param sigmas: Variance of the prior. Can be:
                [np.array]: Variance shared across all variables.
                [np.array_1, ..., np.array_n]: Different variance for each variable W_i
        :type sigmas: list[:class:`np.ndarray`]
        :param shapes: Define the shape of each variable W_i.
        :type shapes: list(tuple(int, int))
        :param fix_means: Fix the mean of the Prior.
        :type fix_means: boolean or list(boolean)
        :param fix_sigmas: Fix the variance of the Prior
        :type fix_sigmas: boolean or list(boolean)
        """

        super(GaussianPrior, self).__init__(shapes, ctx)
        self._register_params(means, sigmas, fix_means, fix_sigmas)

    def _register_params(self, means, sigmas, fix_means, fix_sigmas):
        self._register_param_value("mean", means, fix=fix_means)
        self._register_param_value("sigma", sigmas, fix=fix_sigmas)

    def log_pdf(self, obs):
        self.check_observation_shapes(obs)
        raw_params_ext = self._replicate_shared_parameters()
        return sum([nd.sum(log_gaussian(obs[ii], raw_params_ext["mean"][ii], raw_params_ext["sigma"][ii]))
                    for ii in range(len(self.shapes))])
