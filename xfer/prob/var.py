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
import numpy as np
import mxnet as mx
import json
import mxnet.ndarray as nd

from .prior import Probability, GaussianPrior
from .utils import sample_epsilons, transform_rhos, log_gaussian, transform_gaussian_samples, softplus_inv_numpy,\
    deserialise_ctx, deserialise_shape, MEAN, RHO, PARAMS, JSON, CONTEXT, SHAPE, NPZ


class VariationalPosterior(Probability):
    def __init__(self, shapes, ctx):
        super(VariationalPosterior, self).__init__(shapes, ctx)

    def generate_sample(self):
        pass

    def get_mean(self):
        pass


class GaussianVariationalPosterior(VariationalPosterior):
    """
    Gaussian variational distribution (iid) over a set of variables [W_1, ..., W_n] where W_i is an mxnet.array

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

    def __init__(self, mean_init, sigma_init, shapes, ctx, fix_means=False, fix_sigmas=False):
        """
        Default constructor

        :param mean_init: The initial variational mean.
            float: Different mean for each value of each W_i (sampled from N(mean_init, 1))
            [np.array]: Mean shared across all variables.
            [np.array_1, ..., np.array_n]: Different mean for each variable W_i
        :type mean_init: float or list[:class:`np.ndarray`]
        :param sigma_init: Initial variational variance.
            float: Different variance for each variable (initialize to float value)
            [np.array]: Variance shared across all variables.
            [np.array_1, ..., np.array_n]: Different variance for each variable W_i
        :type sigma_init: float or list[:class:`np.ndarray`]
        :param shapes: Size of the variables [W_1, ..., W_n].
        :type shapes: list(tuple(int, int))
        :param ctx: MXNet context
        :type ctx: :class:`mxnet.Context`
        :param fix_means: Fix the mean of the Variational Posterior.
        :type fix_means: boolean or list(boolean)
        :param fix_sigmas: Fix the variance of the Variational Posterior.
        :type fix_sigmas: boolean or list(boolean)
        """

        super(GaussianVariationalPosterior, self).__init__(shapes, ctx)

        self._register_params(mean_init, sigma_init, fix_means, fix_sigmas)

    def _register_params(self, mean_init, sigma_init, fix_means, fix_sigmas):
            if not isinstance(mean_init, list):
                self._register_param(MEAN, mx.init.Normal(mean_init), fix_means)
            else:
                self._register_param_value(MEAN, mean_init, fix_means)
            if not isinstance(sigma_init, list):
                rho_init = softplus_inv_numpy(sigma_init)
                self._register_param(RHO, mx.init.Constant(rho_init), fix_sigmas)
            else:
                rho_init = [softplus_inv_numpy(ss) for ss in sigma_init]
                self._register_param_value(RHO, rho_init, fix_sigmas)

    def get_mean(self):
        return self.raw_params[MEAN]

    def generate_sample(self):
        epsilons = sample_epsilons(self.shapes, self.ctx)
        raw_params_ext_var_posterior = self._replicate_shared_parameters()
        sigmas = transform_rhos(raw_params_ext_var_posterior[RHO])
        return transform_gaussian_samples(raw_params_ext_var_posterior[MEAN], sigmas, epsilons)

    def log_pdf(self, obs):
        self.check_observation_shapes(obs)
        raw_params_ext = self._replicate_shared_parameters()
        sigmas = transform_rhos(raw_params_ext[RHO])
        return sum([nd.sum(log_gaussian(obs[ii], raw_params_ext[MEAN][ii], sigmas[ii]))
                    for ii in range(len(self.shapes))])

    def is_conjugate(self, other_prob):
        if type(other_prob) == GaussianPrior:
            return True
        else:
            return False

    def KL(self, other_prob):
        if not self.is_conjugate(other_prob):
            raise ValueError("KL cannot be computed in closed form.")

        if (not len(self.shapes) == len(other_prob.shapes)) or \
                (not np.all(np.array([s == o for s, o in zip(self.shapes, other_prob.shapes)]))):
            raise ValueError("KL cannot be computed: The 2 distributions have different support")

        raw_params_ext_var_posterior = self._replicate_shared_parameters()
        sigmas_var_posterior = transform_rhos(raw_params_ext_var_posterior[RHO])
        raw_params_ext_prior = other_prob._replicate_shared_parameters()

        out = 0.0
        for ii in range(len(self.shapes)):
            means_p = raw_params_ext_prior[MEAN][ii]
            var_p = raw_params_ext_prior["sigma"][ii] ** 2
            means_q = raw_params_ext_var_posterior[MEAN][ii]
            var_q = sigmas_var_posterior[ii] ** 2
            inc_means = (means_q - means_p)
            prec_p = 1.0 / var_p
            temp = 0.5 * (var_q*prec_p + ((inc_means ** 2) * prec_p) - 1.0 + nd.log(var_p) - nd.log(var_q))
            if temp.shape == (1, 1):
                # If parameters are shared, multiply by the number of variables
                temp = temp * (self.shapes[ii][0] * self.shapes[ii][1])
            out = out + nd.sum(temp)
        return out

    def save(self, filename):
        """
        Save object to file. This will create two files: filename.json, filename_params.npz

        :param filename: Name to give saved object files
        """
        self._save_params(filename + PARAMS)
        output_dict = self._to_dict()
        with open(filename + JSON, 'w') as fp:
            json.dump(obj=output_dict, fp=fp)

    @staticmethod
    def load(filename):
        """
        Load object from file. This requires two files: filename.json, filename_params.npz

        :param filename: Names of files to be opened and loaded
        """
        with open(filename + JSON, 'r') as json_data:
            input_dict = json.load(json_data)
        ctx = deserialise_ctx(input_dict[CONTEXT])
        shape = deserialise_shape(input_dict[SHAPE])

        gvp = GaussianVariationalPosterior(1, 1, shape, ctx, False, False)

        params, raw_params = GaussianVariationalPosterior._load_params(filename + PARAMS + NPZ)
        gvp.params = params
        gvp.raw_params = raw_params

        return gvp
