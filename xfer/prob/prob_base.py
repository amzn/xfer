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
from mxnet import gluon
import numpy as np

from .utils import serialise_ctx, WRITE, NULL, SHAPE, CONTEXT


class Probability(object):

    """
    Base class for defining a probability over a set of variables [W_1, ..., W_n] where W_i is an mxnet.ndarray.

    :param params: Dictionary containing the parameters that characterize the probability distribution
      (e.g. mean and variance of each W_i when the Probability represents a gaussian distribution)
    :type params: dict(str, :class:`mxnet.gluon.Parameter`)
    :param raw_params: Dictionary containing the values of the parameters stored in params
    :type raw_params: dict(str, list[:class:`mxnet.ndarray`]}
    :param shapes: Shape of the variables over which the probability is defined.
    :type shapes: list(tuple(int, int))
    :param ctx: MXNet context
    :type ctx: :class:`mxnet.Context`
    """

    def __init__(self, shapes, ctx):

        """
        Default constructor

        :param shapes: Shape of the variables over which the probability is defined.
        :type shapes: list(tuple(int, int))
        :param ctx: MXNet context
        :type ctx: :class:`mxnet.Context`
        """

        self.ctx = ctx
        self.params = {}
        self.raw_params = {}
        self.shapes = shapes

    def get_params_list(self):
        # Each element of the dictionary contains a parameter (list(nd.array)). This method
        # iterates over all the elements in the dictionary and outputs a single list(nd.array) containing
        # all parameters. The dictionary is sorted beforehand so the output is deterministic and the
        # method can be tested.
        return [x
                for kk in sorted(self.params)
                for x in self.params[kk]
                ]

    def check_observation_shapes(self, obs):
        shape_obs = [o.shape for o in obs]
        if not self.shapes == shape_obs:
            raise ValueError("Number of observed variables {} does not match \
             the number of expected observed variables {}".format(shape_obs, self.shapes))

    def log_pdf(self, x):
        raise NotImplemented

    def KL(self, other_prob):
        """
        Function to compute the KL divergence between self and a second Probability

        :param other_prob: The probability object that we want to compute th KL distance to
        :type other_prob: :class:`xfer.prob.Probability`
        """
        raise NotImplemented

    def KL_sample(self, other_prob, sample):
        """
        Function to compute the KL divergence between a sample of self and a sample of a second Probability

        :param other_prob: The probability that we want to compute the KL distance to
        :type other_prob: :class:`xfer.prob.Probability`
        :param sample: Sample used to compute the KL
        list[:class:`np.ndarray`]
        """
        return self.log_pdf(sample) - other_prob.log_pdf(sample)

    def _parse_grad_req(self, fix, num_parameters=None):
        """
        :param fix: define for each parameter whether the gradient is computed  (True) or
            not (False). If it is a scalar, the value is replicated num_parameters times.
        :type fix: boolean or list(boolean)
        :return: define for each parameter whether the gradient is computed  ("write") or not ("null")
        :rtype: list[string]
        """
        if isinstance(fix, list):
            grad_req = [NULL if f else WRITE for f in fix]
        elif fix:
            grad_req = [NULL] * num_parameters
        else:
            grad_req = [WRITE] * num_parameters
        return grad_req

    def _register_param(self, name, init, fix=False):
        """
        Register a set of parameters using "init".

        Its shape is equal to the shape of the variables over which the probability is defined (self.shapes).
        In addition to updating the parameters' dictionary (self.params), this function updates a dictionary with the
        value of the parameters (self.raw_params).

        :param string name: Name of the set of parameters.
        :param init: Define how to initialize the set of parameters
        :type init: mxnet.Initializer
        :param fix: define for each parameter whether the gradient is computed  (True) or not (False). If it is a single
         boolean it is applied globally to the full set of variables.
        :type fix: (boolean or list(boolean))
        """

        par_list = []
        grad_req = self._parse_grad_req(fix, len(self.shapes))
        for shape, gg in zip(self.shapes, grad_req):
            par = gluon.Parameter(name, shape=shape, init=init, grad_req=gg)
            par.initialize(ctx=self.ctx)
            par_list.append(par)
        self.params[name] = par_list
        self.raw_params[name] = [x.data(self.ctx) for x in self.params[name]]

    def _register_param_value(self, name, values, fix=False):
        """
        Register a parameter and initialize it with values (list(nd.ndarray)).

        It also checks whether the shape (given by "values") is compatible with the shape of the variables
        over which the probability is defined (self.shapes).

        :param string name: Name of the set of parameters.
        :param values: Value of the set of parameters.
                [np.array]: Parameters shared across all variables.
                [np.array_1, ..., np.array_n]: Different parameters for each variable W_i
        :type values: list[:class:`np.ndarray`]
        :param fix: define for each parameter whether the gradient is computed  (True) or not (False). If it is a single
         boolean it is applied globally to the full set of variables.
        :type fix: (boolean or list(boolean))
        """
        values_shapes = [vv.shape for vv in values]

        if not len(values) == 1:
            if not (len(values_shapes) == len(self.shapes)):
                raise ValueError("Parameter {} size is not compatible".format(name))
            for value_shape, parameter_shape in zip(values_shapes, self.shapes):
                if not (value_shape == parameter_shape or value_shape == (1, 1)):
                    raise ValueError("Parameter {} size is not compatible".format(name))
        else:
            if not values[0].shape == (1, 1):
                raise ValueError("Parameter {} size is not compatible".format(name))

        par_list = []
        grad_req = self._parse_grad_req(fix, len(values))
        for vv, gg in zip(values, grad_req):
            init = mx.init.Constant(vv)
            par = gluon.Parameter(name, shape=vv.shape, init=init, grad_req=gg)
            par.initialize(ctx=self.ctx)
            par_list.append(par)
        self.params[name] = par_list
        self.raw_params[name] = [x.data(self.ctx) for x in self.params[name]]

    def _replicate_shared_parameters(self):
        """
        It returns a dictionary containing the values of each set of parameters of the distribution.

        The difference with self.raw_params is that the shared parameters are replicated.
        """
        raw_params_ext = {}
        for k in self.raw_params:
            if len(self.raw_params[k]) == 1:
                # If the parameter is shared, replicate it len(self.shapes) times
                raw_params_ext[k] = [self.raw_params[k][0] for _ in range(len(self.shapes))]
            else:
                raw_params_ext[k] = self.raw_params[k]
        return raw_params_ext

    def _to_dict(self):
        output_dict = {
            SHAPE: self.shapes,
            CONTEXT: serialise_ctx(self.ctx)
        }
        return output_dict

    def _save_params(self, filename):
        np.savez(filename, **self.params)

    @staticmethod
    def _load_params(filename):
        load_data = np.load(filename)
        params = {}
        raw_params = {}
        for key in load_data.files:
            params[key] = load_data[key]
            raw_params[key] = []
            for parameter in params[key]:
                # Parameter class in mxnet<1.3.0 does not have _stype attribute which causes an error when Parameter
                # saved in mxnet<1.3.0 is loaded with mxnet==1.3.0
                try:
                    parameter._stype
                except AttributeError:
                    parameter._stype = 'default'
                raw_params[key].append(parameter.data())

        return params, raw_params
