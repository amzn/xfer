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
import numpy as np
import mxnet as mx

WRITE = "write"
NULL = "null"
MEAN = "mean"
RHO = "rho"
PARAMS = '_params'
JSON = '.json'
CONTEXT = 'ctx'
SHAPE = 'shape'
NPZ = '.npz'


def log_gaussian(x, mean, sigma):
    return -0.5 * np.log(2.0 * np.pi) - nd.log(sigma) - (x - mean) ** 2 / (2 * sigma ** 2)


def sample_epsilons(param_shapes, ctx):
    return [nd.random_normal(shape=shape, loc=0., scale=1.0, ctx=ctx) for shape in param_shapes]


def transform_gaussian_samples(means, sigmas, epsilons):
    return [means[j] + sigmas[j] * epsilons[j] for j in range(len(epsilons))]


def softplus(x):
    return nd.log(1. + nd.exp(x))


def softplus_inv(x):
    return nd.log(nd.exp(x) - 1.)


def softplus_inv_numpy(x):
    return np.log(np.exp(x) - 1.)


def transform_rhos(rhos):
    return [softplus(rho) for rho in rhos]


def transform_rhos_inverse(sigmas):
    return [softplus_inv(sigma) for sigma in sigmas]


def replace_params_net(layer_params, net, ctx):
    for l_param, param in zip(layer_params, net.collect_params().values()):
        ctx_list = param._ctx_map[ctx.device_typeid & 1]
        if ctx.device_id >= len(ctx_list) or ctx_list[ctx.device_id] is None:
            raise ValueError("Invalid context.")
        dev_id = ctx_list[ctx.device_id]
        param._data[dev_id] = l_param


def serialise_ctx(ctx):
    """
    Convert context to serialisable list
    """
    ctx_list = []
    if type(ctx) is not list:
        ctx = [ctx]
    for c in ctx:
        ctx_list.append((c.device_id, c.device_typeid))
    return ctx_list


def deserialise_ctx(ctx_list):
    """
    Get context from serialised context
    """
    ctx = []
    for ctx_tuple in ctx_list:
        temp_ctx = mx.cpu()
        temp_ctx.device_id = ctx_tuple[0]
        temp_ctx.device_typeid = ctx_tuple[1]
        ctx.append(temp_ctx)
    if len(ctx) == 1:
        ctx = ctx[0]
    return ctx


def deserialise_shape(shape):
    """
    Get shape from serialised shape
    """
    deserialised_shape = []
    for sh in shape:
        deserialised_shape.append(tuple(sh))
    return deserialised_shape
