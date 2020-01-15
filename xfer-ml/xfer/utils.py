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
import os
import mxnet as mx
import json

from .constants import serialization_constants as consts
from .constants import repurposer_keys as keys


def sklearn_model_to_dict(target_model):
    output_dict = {}
    import copy
    # model_dict contains all attributes of model
    model_dict = copy.deepcopy(target_model.__dict__)
    for k in model_dict:
        # Replace any numpy array with [data_type_as_str, array_as_list]
        # e.g np.array([1,2]) -> ['int', [1,2]]
        if isinstance(model_dict[k], np.ndarray):
            type_data = str(model_dict[k].dtype)
            model_dict[k] = [type_data, model_dict[k].tolist()]
        # Replace any tuple with ['tuple', tuple_as_list]
        # e.g (1,2) -> ['tuple', [1,2]]
        if isinstance(model_dict[k], tuple):
            model_dict[k] = [keys.TUPLE, list(model_dict[k])]
    output_dict[keys.MODEL] = {}
    # Model params are public attributes
    output_dict[keys.MODEL][keys.PARAMS] = target_model.get_params()
    # Serialise all private attributes
    output_dict[keys.MODEL][keys.ATTRS] = {}
    for k in model_dict:
        # Serialize private parameters as attributes
        if k[-1] == '_' or k[0] == '_':
            output_dict[keys.MODEL][keys.ATTRS][k] = model_dict[k]
    return output_dict


def sklearn_model_from_dict(model_class, input_dict):
    # Initialize model with serialized model parameters
    model = model_class(**input_dict[keys.MODEL][keys.PARAMS])
    # Set model attributes
    for k in input_dict[keys.MODEL][keys.ATTRS]:
        # Unpack tuples and np.arrays that were serialised as lists
        if isinstance(input_dict[keys.MODEL][keys.ATTRS][k], list) \
           and isinstance(input_dict[keys.MODEL][keys.ATTRS][k][0], str) \
           and type(input_dict[keys.MODEL][keys.ATTRS][k][1]) == list:
            if input_dict[keys.MODEL][keys.ATTRS][k][0] == keys.TUPLE:
                setattr(model, k, tuple(input_dict[keys.MODEL][keys.ATTRS][k][1]))
            else:
                type_data = 'np.' + input_dict[keys.MODEL][keys.ATTRS][k][0]
                type_data = eval(type_data)
                setattr(model, k, np.array(input_dict[keys.MODEL][keys.ATTRS][k][1], dtype=type_data))
        else:
            setattr(model, k, input_dict[keys.MODEL][keys.ATTRS][k])
    return model


def _assert_repurposer_file_exists(repurposer_file_list):
    for file_name in repurposer_file_list:
        if not os.path.isfile(file_name):
            raise NameError('Cannot find repurposer file ({})'.format(file_name))


def save_mxnet_model(model, file_path_prefix, epoch, provide_data=None, provide_label=None):
    if not model.binded:
        if provide_data is None or provide_label is None:
            raise ValueError("provide_data and provide_label are required because mxnet module is not binded")
        model.bind(data_shapes=provide_data, label_shapes=provide_label)
    model.save_checkpoint(file_path_prefix, epoch)


def save_json(file_prefix, output_dict):
    with open(file_prefix + consts.JSON_SUFFIX, mode='w') as fp:
        json.dump(obj=output_dict, fp=fp)


def serialize_ctx_fn(context_function):
    if context_function == mx.cpu:
        return keys.CPU
    elif context_function == mx.gpu:
        return keys.GPU
    else:
        raise ValueError('Unexpected context function {}'.format(context_function))


def deserialize_ctx_fn(context_function):
    if context_function == keys.CPU:
        return mx.cpu
    elif context_function == keys.GPU:
        return mx.gpu
    else:
        raise ValueError('Unexpected context function {}'.format(context_function))
