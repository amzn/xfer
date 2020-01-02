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
import os
import json
import mxnet as mx

from . import utils
from .constants import repurposer_keys, bnn_repurposer_keys
from .constants import serialization_constants as consts
from .constants import serialization_keys as keys

# Import every repurposer
from .lr_repurposer import LrRepurposer  # noqa
from .svm_repurposer import SvmRepurposer  # noqa
from .bnn_repurposer import BnnRepurposer
from .gp_repurposer import GpRepurposer  # noqa
from .neural_network_fine_tune_repurposer import NeuralNetworkFineTuneRepurposer  # noqa
from .neural_network_random_freeze_repurposer import NeuralNetworkRandomFreezeRepurposer  # noqa


def load(model_name, model_directory='', source_model=None):
    """
    Load the repurposed model (target_model and supporting info) from given file_path and deserialize

    :param str model_name: Name of saved repurposer
    :param str model_directory: Directory of saved repurposer
    :param source_model: Source neural network to do transfer learning from. If this is None, then source model
                         (model_name_source-symbol.json & model_name_source-0000.params) will be loaded from file
    :type source_model: :class:`mxnet.mod.Module`
    :return: Repurposer object loaded from file
    :rtype: :class:`Repurposer` (Object returned will be resolved to originally saved class)
    """
    file_prefix = os.path.join(model_directory, model_name)
    utils._assert_repurposer_file_exists([file_prefix + consts.JSON_SUFFIX])

    with open(file_prefix + consts.JSON_SUFFIX, mode='r') as json_data:
        input_dict = json.load(json_data)

    # load source model from file if required
    if source_model is None:
        utils._assert_repurposer_file_exists([file_prefix + '_source-symbol.json', file_prefix + '_source-0000.params'])
        source_model = mx.module.Module.load(file_prefix + consts.SOURCE_SUFFIX, 0,
                                             label_names=[input_dict[keys.LAST_LAYER_NAME_SOURCE] +
                                                          consts.LABEL_SUFFIX])

    # Add file_prefix to input dict
    input_dict[keys.FILE_PATH] = file_prefix

    repurposer_class = eval(input_dict[repurposer_keys.REPURPOSER_CLASS])

    # Deserialize input dict
    context_function = utils.deserialize_ctx_fn(input_dict[repurposer_keys.PARAMS][repurposer_keys.CONTEXT_FN])
    input_dict[repurposer_keys.PARAMS][repurposer_keys.CONTEXT_FN] = context_function

    if repurposer_class == BnnRepurposer:
        bnn_context_function = utils.deserialize_ctx_fn(input_dict[repurposer_keys.PARAMS]
                                                                  [bnn_repurposer_keys.BNN_CONTEXT_FUNCTION])
        input_dict[repurposer_keys.PARAMS][bnn_repurposer_keys.BNN_CONTEXT_FUNCTION] = bnn_context_function

    # Instantiate repurposer and deserialize input dictionary
    repurposer = repurposer_class(source_model, **input_dict[repurposer_keys.PARAMS])
    repurposer.deserialize(input_dict)
    return repurposer
