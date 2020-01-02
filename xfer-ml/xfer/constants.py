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
from .__version__ import __version__


class constants:
    VERSION = __version__


class repurposer_keys:
    PARAMS = 'parameters'
    TARGET_MODEL = 'target_model'
    ATTRS = 'attributes'
    MODEL = 'model'
    GPU = 'gpu'
    CPU = 'cpu'
    TUPLE = 'tuple'
    CONTEXT_FN = 'context_function'
    NUM_DEVICES = 'num_devices'
    PROVIDE_DATA = 'provide_data'
    PROVIDE_LABEL = 'provide_label'
    REPURPOSER_CLASS = 'repurposer_class'


class meta_model_repurposer_keys:
    C = 'c'
    N_JOBS = 'n_jobs'
    TOL = 'tol'
    FEATURE_LAYERS = 'feature_layer_names'
    KERNEL = 'kernel'
    GAMMA = 'gamma'
    PROB_ESTIMATES = 'enable_probability_estimates'


class neural_network_repurposer_keys:
    TRANSFER_LAYER_NAME = 'transfer_layer_name'
    TARGET_CLASS_COUNT = 'target_class_count'
    LEARNING_RATE = 'learning_rate'
    OPTIMIZER = 'optimizer'
    OPTIMIZER_PARAMS = 'optimizer_params'
    BATCH_SIZE = 'batch_size'
    NUM_EPOCHS = 'num_epochs'
    FIXED_LAYERS = 'fixed_layers'
    RANDOM_LAYERS = 'random_layers'
    NUM_LAYERS_TO_DROP = 'num_layers_to_drop'


class neural_network_repurposer_constants:
    DEFAULT_LEARNING_RATE = 0.001


class serialization_constants:
    JSON_SUFFIX = '.json'
    SOURCE_SUFFIX = '_source'
    LABEL_SUFFIX = '_label'
    POSTERIOR_SUFFIX = '_posterior'


class serialization_keys:
    FILE_PATH = 'file_path'
    VERSION = 'version'
    LAST_LAYER_NAME_SOURCE = 'last_layer_name'
    LAST_LAYER_NAME_TARGET = 'last_layer_name_target'


class bnn_repurposer_keys:
    SIGMA = 'sigma'
    NUM_LAYERS = 'num_layers'
    N_HIDDEN = 'n_hidden'
    NUM_SAMPLES_MC = 'num_samples_mc'
    START_ANNEALING = 'start_annealing'
    END_ANNEALING = 'end_annealing'
    ANNEALING_WEIGHT = 'annealing_weight'
    DIM_INPUT = 'dim_input'
    NUM_CLASSES = 'num_classes'
    LEARNING_RATE = 'learning_rate'
    BATCH_SIZE = 'batch_size'
    NUM_EPOCHS = 'num_epochs'
    VERBOSE = 'verbose'
    BNN_CONTEXT_FUNCTION = 'bnn_context_function'
    NUM_SAMPLES_MC_PREDICT = 'num_samples_mc_prediction'


class bnn_constants:
    MEAN_INIT_POSTERIOR = 0.1
    SIGMA_INIT_POSTERIOR = 0.05
    BNN_OPTIMIZER = 'adam'


class gp_repurposer_keys:
    MAX_FUNCTION_EVALUATIONS = 'max_function_evaluations'
    APPLY_L2_NORM = 'apply_l2_norm'
    FEATURE_MEAN = 'feature_mean'
    OPTIMIZER = 'optimizer'
    NUM_INDUCING_SPARSE_GP = 'num_inducing_sparse_gp'
    L2_NORMALIZER = 'l2_normalizer'
