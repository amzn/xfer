# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
import yaml
import argparse
from easydict import EasyDict

def create_dirs(dirs):
    """
    Create directories given by a list if these directories are not found

    :param list dirs: directories
    :return exit_code: 0:success -1:failed
    """
    try:
        for dir_ in dirs:
            if not os.path.exists(dir_):
                os.makedirs(dir_)
        return 0
    except Exception as err:
        print("Creating directories error: {0}".format(err))
        exit(-1)


def get_config_from_json(json_file):
    """
    Get the config from a json file

    :param string json_file: json configuration file
    :return: EasyDict config
    """
    # parse the configurations from the config json file provided
    with open(json_file, 'r') as config_file:
        config_dict = json.load(config_file)

    # convert the dictionary to a namespace using bunch lib
    config = EasyDict(config_dict)
    return config


def get_config_from_yaml(yaml_file):
    """
    Get the config from a yaml file

    :param string yaml_file: yaml configuration file
    :return: EasyDict config
    """
    with open(yaml_file) as fp:
        config_dict = yaml.safe_load(fp)

    # convert the dictionary to a namespace using bunch lib
    config = EasyDict(config_dict)
    return config


def get_args():
    """
    Create argparser for frequent configurations.

    :return: argparser object
    """
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-c', '--config',
        metavar='C',
        default=None,
        help='The Configuration file')
    argparser.add_argument(
        '-k', '--steps',
        default=3,
        type=int,
        help='The number of SIB steps')
    argparser.add_argument(
        '-s', '--seed',
        default=100,
        type=int,
        help='The random seed')
    argparser.add_argument(
        '--gpu',
        default=0,
        type=int,
        help='GPU id')
    argparser.add_argument(
        '--ckpt',
        default=None,
        help='The path to ckpt')
    args = argparser.parse_args()
    return args


def get_config():
    """
    Create experimental config from argparse and config file.

    :return: Configuration EasyDict
    """
    # read manual args
    args = get_args()
    config_file = args.config

    # load experimental configuration
    if config_file.endswith('json'):
        config = get_config_from_json(config_file)
    elif config_file.endswith('yaml'):
        config = get_config_from_yaml(config_file)
    else:
        raise Exception("Only .json and .yaml are supported!")

    # reset config from args
    config.nStep = args.steps
    config.seed = args.seed
    config.gpu = args.gpu
    config.test = False if args.ckpt is None else True
    config.ckptPth = args.ckpt

    # create directories
    config.cacheDir = os.path.join("cache", '{}_{}shot_K{}_seed{}'.format(
        config.expName, config.nSupport, config.nStep, config.seed))
    config.logDir = os.path.join(config.cacheDir, 'logs')
    config.outDir = os.path.join(config.cacheDir, 'outputs')
    create_dirs([config.cacheDir, config.logDir, config.outDir])

    return config
