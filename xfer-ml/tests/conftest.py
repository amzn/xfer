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
import pytest


test_flags = ['notebook', 'notebook_hpo', 'notebook_gluon', 'integration']


def pytest_addoption(parser):
    for option in test_flags:
        parser.addoption(
            "--{}".format(option), action="store_true", default=False, help="run {} tests".format(option)
        )


def pytest_collection_modifyitems(config, items):
    for option in test_flags:
        if config.getoption("--{}".format(option)):
            # --option given in cli: do not skip 'option' tests
            continue
        skip_test = pytest.mark.skip(reason="need --{} option to run".format(option))
        for item in items:
            if option in item.keywords:
                item.add_marker(skip_test)
