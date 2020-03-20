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

from setuptools import setup
import os
import sys

setup(
    name="finite_ntk",
    version="0.0",
    description="Linearisation for transfer learning",
    author="Wesley Maddox, Shuai Tang, Pablo Garcia Moreno, Andrew Gordon Wilson, Andreas Damianou",
    author_email="wjm363@nyu.edu, damianou@amazon.co.uk",
    license="Apache 2.0",
    packages=["finite_ntk"],
    include_package_data=True,
    classifiers=["Programming Language :: Python ::3.6"],
)
