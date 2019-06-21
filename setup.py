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
from setuptools import setup, find_packages
import sys


with open('docs/long_description.md', 'r') as fh:
    long_description = fh.read()


requires = [
    'mxnet>=1.2.0',
    'numpy>=1.10.1',  # GPy import throws error for lower versions
    'scikit-learn>=0.19.2',
    'matplotlib>=2.2.2',
    'GPy>=1.9.5'
]


# get __version__ variable
exec(open("xfer/__version__.py").read())


# enforce >Python3 for all versions of pip/setuptools
assert sys.version_info >= (3,), 'This package requires Python 3.'


setup(
    name='xfer-ml',
    version=__version__,  # noqa
    description='Lightweight deep transfer learning library built on MXNet',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/amzn/xfer',
    packages=find_packages(exclude=['test*']),
    include_package_data=True,
    install_requires=requires,
    python_requires='>=3',  # requires setuptools>=24.2.0 for packaging and pip>=9.0.0 on download for this to work
    license='Apache License 2.0',
    classifiers=(
        # https://pypi.org/pypi?%3Aaction=list_classifiers
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules'
    )
)
