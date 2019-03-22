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
# -*- coding: utf-8 -*-
from datetime import datetime

# import pkg_resources
import sys
import os
from unittest.mock import MagicMock


sys.path.insert(0, os.path.abspath('../'))


class Mock(MagicMock):
    @classmethod
    def __getattr__(cls, name):
        return MagicMock()

MOCK_MODULES = ['scipy', 'matplotlib', 'GPy', 'sklearn', 'sklearn.svm', 'sklearn.linear_model',
                'sklearn.preprocessing']

sys.modules.update((mod_name, Mock()) for mod_name in MOCK_MODULES)


exec(open("../xfer/__version__.py").read())


version = __version__
project = 'xfer'


# Add any Sphinx extension module names here, as strings. They can be extensions
# coming with Sphinx (named 'sphinx.ext.*') or your custom ones.
extensions = ['sphinx.ext.autodoc', 'sphinx.ext.doctest',
              'sphinx.ext.intersphinx', 'sphinx.ext.todo',
              'sphinx.ext.coverage', 'sphinx.ext.autosummary',
              'sphinx.ext.napoleon', 'nbsphinx', 'sphinx.ext.mathjax',
              'sphinx.ext.ifconfig']
exclude_patterns = ['_build', '**.ipynb_checkpoints', '._**']

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

source_suffix = '.rst'  # The suffix of source filenames.
master_doc = 'index'  # The master toctree document.

copyright = '{} Amazon.com, Inc. or its affiliates. All Rights Reserved. SPDX-License-Identifier: \
             Apache-2.0'.format(datetime.now().year)

# The full version, including alpha/beta/rc tags.
release = version

# List of directories, relative to source directory, that shouldn't be searched for source files.
exclude_trees = ['_build']

pygments_style = 'sphinx'

autoclass_content = "both"
autodoc_default_flags = ['show-inheritance', 'members', 'undoc-members', 'inherited-members']
autodoc_member_order = 'bysource'

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_style = 'override.css'
htmlhelp_basename = u'{}doc'.format(project)
html_logo = 'image/logo.png'
html_favicon = 'image/favicon.png'
html_show_sphinx = False

# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {'http://docs.python.org/': None}

# autosummary
autosummary_generate = True
