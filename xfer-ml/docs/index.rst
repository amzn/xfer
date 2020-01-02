.. Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
.. 
.. Licensed under the Apache License, Version 2.0 (the "License").
.. You may not use this file except in compliance with the License.
.. A copy of the License is located at
.. 
..     http://www.apache.org/licenses/LICENSE-2.0
.. 
.. or in the "license" file accompanying this file. This file is distributed 
.. on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either 
.. express or implied. See the License for the specific language governing 
.. permissions and limitations under the License.


Welcome to Xfer's documentation!
============================================

| **Release:**  |version|
| **Date:**     |today|

"""""""""""""""""""""""""""""""

Xfer is a `Transfer Learning <https://en.wikipedia.org/wiki/Transfer_learning/>`_  framework written in Python.


Xfer features Repurposers that can be used to take an MXNet model and train a meta-model or modify the 
model for a new target dataset. To get started with Xfer checkout our introductory tutorial `here <demos/xfer-overview.ipynb>`_.

The code can be found on our `Github project page <https://github.com/amzn/xfer/>`_. It is open source and provided using the Apache 2.0 license.


.. toctree::
   :maxdepth: 1
   :caption: Tutorials
   
   demos/xfer-overview.ipynb
   demos/xfer-modelhandler.ipynb
   demos/xfer-text-transfer.ipynb
   demos/xfer-hpo.ipynb
   demos/xfer-gluon-with-modelhandler.ipynb
   demos/xfer-gluon-source-model.ipynb

   
.. toctree::
  :maxdepth: 1
  :caption: API
      
  api.rst


.. toctree::
  :maxdepth: 1
  :caption: For developers
  
  demos/xfer-custom-repurposers.ipynb


Indices and tables
__________________

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
