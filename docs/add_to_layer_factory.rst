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

Adding to Layer Factory
************************

ModelHandler has been implemented only with a subset of MXNet's available operators.  The API has been designed so that the process for adding additional layer types is as simple and easy as possible.

In this demo, the Fully Connected layer type will be used as an example to show the steps required to add a new layer type to Xfer's ModelHandler.

The first step in implementing a new operator is to define a new :code:`LayerFactory` class.  The arguments for the constructor of this class should be the arguments that the layer operation can take with defaults set appropriately.

.. code-block:: python

   class FullyConnected(LayerFactory):
       def __init__(self, name, num_hidden, no_bias=False):

Add the layer type to the :code:`LayerType Enum` in :code:`model_handler.consts.py`:

.. code-block:: python

   FULLYCONNECTED = 'FullyConnected'

In the :code:`__init__` method you need to call the parent class :code:`__init__` method with the argument :code:`name` and set all the :code:`attributes`, :code:`layer_type` and :code:`SymbolClass`.

.. code-block:: python

   def __init__(self, name, num_hidden, no_bias=False):
       super(FullyConnected, self).__init__(name=name)
       self.attributes['num_hidden'] = num_hidden
       self.attributes['no_bias'] = no_bias
       self.layer_type = consts.LayerType.FULLYCONNECTED.value
       self.SymbolClass = mx.sym.FullyConnected
   
In order for ModelHandler to modify the input of a model containing your new layer type, the class must implement the :code:`_from_dict` method.
This method should take the layer dictionary as generated when the MXNet symbol is converted to json with :code:`mxnet.symbol.Symbol.tojson()` and return the :code:`LayerFactory` class and input symbol.

.. code-block:: python

   def _from_dict(layer_dict):
       name = layer_dict[consts.NAME]
       num_hidden = int(layer_dict[consts.ATTRIBUTES]['num_hidden'])
       no_bias = layer_dict[consts.ATTRIBUTES].get('no_bias', 'False') == 'True'
       input_symbol = _get_input_symbol(layer_dict)
       return FullyConnected(name=name, num_hidden=num_hidden, no_bias=no_bias), input_symbol

By default, LayerFactory objects will implement create_layer as below:

.. code-block:: python

   return self.SymbolClass(data=layer_input, **self.attributes)
   
This will work with most operators but in some cases you will need to implement your own :code:`create_layer()` which returns an MXNet symbol.

This is the full :code:`LayerFactory` class:

.. code-block:: python

   class FullyConnected(LayerFactory):
       """
       Class for fully connected layers.

       :param str name: Name of the resulting layer.
       :param int num_hidden: Number of hidden nodes of the output.
       :param boolean no_bias: Whether to disable bias parameter.
       """
       def __init__(self, name, num_hidden, no_bias=False):
           super(FullyConnected, self).__init__(name=name)
           self.attributes['num_hidden'] = num_hidden
           self.attributes['no_bias'] = no_bias
           self.layer_type = consts.LayerType.FULLYCONNECTED.value
           self.SymbolClass = mx.sym.FullyConnected

       def _from_dict(layer_dict):
           name = layer_dict[consts.NAME]
           num_hidden = int(layer_dict[consts.ATTRIBUTES]['num_hidden'])
           no_bias = layer_dict[consts.ATTRIBUTES].get('no_bias', 'False') == 'True'
           input_symbol = _get_input_symbol(layer_dict)
           return FullyConnected(name=name, num_hidden=num_hidden, no_bias=no_bias), input_symbol
           
The last step of this process is to add the new layer to :code:`class_dict` in :code:`LayerFactory._from_dict()` as :code:`class_dict[op] = class_name`
For example:

.. code-block:: python

   class_dict = {
       'FullyConnected': FullyConnected
   }

