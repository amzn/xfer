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
import json
import logging
import gc
import mxnet as mx
import numpy as np
from collections import OrderedDict

from . import exceptions, consts, layer_factory


class ModelHandler(object):
    """
    Class for model manipulation and feature extraction.

    :param module: MXNet module to be manipulated.
    :type module: :class:`mx.module.Module`
    :param function context_function: MXNet context function.
    :param int num_devices: Number of devices to run process on.
    :param str data_name: Name of input layer of model.
    """
    def __init__(self, module, context_function=mx.context.cpu, num_devices=1, data_name=consts.DATA):
        self.symbol = module.symbol
        if module.binded and module.params_initialized:
            self.arg_params, self.aux_params = module.get_params()
        else:
            self.arg_params = module._arg_params if module._arg_params is not None else {}
            self.aux_params = module._aux_params if module._aux_params is not None else {}

        self.layer_type_dict = self._get_layer_type_dict()
        self.devices = self._get_devices(context_function, num_devices)
        self.data_name = data_name

    def drop_layer_top(self, num_layers_to_drop=1):
        """
        Remove layers from output of model.

        :param int n: Number of layers to remove from model output.
        """
        self._assert_drop_layer_valid(num_layers_to_drop)
        # Get index of new last layer
        last_layer_idx = -(num_layers_to_drop+1)
        # symbol ending at new last layer
        sym = self.symbol.get_internals()[self.layer_names[last_layer_idx] + '_output']
        logging.info('{} deleted from model top'.format(', '.join(self.layer_names[last_layer_idx + 1:])))
        self.update_sym(sym)

    def drop_layer_bottom(self, num_layers_to_drop=1):
        """
        Remove layers from input of model.
        This method requires the entire symbol to be recreated internally.

        :param int n: Number of layers to remove from model input.
        """
        self._assert_drop_layer_valid(num_layers_to_drop)
        symbol_nodes = self._get_symbol_dict()[consts.NODES]
        # Get index of nth layer
        layer_count = 0
        n_layer_idx = 0
        for id, node in enumerate(symbol_nodes):
            if node[consts.OPERATION] != consts.NO_OP:
                layer_count += 1
            if layer_count == num_layers_to_drop:
                n_layer_idx = id
                break
        # Raise error if nth layer is not found
        if n_layer_idx == 0:
            raise exceptions.ModelError('{}th layer could not be found'.format(num_layers_to_drop))
        # Remove nodes in list that relate to first n layers (keep first node because this is data layer)
        symbol_nodes = symbol_nodes[0:1] + symbol_nodes[n_layer_idx+1:]
        id2name = {count: node[consts.LAYER_NAME] for count, node in enumerate(symbol_nodes)}
        # Recreate model from symbol nodes
        sym = self._model_from_nodes(None, symbol_nodes, elements_offset=-n_layer_idx, id2name=id2name,
                                     data_name=self.data_name)
        logging.info('{} deleted from model bottom'.format(', '.join(self.layer_names[:num_layers_to_drop])))
        self.update_sym(sym)

    def add_layer_top(self, layer_factory_list):
        """
        Add layer to output of model.

        :param layer_factory: List of LayerFactory objects to be added to model output.
        :type layer_factory: list(:class:`LayerFactory`)
        """
        if '_label' in self.symbol.get_internals().list_outputs()[consts.LABEL_IDX]:
            raise exceptions.ModelError('Cannot add layer above output layer')
        sym = self.symbol
        added_layer_names = []
        for lf in layer_factory_list:
            self._validate_layer_name(lf.attributes[consts.NAME])
            sym = lf.create_layer(sym)
            added_layer_names.append(lf.attributes[consts.NAME])
        self.update_sym(sym)
        logging.info('Added {} to model top'.format(', '.join(added_layer_names)))

    def add_layer_bottom(self, layer_factory_list):
        """
        Add layer to input of model.
        This method requires the entire symbol to be recreated internally.

        :param layer_factory: List of LayerFactory objects to be added to model input.
        :type layer_factory: list(:class:`LayerFactory`)
        """
        # Create data layer
        symbol_nodes = self._get_symbol_dict()[consts.NODES]
        data_layer = layer_factory.Data(name=self.data_name)
        sym = data_layer.create_layer(None)
        prev_symbols = {self.data_name: sym}
        id2name = {0: self.data_name}
        added_layer_names = []
        # Add new layers
        for id, lf in enumerate(layer_factory_list):
            self._validate_layer_name(lf.attributes[consts.NAME])
            if lf.output is True:
                raise exceptions.ModelError('Cannot add output layer to model bottom ({})'
                                            .format(lf.attributes[consts.NAME]))
            sym = lf.create_layer(sym)
            prev_symbols[lf.attributes[consts.NAME]] = sym
            # Update id2name with newly added layer
            id2name[id + 1] = lf.attributes[consts.NAME]
            added_layer_names.append(lf.attributes[consts.NAME])
        # Add remaining symbol names to id2name (beginning slice at 1 to skip data node which is already in id2name)
        for count, node in enumerate(symbol_nodes[1:]):
            id2name[count + len(layer_factory_list) + 1] = node[consts.LAYER_NAME]
        # Recreate model
        sym = self._model_from_nodes(sym, symbol_nodes, elements_offset=len(layer_factory_list),
                                     prev_symbols=prev_symbols, id2name=id2name, data_name=self.data_name)
        self.update_sym(sym)
        logging.info('Added {} to model bottom'.format(', '.join(added_layer_names)))

    def get_module(self, iterator, fixed_layer_parameters=None, random_layer_parameters=None):
        """
        Return MXNet Module using the model symbol and parameters.

        :param iterator: MXNet iterator to be used with model.
        :type iterator: :class:`mxnet.io.DataIter`
        :param list(str) fixed_layer_parameters: List of layer parameters to keep fixed.
        :param list(str) random_layer_parameters: List of layer parameters to randomise.
        :return: MXNet module
        :rtype: :class:`mx.module.Module`
        """
        if fixed_layer_parameters is not None:
            fixed_layer_parameters = self._prune_parameters(fixed_layer_parameters)
        if random_layer_parameters is None:
            arg_params, aux_params = self.arg_params.copy(), self.aux_params.copy()
        else:
            arg_params, aux_params = self._remove_random_parameters(random_layer_parameters)
        mod = mx.mod.Module(symbol=self.symbol, context=self.devices, fixed_param_names=fixed_layer_parameters,
                            label_names=(self.layer_names[-1] + "_label",), data_names=(self.data_name,))
        mod.bind(data_shapes=iterator.provide_data, label_shapes=iterator.provide_label)
        mod.init_params(mx.init.Xavier(rnd_type='gaussian', factor_type='in', magnitude=2))
        try:
            mod.set_params(arg_params, aux_params, allow_missing=True, force_init=True)
        except mx.MXNetError as e:
            exceptions._handle_mxnet_error(e)
        return mod

    def get_layer_type(self, layer_name):
        """
        Return type of named layer.

        :param str name: Name of layer being inspected.
        :return: Layer type
        :rtype: str
        """
        try:
            return self.layer_type_dict[layer_name]
        except KeyError:
            raise ValueError('{} layer does not exist in model'.format(layer_name))

    def get_layer_names_matching_type(self, layer_type):
        """
        Return names of layers of specified type.

        :param LayerType layer_type: Return list of layers of this type. Should be a LayerType enum.
        :return: Names of layers with specified type
        :rtype: list(str)
        """
        return [layer_name for layer_name, l_type in self.layer_type_dict.items() if l_type == layer_type.value]

    def get_layer_output(self, data_iterator, layer_names):
        """
        Function to extract features from data iterator with model.
        Returns a dictionary of layer_name -> numpy array of features extracted (flattened).

        :param data_iterator: Iterator containing input data.
        :type data_iterator: :class:`mxnet.io.DataIter`
        :param list(str) layer_names: List of names of layers to extract features from.
        :return: Ordered Dictionary of features ({layer_name: features}), list of labels
                 Layer names in the ordered dictionary follow the same order as input list of layer_names
        :rtype: OrderedDict[str, :class:`numpy.array`], list(int)
        """
        if not isinstance(data_iterator, mx.io.DataIter):
            raise TypeError('Iterator should be an MXNet DataIter object'.format(type(data_iterator)))
        if type(layer_names) is not list:
            raise TypeError('Layers must be passed as list.')
        logging.info('Extracting features from layers: ' + ' '.join(layer_names))

        features = OrderedDict()
        labels = np.array([])
        data_shape = data_iterator.provide_data[0][1]
        batch_no = 0
        id2layer = dict(enumerate(layer_names))
        data_iterator.reset()

        intermediate_layer_symbols = [self.symbol.get_internals()[l + '_output'] for l in layer_names]
        intermediate_symbol = mx.sym.Group(intermediate_layer_symbols)

        module = mx.mod.Module(symbol=intermediate_symbol, context=self.devices, label_names=None,
                               data_names=(self.data_name,))
        module.bind(for_training=False, data_shapes=[(self.data_name, data_shape)], label_shapes=None)
        module.set_params(self.arg_params, self.aux_params)

        # MXNet Module has issues releasing memory from GPU (https://github.com/apache/incubator-mxnet/issues/5983)so we
        # force full garbage collection before feature extraction and force a shallow garbage collection after every
        # loop to free up any memory that has been allocated during the loop and should have been released
        gc.collect(2)  # Full garbage collection before begininning to extract features

        while True:
            try:
                batch = data_iterator.next()
            except StopIteration:
                break
            batch_no += 1
            labels = np.append(labels, batch.label[0].asnumpy())
            module.forward(batch)
            forward_output = module.get_outputs()

            for lid in id2layer:
                feature = forward_output[lid].asnumpy().flatten().reshape(data_shape[0], -1)
                # Remove extra values from padding if they are present
                feature = feature[:feature.shape[0]-batch.pad]
                if id2layer[lid] in features:
                    features[id2layer[lid]] = np.vstack((features[id2layer[lid]], feature))
                else:
                    features[id2layer[lid]] = feature
            # Remove extra values from padding if they are present
            labels = labels[:labels.shape[0]-batch.pad]

            logging.info('Processed batch {0}'.format(batch_no))
            del forward_output
            del feature
            del batch
            gc.collect(0)  # Shallow garbage collect for objects created during loop

        return features, labels.astype(int)

    def get_layer_parameters(self, layer_names):
        """
        Get list of layer parameters associated with the the layer names given.

        :param list(str) layer_names: List of layer names.
        :return: List of layer parameters
        :rtype: list(str)
        """
        if type(layer_names) is not list:
            raise TypeError('layer_names must be passed as list.')
        params = []
        for layer_name in layer_names:
            for arg_dict in [self.arg_params, self.aux_params]:
                new_params = [param for param in list(arg_dict.keys()) if layer_name in param]
                params = params + new_params
        return params

    def visualize_net(self):
        """
        Display computational graph of model.
        """
        return mx.viz.plot_network(self.symbol, node_attrs={'fixedsize': 'false'})

    def save_symbol(self, model_name):
        """
        Serialise model symbol graph.

        :param str model_name: Prefix to file name (model_name-symbol.json).
        """
        self.symbol.save(model_name + '-symbol.json')

    def _validate_layer_name(self, layer_name):
        """
        Validate name of layer.

        :param str layer_name: Name to be validated.
        """
        # Input name is not included in layer_names so the check for conflict is done with symbol inputs and layer_names
        if layer_name in self.symbol.list_inputs() or layer_name in self.layer_names:
            raise ValueError("Layer name '{}' conflicts with name already in model.".format(layer_name))
        # MXNet uses these suffixes for specific things so we are avoiding using them in layer names
        for suffix in ['output', 'label', 'weight', 'bias', 'moving_mean', 'moving_var', 'gamma', 'beta']:
            if '_' + suffix in layer_name:
                raise ValueError("Layer name cannot contain '{}'".format(suffix))

    @staticmethod
    def _model_from_nodes(symbol, symbol_nodes, elements_offset=0, prev_symbols=None, id2name=None,
                          data_name=consts.DATA):
        """
        Recreate model from dictionary above symbol.

        :param :class:`LayerFactory` symbol: Symbol on which to create model.
        :param list symbol_nodes: List of dictionaries which desribe model layers.
        :param int elements_offset: Number of layers added to model since symbol dictionary was last updated.
        :param dict(str, :class:`Symbol`) prev_symbols: Dictionary of all MXNet symbols corresponding to previously
                                                        added layers -- layer_name: MXNet symbol.
        :param dict(int, str) id2name: Names of symbols in model -- {position of symbol in model: symbol name}.
        :rtype: :class:`Symbol`
        """
        prev_symbols = {} if prev_symbols is None else prev_symbols
        for count, node in enumerate(symbol_nodes):
            node[consts.ELEMENTS_OFFSET] = elements_offset
            node[consts.PREV_SYMBOLS] = prev_symbols
            node[consts.ID2NAME] = id2name
            node[consts.DATA] = data_name
            new_layer, input_symbol = layer_factory.LayerFactory._from_dict(node)
            # If node does not represent a layer operation then new_layer will be None
            if new_layer:
                symbol = new_layer.create_layer(input_symbol)
                prev_symbols[new_layer.attributes[consts.NAME]] = symbol
        return symbol

    def _prune_parameters(self, parameter_names):
        """
        Remove parameter names from list which are not in model parameter dicts.  Logs warning for removed names.

        :param list(str) parameter_names: List of parameter names.
        :return: Parameter names that are all in the model parameter dicts
        :rtype: list(str)
        """
        all_keys = list(self.arg_params.keys()) + list(self.aux_params.keys())
        names_not_found = [parameter_name for parameter_name in parameter_names if parameter_name not in all_keys]
        if len(names_not_found) > 0:
            logging.warning('Could not find layer parameters: {}'.format(', '.join(names_not_found)))
        return [parameter_name for parameter_name in parameter_names if parameter_name in all_keys]

    def _remove_random_parameters(self, random_parameter_names):
        """
        Return model parameter dicts with random parameters removed.

        :param list(str) random_parameter_names: Names of random parameters to be removed from model parameter dicts.
        :return: arg_params, aux_params
        :rtype: dict, dict
        """
        random_parameter_names = self._prune_parameters(random_parameter_names)
        arg_params = self.arg_params.copy()
        aux_params = self.aux_params.copy()
        for parameter_name in random_parameter_names:
            arg_params.pop(parameter_name, None)
            aux_params.pop(parameter_name, None)
        return arg_params, aux_params

    @staticmethod
    def _get_devices(context_function=mx.context.cpu, num_devices=1):
        """
        Return devices list.

        :param function context_function: MXNet context function which returns a context.
        :param int num_devices: Number of devices to use for processing (e.g number of cpus or gpus to use).
        :return: List of devices
        :rtype: list
        """
        return [context_function(i) for i in range(num_devices)]

    def _assert_drop_layer_valid(self, n):
        """
        Raise exception if the number of layers being dropped is more than or equal to the number of layers in the
        model.
        """
        if len(self.layer_names) < n + 1:
            raise exceptions.ModelError('Cannot drop {} layers. Model only has {} layers'.format(n,
                                                                                                 len(self.layer_names)))

    def _get_layer_type_dict(self):
        """
        Return dictionary of layer types.

        :return: Dictionary of layer names to layer types
        :rtype: dict
        """
        symbol_dict = self._get_symbol_dict()
        # Each layer in the model has one output (along with a bias or weight etc.) so we filter for all the list
        # entries containing '_output' and then we remove this substring from the list entries to leave just the layer
        # names
        layer_names = [l.replace('_output', '') for l in self.symbol.get_internals().list_outputs() if '_output' in l]
        symbol_layers = [node for node in symbol_dict[consts.NODES] if node[consts.LAYER_NAME] in layer_names]
        layer_type_dict = OrderedDict()
        for layer in symbol_layers:
            layer_type_dict[layer[consts.LAYER_NAME]] = layer[consts.OPERATION]
        return layer_type_dict

    def _get_symbol_dict(self):
        """
        Get symbol dictionary.

        :return: Symbol dictionary
        :rtype: dict
        """
        return json.loads(self.symbol.tojson())

    @property
    def layer_names(self):
        """
        Get list of names of model layers.

        :return: List of layer names
        :rtype: list[str]
        """
        return list(self.layer_type_dict.keys())

    @staticmethod
    def _clean_params(symbol, parameter_dict):
        """
        Return a copy of parameter_dict with parameters for layers that are not in the symbol removed.

        :param symbol: Symbol to give point of reference for removing parameters.
        :type symbol: :class:`mx.symbol.Symbol`
        :param dict parameter_dict: Dictionary of model parameters.
        :return: Parameter dictionary with all parameters referring to layer(s) in the symbol
        :rtype: dict
        """
        parameter_dict = parameter_dict.copy()
        keys_to_delete_arg = set(parameter_dict.keys()) - set(symbol.get_internals().list_outputs())
        for key in keys_to_delete_arg:
            del parameter_dict[key]
        return parameter_dict

    def update_sym(self, new_symbol):
        """
        Update symbol attribute, layer names, and layer types dict and clean parameters.

        :param new_symbol: Symbol with which to update ModelHandler.
        :type new_symbol: :class:`mx.symbol.Symbol`
        """
        self.symbol = new_symbol
        self.layer_type_dict = self._get_layer_type_dict()
        self.arg_params = self._clean_params(self.symbol, self.arg_params)
        self.aux_params = self._clean_params(self.symbol, self.aux_params)
