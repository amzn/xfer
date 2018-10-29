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
import copy

from . import exceptions, consts


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

    def drop_layer_top(self, num_layers_to_drop=1, keep_branch_names=None):
        """
        Remove layers from output of model.

        :param int n: Number of layers to remove from model output.
        :param list[str] keep_branch_names: In cases where the top layer is ambiguous because there are branches, this
                                         should contain the name of the top layer of the branch to keep.
        """
        sym = self.symbol

        self._assert_drop_layer_valid(num_layers_to_drop)
        self._assert_model_has_single_output(self._get_symbol_dict(sym))

        # If keep_branch_names is not a list it can cause unexpected behaviour
        if keep_branch_names is not None:
            assert type(keep_branch_names) == list, 'keep_branch_names must be a list of strings'

        layers_dropped = []
        for n in range(num_layers_to_drop):
            # Get updated symbol dictionary
            symbol_dict = self._get_symbol_dict(sym)

            drop_layer_name = symbol_dict[consts.NODES][-1][consts.NAME]  # Get name of last layer
            last_layer_input_names = self._get_names_of_inputs_to_layer(symbol_dict=symbol_dict,
                                                                        node_idx=len(symbol_dict[consts.NODES]) - 1)
            new_last_layer = self._get_layer_name_ambiguous(available_layer_names=last_layer_input_names,
                                                            reference_layer_names=keep_branch_names, n=n)

            # If case is not ambiguous then new last layer is simply the penultimate layer
            if new_last_layer is None:
                next_operation = False
                for node in reversed(symbol_dict[consts.NODES]):
                    if next_operation:
                        if node[consts.OPERATION] != consts.NO_OP:
                            new_last_layer = node[consts.NAME]
                            break
                    if node[consts.NAME] == drop_layer_name:
                        next_operation = True

            layers_dropped.append(drop_layer_name)
            sym = sym.get_internals()[new_last_layer + consts.OUTPUT]

        logging.info('{} deleted from model top'.format(', '.join(layers_dropped)))
        if keep_branch_names is not None:
            if len(keep_branch_names) > 0:
                logging.warning('Did not use all of keep_branch_names: {}'.format(', '.join(keep_branch_names)))
        self.update_sym(sym)

    def drop_layer_bottom(self, num_layers_to_drop=1, drop_layer_names=None):
        """
        Remove layers from input of model.
        This method requires the entire symbol to be recreated internally.

        :param int n: Number of layers to remove from model input.
        """
        symbol_dict = self._get_symbol_dict(self.symbol)

        # Validate the action
        self._assert_drop_layer_valid(num_layers_to_drop)

        if drop_layer_names is not None:
            assert type(drop_layer_names) == list, 'drop_layer_names must be a list of strings'

        layers_dropped = []
        for n in range(num_layers_to_drop):
            original_nodes = copy.deepcopy(symbol_dict[consts.NODES])
            output_layer_names = self._get_output_layer_names(symbol_dict)
            nodes_using_zero_ids = self._get_layer_ids_with_node_zero_as_input(symbol_dict[consts.NODES])
            nodes_using_zero_ids_names = [v[consts.NAME] for c, v in enumerate(symbol_dict[consts.NODES])
                                          if c in nodes_using_zero_ids]
            drop_layer_name = self._get_layer_name_ambiguous(available_layer_names=nodes_using_zero_ids_names,
                                                             reference_layer_names=drop_layer_names, n=n)
            # If case is not ambiguous then the dropped layer is simply the first layer
            if drop_layer_name is None:
                drop_layer_name = self._get_name_of_first_operation(symbol_dict[consts.NODES])

            logging.info('Dropping {}'.format(drop_layer_name))
            layers_dropped.append(drop_layer_name)

            # Find node index of operator being deleted
            del_node_op_idx = self._get_layer_node_idx(symbol_dict[consts.NODES], drop_layer_name)
            symbol_dict = self._delete_layer_nodes_given_operator_node(symbol_dict, del_node_op_idx)

            symbol_dict, join_idx, join_deleted, join_layer_name = self._remove_redundant_join_layer(
                                                                    symbol_dict=symbol_dict,
                                                                    drop_layer_name=drop_layer_name,
                                                                    nodes_before=original_nodes,
                                                                    deleted_node_operator_idx=del_node_op_idx)
            if join_deleted:
                layers_dropped.append(join_layer_name)

            # Update symbol dictionary attributes
            symbol_dict[consts.ARG_NODES] = self._get_arg_nodes(symbol_dict[consts.NODES])
            symbol_dict[consts.HEADS] = self._get_heads(nodes=symbol_dict[consts.NODES],
                                                        output_layer_names=output_layer_names)
            symbol_dict[consts.NODES] = self._update_inputs(nodes=symbol_dict[consts.NODES],
                                                            original_nodes=original_nodes,
                                                            drop_layer_name=drop_layer_name,
                                                            join_deleted=join_deleted,
                                                            join_idx=join_idx)
        sym = self._get_symbol(symbol_dict)

        logging.info('{} deleted from model bottom'.format(', '.join(layers_dropped)))
        if drop_layer_names is not None:
            if len(drop_layer_names) > 0:
                logging.warning('Did not use all of drop_layer_names: {}'.format(', '.join(drop_layer_names)))
        self.update_sym(sym)

    def add_layer_top(self, layer_list):
        """
        Add layer to output of model.
        model layers = (layer1, layer2, layer3), layer_list = [layerA, layerB] -> model layers =
        (layer1, layer2, layer3, layerA, layerB)

        :param layer_list: List of MxNet symbol layers to be added to model output.
        :type layer_list: list(:class:`mx.symbol`)
        """
        if '_label' in self.symbol.get_internals().list_outputs()[consts.LABEL_IDX]:
            raise exceptions.ModelError('Cannot add layer above output layer')
        network_symbol = self._get_symbol_dict(self.symbol)
        # Concatentate nodes of new layers
        added_layer_names = []
        new_nodes = []
        for layer in layer_list:
            layer_nodes = self._get_symbol_dict(layer)[consts.NODES]
            # Shift input indices of layer nodes by the number of nodes in the existing network and the nodes added
            # before. -1 because input indexing begins at zero.
            layer_nodes = self._shift_input_indices(layer_nodes, len(network_symbol[consts.NODES]) + len(new_nodes) - 1)
            new_nodes += layer_nodes[1:]  # Excluding input node at index 0
            layer_name = layer_nodes[-1][consts.NAME]  # Last node contains layer name
            self._validate_layer_name(layer_name)
            added_layer_names.append(layer_name)

        # Concatentate entire network nodes list with new nodes list
        network_symbol[consts.NODES] = network_symbol[consts.NODES] + new_nodes
        # Update attributes of new network symbol dictionary
        network_symbol[consts.HEADS] = self._get_heads(nodes=network_symbol[consts.NODES],
                                                       output_layer_names=[added_layer_names[-1]])
        network_symbol[consts.ARG_NODES] = self._get_arg_nodes(network_symbol[consts.NODES])

        sym = self._get_symbol(network_symbol)
        self.update_sym(sym)
        logging.info('Added {} to model top'.format(', '.join(added_layer_names)))

    def add_layer_bottom(self, layer_list):
        """
        Add layer to input of model.
        model layers = (layer1, layer2, layer3), layer_list = [layerA, layerB] -> model layers =
        (layerA, layerB, layer1, layer2, layer3)

        :param layer_list: List of MxNet symbol layers to be added to model input.
        :type layer_list: list(:class:`mx.symbol`)
        """
        network_symbol = self._get_symbol_dict(self.symbol)
        # Concatentate nodes of new layers
        new_nodes = []
        added_layer_names = []
        for layer in layer_list:
            layer_nodes = self._get_symbol_dict(layer)[consts.NODES]
            # Shift input indices of new layer by number of nodes added before it
            layer_nodes = self._shift_input_indices(layer_nodes, len(new_nodes))
            new_nodes += layer_nodes[1:]  # adding all but input node
            layer_name = layer_nodes[-1][consts.NAME]  # Last node contains layer name
            self._validate_layer_name(layer_name)
            added_layer_names.append(layer_name)

        output_layer_names = self._get_output_layer_names(network_symbol)
        # Shift input indices of existing nodes by the number of nodes being added. Exclude input node.
        shifted_input_network_nodes = self._shift_input_indices(network_symbol[consts.NODES][1:], len(new_nodes))
        # Concatentate data node of network, new layer nodes and remaining network nodes
        network_symbol[consts.NODES] = [network_symbol[consts.NODES][0]] + new_nodes + shifted_input_network_nodes
        network_symbol[consts.HEADS] = self._get_heads(network_symbol[consts.NODES],
                                                       output_layer_names=output_layer_names)
        network_symbol[consts.ARG_NODES] = self._get_arg_nodes(network_symbol[consts.NODES])

        sym = self._get_symbol(network_symbol)
        self.update_sym(sym)
        logging.info('Added {} to model bottom'.format(', '.join(added_layer_names)))

    @staticmethod
    def _shift_input_indices(nodes, shift_constant):
        """
        Shift input indices of nodes by shift constant.
        """
        for node in nodes:
            for node_input in node[consts.INPUTS]:
                node_input[0] += shift_constant
        return nodes

    @staticmethod
    def _get_layer_node_idx(nodes, layer_name):
        for idx, node in enumerate(nodes):
            if node[consts.NAME] == layer_name:
                return idx
        raise ValueError("No node with name matching '{}'".format(layer_name))

    def _remove_redundant_join_layer(self, symbol_dict, drop_layer_name, nodes_before, deleted_node_operator_idx):
        """
        Remove a joining layer if it only has a single input and is therefore useless.
        """
        # Get list of indexes for nodes which have the zeroth node as an input
        nodes_using_zero_ids = self._get_layer_ids_with_node_zero_as_input(nodes_before)

        join_deleted = False
        join_idx = None
        join_layer_name = None
        # If more than one node had zero as input there may be a join layer than needs deleting
        if len(nodes_using_zero_ids) > 1:
            # Find joining node index
            join_idx = self._get_join_idx(nodes_using_zero_ids, symbol_dict[consts.NODES], nodes_before,
                                          drop_layer_name)
            # Delete input to join layer from deleted layer if it is there
            # MXNet 1.2.x uses the form [i, 0, 0] for inputs and 1.3.x uses [i, 0]
            for input_list in [[deleted_node_operator_idx, 0, 0], [deleted_node_operator_idx, 0]]:
                if input_list in symbol_dict[consts.NODES][join_idx][consts.INPUTS]:
                    symbol_dict[consts.NODES][join_idx][consts.INPUTS].remove(input_list)
            # Determine number of inputs to join node
            num_inputs = len(symbol_dict[consts.NODES][join_idx][consts.INPUTS])
            # Remove join layer if it has fewer than 2 inputs
            if num_inputs < 2:
                join_deleted = True
                join_layer_name = symbol_dict[consts.NODES][join_idx][consts.NAME]
                logging.info('Dropping {} (join node auto-deleted)'.format(join_layer_name))
                join_input = symbol_dict[consts.NODES][join_idx][consts.INPUTS][0]
                # Delete join nodes
                symbol_dict = self._delete_layer_nodes_given_operator_node(symbol_dict, join_idx)
                # Replace output of join with former input of join
                for i, node in enumerate(symbol_dict[consts.NODES]):
                    for j, input_list in enumerate(node[consts.INPUTS]):
                        if input_list[0] == join_idx:
                            symbol_dict[consts.NODES][i][consts.INPUTS][j] = join_input

        return symbol_dict, join_idx, join_deleted, join_layer_name

    def _get_layer_name_ambiguous(self, available_layer_names, reference_layer_names, n):
        """
        If there is no ambiguity in choosing the layer (i.e. there is only one available layer) then return
        None. Otherwise, use reference_layer_names for disambiguation and return layer name.
        """
        relevant_layer_name = None
        # If there are multiple available layers then this is an ambiguous case. So the first item in reference layer
        # names is used to disambiguate.
        if len(available_layer_names) > 1:
            try:
                relevant_layer_name = reference_layer_names.pop(0)
            # If an AttributeError is raised, it means that no layer names were given in reference_layer_names
            # If an IndexError is raised, it means that not enough layer names were given in reference_layer_names
            except (AttributeError, IndexError):
                raise exceptions.ModelError(self._ambiguous_layer_drop_error_message(
                                            available_layer_names, n))
            if relevant_layer_name is None or relevant_layer_name not in available_layer_names:
                raise exceptions.ModelError(self._ambiguous_layer_drop_error_message(
                                            available_layer_names, n))
        return relevant_layer_name

    @staticmethod
    def _get_name_of_first_operation(nodes):
        """
        Return the name of the first operation layer.
        """
        for node in nodes:
            if node[consts.OPERATION] != consts.NO_OP:
                return node[consts.NAME]
        raise exceptions.ModelError('nodes does not contain any operation nodes')

    @staticmethod
    def _get_idx_of_first_node_of_layer(operation_idx, arg_nodes):
        """
        Return the index of the first node of a layer given the index of the operation node of the layer and arg_nodes.
        """
        for node_idx in reversed(range(min(arg_nodes), operation_idx + 1)):  # +1 because range(a,b) doesn't include b
            # -1 because the first index is the index before the first that doesn't appear in arg nodes
            # Do not want to delete input (node 0)
            if (node_idx - 1) not in arg_nodes or (node_idx - 1) == 0:
                return node_idx
        raise exceptions.ModelError('Could not find first node of layer.')

    def _delete_layer_nodes_given_operator_node(self, symbol_dict, node_idx):
        """
        Return symbol dictionary with the nodes of a layer deleted. The layer to delete is given by the index of its
        operator node.
        """
        symbol_dict[consts.ARG_NODES] = self._get_arg_nodes(symbol_dict[consts.NODES])
        # Find first node for this layer
        first_idx = self._get_idx_of_first_node_of_layer(node_idx, symbol_dict[consts.ARG_NODES])
        for i in reversed(range(first_idx, node_idx+1)):  # Add 1 because range(a,b) doesn't include b
            del symbol_dict[consts.NODES][i]
        return symbol_dict

    @staticmethod
    def _get_arg_nodes(nodes):
        """
        Return arg_nodes given nodes list.
        """
        arg_nodes = []
        for idx, node in enumerate(nodes):
            if node[consts.OPERATION] == consts.NO_OP:
                arg_nodes.append(idx)
        return arg_nodes

    def _get_heads(self, nodes, output_layer_names):
        """
        Return heads given the nodes list and a list of output layer names.
        """
        heads = []
        for idx, node in enumerate(nodes):
            if node[consts.NAME] in output_layer_names:
                # symbol.load_json expects heads in the form [i, 0] (1.3.x), or [i, 0, 0] (1.2.x+)
                heads.append([idx, 0, 0])
        return heads

    @staticmethod
    def _get_name_input_map(nodes):
        """
        Get dictionary {layer name: [input layer names]} from nodes list.
        """
        input_map = {}
        for node in nodes:
            input_map[node[consts.NAME]] = []
            for i in node[consts.INPUTS]:
                input_map[node[consts.NAME]].append(nodes[i[0]][consts.NAME])
        return input_map

    def _update_inputs(self, nodes, original_nodes, drop_layer_name, join_deleted, join_idx):
        """
        Return the nodes list with correct input values.
        """
        name_input_map = self._get_name_input_map(original_nodes)
        # Get original index of dropped layer
        for drop_layer_original_idx, node in enumerate(original_nodes):
            if node[consts.NAME] == drop_layer_name:
                break
        # Check if dropped layer was originally an input to the join layer
        drop_layer_input_to_join = False
        if join_idx is not None:
            # Get join index for original nodes
            for join_idx_original, node in enumerate(original_nodes):
                if node[consts.NAME] == nodes[join_idx][consts.NAME]:
                    break
            for i in original_nodes[join_idx_original][consts.INPUTS]:
                if i[0] == drop_layer_original_idx:
                    drop_layer_input_to_join = True
                    break
        # We only want to ignore an input if a join was deleted or the dropped layer was the input to a join layer
        skip_inputs = []
        if join_deleted or drop_layer_input_to_join:
            skip_inputs = [drop_layer_name]
        name2idx = {node[consts.NAME]: idx for idx, node in enumerate(nodes)}  # Generate name to idx mapping
        for node_id, node in enumerate(nodes):
            # If node has no inputs then skip it
            if len(node[consts.INPUTS]) == 0:
                continue
            inputs = name_input_map[node[consts.NAME]]
            inputs_set = set(inputs)
            available_layers = list(name2idx.keys())
            # Continue to loop until the input names are all in the list of available layers
            while not inputs_set.issubset(available_layers):
                unavailable_inputs = [i for i in inputs if i not in available_layers]
                replacement_inputs = []
                # Replace unavailable nodes with their inputs unless in skip_inputs
                for unavailable_input in unavailable_inputs:
                    replacement_inputs = replacement_inputs + [i for i in name_input_map[unavailable_input]
                                                               if i not in skip_inputs]
                inputs = [i for i in inputs if i in available_layers] + replacement_inputs
                inputs_set = set(inputs)

            inputs = sorted([name2idx[name] for name in inputs])
            # symbol.load_json expects inputs in the form [i, 0] (1.3.x), or [i, 0, 0] (1.2.x+)
            nodes[node_id][consts.INPUTS] = [[i, 0, 0] for i in inputs]
        return nodes

    @staticmethod
    def _get_output_layer_names(symbol_dict):
        """
        Return names of output layers given symbol dictionary.
        """
        return [symbol_dict[consts.NODES][i[0]][consts.NAME] for i in symbol_dict[consts.HEADS]]

    @staticmethod
    def _assert_model_has_single_output(symbol_dict):
        """
        Raise ModelError if model has more than one output.
        """
        output_layer_names = []
        if len(symbol_dict[consts.HEADS]) > 1:
            for head in symbol_dict[consts.HEADS]:
                output_layer_names.append(symbol_dict[consts.NODES][head[0]][consts.NAME])
            raise exceptions.ModelError(
                'ModelHandler does not support this operation for models with more than one output. ({})'.format(
                    ', '.join(output_layer_names)))

    @staticmethod
    def _get_names_of_inputs_to_layer(symbol_dict, node_idx):
        """
        Get the names of the layers that are inputs to specified layer.
        """
        # Assert node_idx refers to an operation
        assert symbol_dict[consts.NODES][node_idx][consts.OPERATION] != consts.NO_OP,\
            'node_idx: {} does not refer to a layer'.format(node_idx)
        # Get list of layer inputs to layer
        inputs_to_layer = []
        for i in symbol_dict[consts.NODES][node_idx][consts.INPUTS]:
            # Do not add null layers to list of inputs
            if symbol_dict[consts.NODES][i[0]][consts.OPERATION] != consts.NO_OP:
                inputs_to_layer.append(i[0])
        return [node[consts.NAME] for idx, node in enumerate(symbol_dict[consts.NODES])
                if idx in inputs_to_layer]

    @staticmethod
    def _ambiguous_layer_drop_error_message(layer_names, n):
        """
        Return error message for ambiguous layer drops.
        """
        return 'Found an ambiguous layer (drop layer number: {}). Please choose one from: {}'\
            .format(n, ', '.join(layer_names))

    @staticmethod
    def _get_layer_ids_with_node_zero_as_input(nodes):
        """
        Get list of ids of layers that have the zeroth node as an input.
        """
        layer_ids = []
        for idx, node in enumerate(nodes):
            for input_list in node[consts.INPUTS]:
                if input_list[0] == 0:
                    layer_ids.append(idx)
                    # Avoid counting twice in the case where a node has two inputs from 0 node
                    break
        return layer_ids

    @staticmethod
    def _get_join_idx(nodes_using_zero_ids, nodes, nodes_before_layer_deleted, drop_layer_name):
        """
        Get index of the node closest to the input that joins two or more layers including the dropped layer
        (directly or indirectly).
        """
        # Get index of drop_layer
        for drop_layer_idx, node in enumerate(nodes_before_layer_deleted):
            if node[consts.NAME] == drop_layer_name:
                break
        # Create dictionary with key for each node using zeroth node as input
        parents = {node_idx: [] for node_idx in nodes_using_zero_ids}
        # We want to find the nodes with nodes_using_zero_ids as input
        next_interest_nodes = nodes_using_zero_ids
        while True:
            # Update interest nodes
            current_interest_nodes = next_interest_nodes
            next_interest_nodes = []
            # Loop through original nodes
            for idx, node in enumerate(nodes_before_layer_deleted):
                for i in node[consts.INPUTS]:
                    # If input is in interest nodes then add it to the parent dictionary with the key being the
                    # relevant node
                    if i[0] in current_interest_nodes:
                        for key in parents.keys():
                            if i[0] in parents[key] or i[0] == key:
                                parents[key].append(idx)
                        # Add any newly found parent nodes to be the next interest nodes
                        next_interest_nodes.append(idx)
            # Find intersection with dropped layer parent list and other parent lists
            intersection = []
            for k, v in parents.items():
                if k == drop_layer_idx:
                    continue
                intersection = intersection + list(set(parents[drop_layer_idx]).intersection(v))
            if len(intersection) > 0:
                break

        # The first node they have in common is the join node
        join_idx = min(list(intersection))

        join_name = nodes_before_layer_deleted[join_idx][consts.NAME]
        # Get node index in current node list
        for join_idx, node in enumerate(nodes):
            if node[consts.NAME] == join_name:
                break
        return join_idx

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

        :param str layer_type: Return list of layers of this type.
        :return: Names of layers with specified type
        :rtype: list(str)
        """
        return [layer_name for layer_name, l_type in self.layer_type_dict.items()
                if l_type.lower() == layer_type.lower()]

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
        symbol_dict = self._get_symbol_dict(self.symbol)
        # Each layer in the model has one output (along with a bias or weight etc.) so we filter for all the list
        # entries containing '_output' and then we remove this substring from the list entries to leave just the layer
        # names
        layer_names = [l.replace('_output', '') for l in self.symbol.get_internals().list_outputs() if '_output' in l]
        symbol_layers = [node for node in symbol_dict[consts.NODES] if node[consts.LAYER_NAME] in layer_names]
        layer_type_dict = OrderedDict()
        for layer in symbol_layers:
            layer_type_dict[layer[consts.LAYER_NAME]] = layer[consts.OPERATION]
        return layer_type_dict

    @staticmethod
    def _get_symbol_dict(symbol):
        """
        Get symbol dictionary.

        :return: Symbol dictionary
        :rtype: dict
        """
        return json.loads(symbol.tojson())

    @staticmethod
    def _get_symbol(symbol_dict):
        """
        Get MXNet symbol from its symbol dictionary.
        """
        return mx.sym.load_json(json.dumps(symbol_dict))

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
