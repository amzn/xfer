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
import numpy as np
import mxnet as mx
import mxnet.ndarray as nd
from mxnet import autograd, gluon
from sklearn.preprocessing import Normalizer
import logging

from .meta_model_repurposer import MetaModelRepurposer
from .prob import GaussianPrior, GaussianVariationalPosterior, BNNLoss, Categorical
from .bnn_classifier import BnnClassifier
from .prob.utils import replace_params_net
from .constants import repurposer_keys, serialization_constants, serialization_keys
from .constants import bnn_repurposer_keys as keys
from .constants import bnn_constants as consts

from . import utils


class BnnRepurposer(MetaModelRepurposer):
    """
    Perform Transfer Learning through a Bayesian Neural Network (BNN) meta-model which repurposes the source neural
    network.

    :param source_model: Source neural network to do transfer learning from.
    :type source_model: :class:`mxnet.mod.Module`
    :param feature_layer_names: Name of layer(s) in source_model from which features should be transferred.
    :type feature_layer_names: list[str]
    :param context_function: MXNet context function that provides device type context. It is used to extract features
        with the source model.
    :type context_function: function(int)->:class:`mx.context.Context`
    :param int num_devices: Number of devices to use to extract features from source_model.
    :param bnn_context_function: MXNet context function used to train the BNN.
    :type bnn_context_function: function(int)->:class:`mx.context.Context`
    :param float sigma: Standard deviation of the Gaussian prior used for the weights of the BNN meta-model
        (w_i \sim N(0, \sigma^2)).
    :param int num_layers: Number of layers of the BNN meta-model.
    :param int n_hidden: Dimensionality of the hidden layers of the BNN meta-model
        (all hidden layers have the same dimensionality).
    :param int num_samples_mc: Number of samples used for the Monte Carlo approximation of the variational bound.
    :param float learning_rate: Learning rate for the BNN meta-model training.
    :param int batch_size: Mini-batch size for the BNN meta-model training.
    :param int num_epochs: Number of epochs for the BNN meta-model training.
    :param int start_annealing: To help the training of the BNN meta-model, we anneal the KL term using a weight
        that varies fromm zero to one. start_annealing determines the epoch at which the annealing weight start
        to increase linearly until it reaches one in the epoch given by end_annealing.
    :param int end_annealing: Determines the epoch at which the annealing process of the KL term ends.
    :param float step_annealing_sample_weight: Amount that the annealing weight is incremented in each epoch
       (from start_annealing to end_annealing).
    :param int num_samples_mc_prediction: Number of Monte Carlo samples to use on prediction.
    :param bool verbose: Flag to control whether accuracy monitoring is logged during repurposing.
    :param float annealing_weight: Annealing weight in the current epoch.
    :param train_acc: Accuracy in training set in each epoch.
    :type train_acc: list[float]
    :param test_acc: Accuracy in validation set in each epoch.
    :type test_acc: list[float]
    :param moving_loss_total: Total loss (negative ELBO) smoothed across epochs.
    :type moving_loss_total: list[float]
    :param average_loss: Average loss (negative ELBO) per data point.
    :type average_loss: list[float]
    :param anneal_weights: Annealing weight used in each epoch.
    :type anneal_weights: list[float]
    """

    def __init__(self, source_model: mx.mod.Module, feature_layer_names, context_function=mx.cpu, num_devices=1,
                 bnn_context_function=mx.cpu, sigma=100.0, num_layers=1, n_hidden=10, num_samples_mc=3,
                 learning_rate=1e-3, batch_size=20, num_epochs=200, start_annealing=None, end_annealing=None,
                 num_samples_mc_prediction=100, verbose=0):

        # Call base class constructor with parameters required for meta-models
        super().__init__(source_model, feature_layer_names, context_function, num_devices)

        # Initialize BNN specific parameters
        self.sigma = sigma
        self.num_layers = num_layers
        self.n_hidden = n_hidden
        self.num_samples_mc = num_samples_mc
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.num_samples_mc_prediction = num_samples_mc_prediction
        self.verbose = verbose

        self.start_annealing = start_annealing
        self.end_annealing = end_annealing
        self.step_annealing_sample_weight = 1.0 / float(self.end_annealing - self.start_annealing)
        self.annealing_weight = 0.0

        # Initialize variables to track performance
        self.train_acc = []
        self.test_acc = []
        self.moving_loss_total = []
        self.current_loss_total = []
        self.average_loss = []
        self.anneal_weights = []

        # L2 normalization of the features
        self.normalizer = Normalizer(norm='l2')

        self.bnn_context_function = bnn_context_function
        self._context_bnn = self.bnn_context_function()

        # init parameters for constructing network to None. These will be set during repurposing
        self.dim_input = None
        self.num_classes = None

    def get_params(self):
        """
        Get parameters of repurposer that are in the constructor.

        :rtype: dict
        """
        param_dict = super().get_params()
        param_dict[keys.BNN_CONTEXT_FUNCTION] = utils.serialize_ctx_fn(self.bnn_context_function)
        param_dict[keys.SIGMA] = self.sigma
        param_dict[keys.NUM_LAYERS] = self.num_layers
        param_dict[keys.N_HIDDEN] = self.n_hidden
        param_dict[keys.NUM_SAMPLES_MC] = self.num_samples_mc
        param_dict[keys.LEARNING_RATE] = self.learning_rate
        param_dict[keys.BATCH_SIZE] = self.batch_size
        param_dict[keys.NUM_EPOCHS] = self.num_epochs
        param_dict[keys.START_ANNEALING] = self.start_annealing
        param_dict[keys.END_ANNEALING] = self.end_annealing
        param_dict[keys.VERBOSE] = self.verbose
        param_dict[keys.NUM_SAMPLES_MC_PREDICT] = self.num_samples_mc_prediction
        return param_dict

    def _get_attributes(self):
        """
        Get parameters of repurposer not in constructor.

        :rtype: dict
        """
        attr_dict = super()._get_attributes()
        attr_dict[keys.ANNEALING_WEIGHT] = self.annealing_weight
        attr_dict[keys.DIM_INPUT] = self.dim_input
        attr_dict[keys.NUM_CLASSES] = self.num_classes
        return attr_dict

    def _set_attributes(self, input_dict):
        """
        Set attributes of class from input_dict.
        These attributes are the same as those returned by get_attributes method.

        :param input_dict: Dictionary containing attribute values.
        :return: None
        """
        super()._set_attributes(input_dict)
        self.annealing_weight = input_dict[keys.ANNEALING_WEIGHT]
        self.dim_input = input_dict[keys.DIM_INPUT]
        self.num_classes = input_dict[keys.NUM_CLASSES]

    @property
    def start_annealing(self):
        """
        Determines the epoch at which the annealing process of the KL term starts.
        """
        return self._start_annealing

    @start_annealing.setter
    def start_annealing(self, value):
        if value is None:
            self._start_annealing = int(0.1*self.num_epochs)
        else:
            if value >= self.num_epochs:
                error = ("start_annealing must be smaller than the number of epochs."
                         " Instead got start_annealing = {}, number of epochs = {}.".format(value, self.num_epochs))
                raise ValueError(error)
            self._start_annealing = int(value)

    @property
    def end_annealing(self):
        """
        Determines the epoch at which the annealing process of the KL term ends.
        """
        return self._end_annealing

    @end_annealing.setter
    def end_annealing(self, value):
        if value is None:
            self._end_annealing = self.start_annealing + int(max(0.1 * self.num_epochs, 1))
        else:
            if value > self.num_epochs:
                error = ("end_annealing must be smaller than the number of epochs."
                         " Instead got end_annealing = {}, number of epochs = {}.".format(value, self.num_epochs))
                raise ValueError(error)

            if value <= self.start_annealing:
                error = ("end_annealing must be greater than start_annealing."
                         " Instead got start_annealing = {}, end_annealing = {}.".format(self.start_annealing, value))
                raise ValueError(error)

            self._end_annealing = int(value)

    def _build_nn(self, train_data, num_classes):
        _, (data, _) = next(enumerate(train_data))
        dim_input = data.shape[1]
        net = self._get_net(num_classes)
        shapes = self._get_shapes(net, data, dim_input)

        # save parameters used for constructing network
        self.dim_input = dim_input
        self.num_classes = num_classes

        return net, shapes

    def _get_net(self, num_classes):
        net = gluon.nn.Sequential()
        with net.name_scope():
            for i in range(self.num_layers):
                net.add(gluon.nn.Dense(self.n_hidden, activation="relu"))
            net.add(gluon.nn.Dense(num_classes))
        # Init parameters
        net.collect_params().initialize(mx.init.Xavier(magnitude=2.24), ctx=self._context_bnn)
        return net

    def _get_shapes(self, net, data, dim_input):
        # first pass to init size params
        data = data.as_in_context(self._context_bnn).reshape((-1, dim_input))
        net(data)
        shapes = [x.shape for x in net.collect_params().values()]
        return shapes

    def _update_annealing_weight(self, epoch):
        if (epoch > self.start_annealing) and (epoch <= self.end_annealing):
            self.annealing_weight += self.step_annealing_sample_weight

    def _compute_loss_grad(self, data, label_one_hot, num_train, net, bnn_loss):
        with autograd.record():
            # calculate the loss
            # Note: It triggers the hybrid_forward method of the bnn_loss object
            loss = bnn_loss(data, net, label_one_hot, self.num_samples_mc,
                            num_train, self.annealing_weight)
            # back-propagate for gradient calculation
            loss.backward()
        return loss

    def _evaluate_accuracy(self, data_iterator, net, layer_params):
        numerator = 0.
        denominator = 0.
        for i, (data, label) in enumerate(data_iterator):
            data = data.as_in_context(self._context_bnn).reshape((-1, data.shape[1]))
            label = label.as_in_context(self._context_bnn)
            replace_params_net(layer_params, net, self._context_bnn)
            output = net(data)
            predictions = nd.argmax(output, axis=1)
            numerator += nd.sum(predictions == label)
            denominator += data.shape[0]
        return (numerator / denominator).asscalar()

    def _monitor_acc(self, epoch, train_data, net, posterior_mean, test_data=None):
        """
        Monitor accuracy on training set and test set (if provided).
        """
        train_accuracy = self._evaluate_accuracy(train_data, net, posterior_mean)
        self.train_acc.append(np.asscalar(train_accuracy))
        if self.verbose:
            logging.info("Epoch {}. Train Loss: {}, Train_acc {}" .format(epoch, self.average_loss[-1],
                                                                          self.train_acc[-1]))

        if test_data is not None:
            test_accuracy = self._evaluate_accuracy(test_data, net, posterior_mean)
            self.test_acc.append(np.asscalar(test_accuracy))
            if self.verbose:
                logging.info("Epoch {}. Train Loss: {}, Train_acc {}, Test_acc {}".format(epoch, self.average_loss[-1],
                                                                                          self.train_acc[-1],
                                                                                          self.test_acc[-1]))

    def _train_model_from_features(self, features, labels, feature_indices_per_layer=None):
        """
        Train a BNN model using features extracted from source neural network.

        :param features: Features extracted from source neural network.
        :type features: :class:`numpy.ndarray`
        :param labels: Labels to use for training.
        :type labels: :class:`numpy.ndarray`
        :param feature_indices_per_layer: Mapping of feature_layer_names to indices in features array
        i.e. {layer_name, feature_indices} Note that this param is currently not consumed by bnn_repurposer.
        :type feature_indices_per_layer: OrderedDict[str, :class:`numpy.ndarray`]
        :return: BNN model trained with given features and labels.
        :rtype: :class: `xfer.bnn_classifier.BnnClassifier`
        """

        # Load data
        x_tr = features.astype(np.dtype(np.float32))
        x_tr = self.normalizer.fit_transform(x_tr)
        y_tr = labels.astype(np.dtype(np.float32))

        num_classes = len(set(y_tr))

        train_data = gluon.data.DataLoader(gluon.data.ArrayDataset(x_tr, y_tr), batch_size=self.batch_size,
                                           shuffle=True)
        num_train = sum([self.batch_size for _ in train_data])

        # Build net and initialize parameters
        net, shapes = self._build_nn(train_data, num_classes)

        # Create Prior, variational posterior and observation model
        prior = GaussianPrior([np.array([[0.0]])], [np.array([[self.sigma]])], shapes, self._context_bnn)
        var_posterior = GaussianVariationalPosterior(mean_init=consts.MEAN_INIT_POSTERIOR,
                                                     sigma_init=consts.SIGMA_INIT_POSTERIOR, shapes=shapes,
                                                     ctx=self._context_bnn)
        obs_model = Categorical(self._context_bnn)

        # Build loss (negative of the ELBO)
        bnn_loss = BNNLoss(prior=prior, obs_model=obs_model, var_posterior=var_posterior)

        # Training

        trainer = gluon.Trainer(params=var_posterior.get_params_list() + prior.get_params_list(),
                                optimizer=consts.BNN_OPTIMIZER, optimizer_params={'learning_rate': self.learning_rate})
        smoothing_constant = .01
        moving_loss = 0.0
        for e in range(self.num_epochs):
            self._update_annealing_weight(e)

            curr_loss_total = 0.0
            for i, (data, label) in enumerate(train_data):
                data = data.as_in_context(self._context_bnn).reshape((-1, data.shape[1]))
                label = label.as_in_context(self._context_bnn)
                label_one_hot = nd.one_hot(label, num_classes)

                loss = self._compute_loss_grad(data, label_one_hot, num_train, net, bnn_loss)

                trainer.step(data.shape[0])

                # calculate moving loss for monitoring convergence
                curr_loss = nd.mean(loss).asscalar()
                curr_loss_total = curr_loss_total + curr_loss
                moving_loss = (curr_loss if ((i == 0) and (e == 0))
                               else (1 - smoothing_constant) * moving_loss + smoothing_constant * curr_loss)

            self.anneal_weights.append(self.annealing_weight)
            self.moving_loss_total.append(moving_loss)
            self.current_loss_total.append(curr_loss)
            self.average_loss.append(curr_loss_total / float(num_train))

            # Monitor accuracy on training set and test set (if provided)
            #: as self.verbose in the constructor
            self._monitor_acc(e, train_data, net, var_posterior.get_mean())

        return BnnClassifier(net, var_posterior, self.normalizer)

    def _predict_probability_from_features(self, features):
        """
        Run predictions with target_model on features extracted from source neural network.
        Use xfer.bnn_classifier.BnnClassifier's predict method and return predicted probabilities.

        :param features: Features extracted from source neural network.
        :type features: :class:`numpy.ndarray`
        :return: Predicted probabilities.
        :rtype: :class:`numpy.ndarray`
        """
        return self.target_model.predict(features, self.num_samples_mc_prediction, context=self._context_bnn)[1]

    def _predict_label_from_features(self, features):
        """
        Run predictions with target_model on features extracted from source neural network.
        Use xfer.bnn_classifier.BnnClassifier's predict method and return predicted labels.

        :param features: Features extracted from source neural network.
        :type features: :class:`numpy.ndarray`
        :return: Predicted labels.
        :rtype: :class:`numpy.ndarray`
        """
        return self.target_model.predict(features, self.num_samples_mc_prediction, context=self._context_bnn)[0]

    def serialize(self, file_prefix):
        """
        Saves repurposer (excluding source model) to file_prefix.json, file_prefix_posterior.json,
        file_prefix_posterior_params.npz.

        :param str file_prefix: Prefix to save file with.
        """
        output_dict = {}
        output_dict[repurposer_keys.PARAMS] = self.get_params()
        output_dict.update(self._get_attributes())

        utils.save_json(file_prefix, output_dict)

        self.target_model.var_posterior.save(file_prefix + serialization_constants.POSTERIOR_SUFFIX)

    def deserialize(self, input_dict):
        """
        Uses dictionary to set attributes of repurposer.

        :param dict input_dict: Dictionary containing values for attributes to be set to.
        """
        # Set attributes of the repurposer from input_dict
        self._set_attributes(input_dict)

        # Deserialize the target bnn model
        var_posterior = GaussianVariationalPosterior.load(input_dict[serialization_keys.FILE_PATH] +
                                                          serialization_constants.POSTERIOR_SUFFIX)
        net = self._get_net(self.num_classes)

        # Make a forward pass to initialize parameter shapes for the network
        net(mx.ndarray.zeros(shape=(self.batch_size, self.dim_input)))

        self.target_model = BnnClassifier(net, var_posterior, self.normalizer)
