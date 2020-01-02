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
from mxnet import gluon
import mxnet as mx
from sklearn.preprocessing import Normalizer

from .prob import VariationalPosterior
from .prob.utils import replace_params_net


class BnnClassifier(object):
    """
    Repurpose neural network to create BNN meta-model through Transfer Learning

    :param model: neural network model
    :type model: :class:`mxnet.gluon.nn.Sequential`
    :param var_posterior: Variational posterior distribution
    :type var_posterior: :class:`xfer.prob.VariationalPosterior`
    :param normalizer: It is applied to the input features before computing the predictions
    :type normalizer: :class:`sklearn.preprocessing.Normalizer`
    """

    def __init__(self, model: gluon.nn.Sequential, var_posterior: VariationalPosterior,
                 normalizer: Normalizer):

        self.model = model
        self.var_posterior = var_posterior
        self.normalizer = normalizer

    def predict(self, features, num_samples_mc_prediction=100, context=mx.cpu()):
        """
        Function to make predictions

        :param features: Features extracted from source neural network
        :type features: :class:`numpy.ndarray`
        :param int num_samples_mc_prediction: Num of samples from the posterior to compute the predictions
        :param context: MXNet context
        :type context: :class:`mxnet.Context`
        :return: Tuple containing predicted labels and probabilities
        :rtype: tuple(:class:`numpy.ndarray`, :class:`numpy.ndarray`)
        """
        if self.var_posterior.get_mean() is None:
            raise RuntimeError("Repurposer is not trained!")

        n_te, dim = features.shape
        features = features.astype(np.dtype(np.float32))
        features = self.normalizer.transform(features)
        data = mx.nd.array(features)
        data = data.as_in_context(context).reshape((-1, dim))

        for j in range(num_samples_mc_prediction):
            layer_params = self.var_posterior.generate_sample()

            replace_params_net(layer_params, self.model, context)

            output_temp = self.model(data)
            output = mx.nd.softmax(output_temp).asnumpy()

            if j == 0:
                predictions = np.zeros((n_te, output.shape[1]))

            for i in range(n_te):
                p = output[i, :].astype(np.dtype(np.float64))
                p /= p.sum()
                predictions[i, :] += np.random.multinomial(1, p)

        return predictions.argmax(axis=1), predictions/float(num_samples_mc_prediction)
