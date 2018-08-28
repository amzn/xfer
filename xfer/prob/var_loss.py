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
import mxnet as mx
from mxnet import gluon

from .utils import replace_params_net


class BNNLoss(gluon.loss.Loss):
    """
    Variational loss to minimize (negative ELBO)

    :param prior: Prior distribution defined on the weights of the neural network
    :type prior: :class:`xfer.prob.Prior`
    :param obs_model: Likelihood of the observations
    :type obs_model: :class:`xfer.prob.Likelihood`
    :param var_posterior: Variational posterior distribution defined on the weights of the neural network
    :type var_posterior: :class:`xfer.prob.VariationalPosterior`
    :param float weight: Global scalar weight for loss
    :param int batch_axis: The axis that represents mini-batch
    :param ctx: MXNet context
    :type ctx: :class:`mxnet.Context`
    """

    def __init__(self, prior, obs_model, var_posterior, weight=None, batch_axis=0, ctx=mx.cpu()):

        super(BNNLoss, self).__init__(weight, batch_axis)
        self.obs_model = obs_model
        self.var_posterior = var_posterior
        self.prior = prior
        self.kldiv_exact = self.var_posterior.is_conjugate(self.prior)
        self.context = ctx

    def hybrid_forward(self, F, data, net, label, num_samples, total_number_data, anneal_weight=1.0):
        # This function overrides the hybrid_forward method in gluon.loss.Loss.
        # gluon.loss.Loss defines hybrid_forward  as a function call operator using __call__

        loss = 0.0
        kl_weight = anneal_weight * (data.shape[0] / float(total_number_data))
        for _ in range(num_samples):
            # generate sample
            layer_params = self.var_posterior.generate_sample()

            replace_params_net(layer_params, net, self.context)

            # forward-propagate the batch
            self.obs_model.set_unnormalized_mean(net(data))

            # loss for the current sample
            log_likelihood_sum = self.obs_model.log_pdf(label)
            loss = loss - log_likelihood_sum

            if not self.kldiv_exact:
                loss = loss + kl_weight * self.var_posterior.KL_sample(self.prior, layer_params)

        loss = loss / float(num_samples)

        # add exact kl
        if self.kldiv_exact:
            loss = loss + kl_weight * self.var_posterior.KL(self.prior)
        return loss
