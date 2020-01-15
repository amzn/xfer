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
from unittest import TestCase

import random
import os
import numpy as np

import mxnet as mx
from mxnet import nd

from xfer import prob as mbprob


class TestProb(TestCase):
    def setUp(self):
        np.random.seed(1)
        random.seed(1)
        mx.random.seed(1)

        self.context = mx.cpu(0)
        self.obs_variables = [nd.array([[1, 0.5, 1.5], [2, 2.5, 2]], ctx=self.context),
                              nd.array([[3, -0.5, 1]], ctx=self.context)]
        self.shapes = [x.shape for x in self.obs_variables]

        self.prior1 = mbprob.GaussianPrior([np.array([[0.0]])], [np.array([[1.0]])],
                                           self.shapes, self.context)
        self.prior2 = mbprob.GaussianPrior([np.array([[0.0]]), np.array([[1.0]])],
                                           [np.array([[1.0]]), np.array([[2.0]])], self.shapes, self.context)
        self.prior3 = mbprob.GaussianPrior([np.array([[0.0, 0.5, 1.0], [-0.5, 0.5, 1.0]]), np.array([[1.0]])],
                                           [np.array([[1.0]]), np.array([[2.0, 3.0, 4.0]])],
                                           self.shapes, self.context)

        self.posterior1 = mbprob.GaussianVariationalPosterior([np.array([[0.0]])], [np.array([[1.0]])],
                                                              self.shapes, self.context)
        self.posterior2 = mbprob.GaussianVariationalPosterior([np.array([[0.0]]), np.array([[1.0]])],
                                                              [np.array([[1.0]]), np.array([[2.0]])], self.shapes,
                                                              self.context)
        self.posterior3 = mbprob.GaussianVariationalPosterior(
            [np.array([[0.0, 0.5, 1.0], [-0.5, 0.5, 1.0]]), np.array([[1.0]])],
            [np.array([[1.0]]), np.array([[2.0, 3.0, 4.0]])], self.shapes, self.context)

    def test_prior_log_pdf(self):
        # GaussianPrior
        assert self.prior1.log_pdf(self.obs_variables) == -22.270447
        assert self.prior2.log_pdf(self.obs_variables) == -20.006138
        assert self.prior3.log_pdf(self.obs_variables) == -18.323502

    def test_posterior_log_pdf(self):
        # GaussianVariationalPosterior
        assert self.posterior1.log_pdf(self.obs_variables) == -22.270447
        assert self.posterior2.log_pdf(self.obs_variables) == -20.006138
        assert self.posterior3.log_pdf(self.obs_variables) == -18.323502

    def test_posterior_get_mean(self):
        # GaussianVariationalPosterior
        mean_ref = [np.array([[0.0, 0.5, 1.0], [-0.5, 0.5, 1.0]]), np.array([[1.0]])]
        mean = [x.asnumpy() for x in self.posterior3.get_mean()]
        for m, m_ref in zip(mean, mean_ref):
            assert np.allclose(m, m_ref)

    def test_posterior_is_conjugate(self):
        # GaussianVariationalPosterior
        assert self.posterior3.is_conjugate(self.prior2)

    def test_posterior_KL(self):
        # GaussianVariationalPosterior
        assert self.posterior1.KL(self.prior1) == 0
        assert self.posterior2.KL(self.prior2) == 0
        assert self.posterior3.KL(self.prior3) == 0
        assert (self.posterior3.KL(self.prior2) - 2.4013877) < 1e-5
        assert (self.posterior1.KL(self.prior2) - 1.3294418) < 1e-5

    def test_likelihood_logpdf(self):
        # Categorical
        self.likelihood1 = mbprob.Categorical(self.context)
        self.likelihood1.set_unnormalized_mean(nd.array([[1, 0.5, 1.5], [2, 2.5, 2]], ctx=self.context))
        y = nd.array([[1, 0.0, 0.0], [0.0, 1.0, 0.0]], ctx=self.context)
        assert self.likelihood1.log_pdf(y).asnumpy() == -1.9746466

    def test_prob_base_register_param(self):
        init_means1 = mx.init.Constant(3.0)
        probability1 = mbprob.Probability(self.shapes, self.context)
        probability1._register_param("mean1", init_means1, fix=True)

        assert np.allclose(probability1.params["mean1"][0].data().asnumpy(), 3.0 * np.ones((2, 3)))
        assert np.allclose(probability1.raw_params["mean1"][0].asnumpy(), 3.0 * np.ones((2, 3)))
        assert np.allclose(probability1.params["mean1"][1].data().asnumpy(), np.array([[3.0]]))
        assert np.allclose(probability1.raw_params["mean1"][1].asnumpy(), np.array([[3.0]]))
        assert probability1.params["mean1"][0].grad_req == "null"
        assert probability1.params["mean1"][1].grad_req == "null"

        probability1._register_param("mean2", init_means1, fix=[False, True])
        assert np.allclose(probability1.params["mean2"][0].data().asnumpy(), 3.0 * np.ones((2, 3)))
        assert np.allclose(probability1.raw_params["mean2"][0].asnumpy(), 3.0 * np.ones((2, 3)))
        assert np.allclose(probability1.params["mean2"][1].data().asnumpy(), np.array([[3.0]]))
        assert np.allclose(probability1.raw_params["mean2"][1].asnumpy(), np.array([[3.0]]))
        assert probability1.params["mean2"][0].grad_req == "write"
        assert probability1.params["mean2"][1].grad_req == "null"

    def test_prob_base_register_param_value(self):
        means1 = [np.array([[0.03629481, -0.4902442, -0.95017916],
                            [0.03751944, -0.72984636, -2.0401056]]), np.array([[1.0]])]
        probability1 = mbprob.Probability(self.shapes, self.context)
        probability1._register_param_value("mean1", means1, fix=True)

        assert np.allclose(probability1.params["mean1"][0].data().asnumpy(), means1[0])
        assert np.allclose(probability1.raw_params["mean1"][0].asnumpy(), means1[0])
        assert np.allclose(probability1.params["mean1"][1].data().asnumpy(), means1[1])
        assert np.allclose(probability1.raw_params["mean1"][1].asnumpy(), means1[1])
        assert probability1.params["mean1"][0].grad_req == "null"
        assert probability1.params["mean1"][1].grad_req == "null"

        means2 = [np.array([[0.03629481, -0.4902442, -0.95017916],
                            [0.03751944, -0.72984636, -2.0401056]]), np.array([[2.0]])]
        probability1._register_param_value("mean2", means2, fix=[False, True])
        assert np.allclose(probability1.params["mean2"][0].data().asnumpy(), means2[0])
        assert np.allclose(probability1.raw_params["mean2"][0].asnumpy(), means2[0])
        assert np.allclose(probability1.params["mean2"][1].data().asnumpy(), means2[1])
        assert np.allclose(probability1.raw_params["mean2"][1].asnumpy(), means2[1])
        assert probability1.params["mean2"][0].grad_req == "write"
        assert probability1.params["mean2"][1].grad_req == "null"

        means3 = [np.array([[0.03629481, -0.4902442, -0.95017916], [0.03751944, -0.72984636, -2.0401056]]),
                  np.array([[1.0, 2.0]])]
        with self.assertRaises(ValueError):
            probability1._register_param_value("means3", means3, fix=True)
        means4 = [np.array([[0.03629481, -0.4902442, -0.95017916], [0.03751944, -0.72984636, -2.0401056]]),
                  np.array([[1.0]]), np.array([[1.0]])]
        with self.assertRaises(ValueError):
            probability1._register_param_value("means4", means4, fix=True)

    def test_prob_base_replicate_shared_parameters(self):
        means1 = [np.array([[0.03629481, -0.4902442, -0.95017916], [0.03751944, -0.72984636, -2.0401056]]),
                  np.array([[1.0]])]
        probability1 = mbprob.Probability(self.shapes, self.context)
        probability1._register_param_value("mean1", means1, fix=True)
        raw_params_ext = probability1._replicate_shared_parameters()
        assert np.allclose(raw_params_ext["mean1"][0].asnumpy(), np.array([[0.03629481, -0.4902442, -0.95017916],
                                                                           [0.03751944, -0.72984636, -2.0401056]]))
        assert np.allclose(raw_params_ext["mean1"][1].asnumpy(), np.array([[1.0, 1.0, 1.0]]))

    def test_check_observation_shapes(self):
        obs1 = [np.zeros((2, 3)), np.ones((1, 3))]
        obs2 = [np.zeros((2, 3)), np.ones((1, 3)), np.ones((1, 3))]
        obs3 = [np.zeros((2, 3)), np.ones((1, 4))]
        probability1 = mbprob.Probability(self.shapes, self.context)
        probability1.check_observation_shapes(obs1)
        with self.assertRaises(ValueError):
            probability1.check_observation_shapes(obs2)
        with self.assertRaises(ValueError):
            probability1.check_observation_shapes(obs3)

    def test_get_params_list(self):
        probability1 = mbprob.Probability(self.shapes, self.context)
        means1 = [np.array([[0.03629481, -0.4902442, -0.95017916], [0.03751944, -0.72984636, -2.0401056]]),
                  np.array([[1.0]])]
        means2 = [np.array([[2.0]]), np.array([[3.0]])]
        probability1._register_param_value("param1", means1, fix=True)
        probability1._register_param_value("param2", means2, fix=True)
        list_params = probability1.get_params_list()
        assert np.allclose(list_params[0].data().asnumpy(), means1[0])
        assert np.allclose(list_params[1].data().asnumpy(), means1[1])
        assert np.allclose(list_params[2].data().asnumpy(), means2[0])
        assert np.allclose(list_params[3].data().asnumpy(), means2[1])

    def test_parse_grad_req(self):
        probability1 = mbprob.Probability(self.shapes, self.context)
        grad_req1 = probability1._parse_grad_req(False, 2)
        grad_req2 = probability1._parse_grad_req(True, 2)
        grad_req3 = probability1._parse_grad_req([False, True])
        assert grad_req1[0] == "write"
        assert grad_req1[1] == "write"
        assert grad_req2[0] == "null"
        assert grad_req2[1] == "null"
        assert grad_req3[0] == "write"
        assert grad_req3[1] == "null"

    def test_gaussian_var_post_save_load(self):
        file_path = 'gvp_save'
        assert not os.path.isfile(file_path + '.json')
        assert not os.path.isfile(file_path + '_params.npz')

        self.posterior1.save(file_path)

        assert os.path.isfile(file_path + '.json')
        assert os.path.isfile(file_path + '_params.npz')

        loaded = mbprob.GaussianVariationalPosterior.load(file_path)

        assert self.posterior1.shapes == loaded.shapes
        assert self.posterior1.ctx == loaded.ctx

        for key in self.posterior1.raw_params.keys():
            for count, _ in enumerate(self.posterior1.raw_params[key]):
                assert np.array_equal(self.posterior1.raw_params[key][count].asnumpy(),
                                      loaded.raw_params[key][count].asnumpy())

        for key in self.posterior1.params.keys():
            for count, value in enumerate(self.posterior1.params[key]):
                assert np.array_equal(value.data(), loaded.params[key][count].data())
                assert value.grad_req == loaded.params[key][count].grad_req
                assert value.name == loaded.params[key][count].name

        os.remove(file_path + '.json')
        os.remove(file_path + '_params.npz')
