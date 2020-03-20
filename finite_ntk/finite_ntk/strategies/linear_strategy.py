# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
#!/usr/bin/env python3

import torch

from gpytorch import settings
from gpytorch.models.exact_prediction_strategies import DefaultPredictionStrategy
from gpytorch.utils.memoize import cached
from gpytorch.lazy import ZeroLazyTensor, RootLazyTensor
from gpytorch.utils.lanczos import lanczos_tridiag, lanczos_tridiag_to_diag


class LinearPredictionStrategy(DefaultPredictionStrategy):
    def __init__(
        self,
        train_inputs,
        train_prior_dist,
        train_labels,
        likelihood,
        root=None,
        inv_root=None,
        epsilon=1e-3,
        preconditioner=None,
    ):
        r"""
        LinearPredictionStrategy is a variant of gpytorch's default prediction strategy
        that does all inference and caching in parameter space rather than in function 
        space. It will only work for kernels which have a finite number of basis functions
        (e.g. basis function models such as the finite neural tangent kernel). 
        train_inputs: input training data
        train_prior_dist: prior distribution created by model (e.g. by calling model(train_inputs))
        train_labels: training response
        likelihood: likelihood for the data (e.g. a GaussianLikelihood)
        root: root decomposition of prior distribution (used internally)
        inv_root: inverse root decomposition of training prior distribution (used internally)
        epsilon: hyper-parameter for fast Fisher vector products (TODO: see if can be replaced by kwargs)
        preconditioner: unused (TODO: see if should be deprecated)
        """
        # initialize in the standard manner
        super(LinearPredictionStrategy, self).__init__(
            train_inputs, train_prior_dist, train_labels, likelihood, root, inv_root
        )

        self.preconditioner = preconditioner

        # store mean since that's not stored for some reason?
        mvn = self.likelihood(train_prior_dist, train_inputs)
        self.lik_train_mean = mvn.mean

        # now pre-compute the solves and eigendecompositions
        self.kernel_class = self.lik_train_train_covar.lazy_tensors[0].evaluate_kernel()
        # lazy matmul that is F v
        fvp = self.kernel_class.get_expansion(epsilon=epsilon)

        # also add some jitter: eq to prior inverse variance
        # store F + \sigma^2 I
        self.noise = self.lik_train_train_covar.lazy_tensors[1]._diag[0].item()
        self.lik_train_train_expansion = fvp.add_jitter(self.noise)

    def _exact_predictive_covar_inv_quad_form_root(self, expanded_lt, test_train_covar):
        """
        Computes :math:`K_{X^{*}X} S` given a precomputed cache
        Where :math:`S` is a tensor such that :math:`SS^{\\top} = (K_{XX} + \sigma^2 I)^{-1}`
        Args:
            precomputed_cache (:obj:`torch.tensor`): What was computed in _exact_predictive_covar_inv_quad_form_cache
            test_train_covar (:obj:`torch.tensor`): The observed noise (from the likelihood)
        Returns
            :obj:`~gpytorch.lazy.LazyTensor`: :math:`K_{X^{*}X} S`
        """
        qmats, tmats = lanczos_tridiag(
            expanded_lt.matmul,
            max_iter=settings.max_root_decomposition_size.value(),
            matrix_shape=expanded_lt.shape,
            device=expanded_lt.device,
            dtype=expanded_lt.dtype,
        )
        evals, evecs = lanczos_tridiag_to_diag(tmats)

        self.gram_evecs = qmats @ evecs
        self.gram_evals = evals

        covar_root = self.gram_evecs @ torch.diag(evals.pow(-0.5))

        return covar_root

    @property
    @cached(name="covar_cache")
    def covar_cache(self):
        # #TODO: THE CACHE SHOULD RETURN ONLY S OR S M
        return self._exact_predictive_covar_inv_quad_form_root(
            self.lik_train_train_expansion, self._last_test_train_covar
        )

    @property
    @cached(name="mean_cache")
    def mean_cache(self):
        # TODO: THE CACHE SHOULD BE THE ENTIRE SOLVE - F^+ J^T Y
        train_labels_offset = (self.train_labels - self.lik_train_mean).unsqueeze(-1)

        features_trainx = self.kernel_class.get_root()
        response = features_trainx.matmul(train_labels_offset)

        # with settings.cg_tolerance(1e-4):
        mean_cache = self.lik_train_train_expansion.inv_matmul(response)
        if settings.detach_test_caches.on():
            return mean_cache.detach()
        else:
            return mean_cache

    def exact_predictive_mean(self, test_mean, test_train_covar):
        """
        Computes the posterior predictive covariance of a GP
        Args:
            test_mean (:obj:`torch.tensor`): The test prior mean
            test_train_covar (:obj:`gpytorch.lazy.LazyTensor`): Covariance matrix between test and train inputs
        Returns:
            :obj:`torch.tensor`: The predictive posterior mean of the test points
        """
        features_xstar = test_train_covar.evaluate_kernel().get_root(dim=-2)

        res = features_xstar.t().matmul(self.mean_cache)
        res = res.view_as(test_mean) + test_mean
        return res

    def exact_predictive_covar(self, test_test_covar, test_train_covar):
        """
        Computes the posterior predictive covariance of a GP
        Args:
            test_train_covar (:obj:`gpytorch.lazy.LazyTensor`): Covariance matrix between test and train inputs
            test_test_covar (:obj:`gpytorch.lazy.LazyTensor`): Covariance matrix between test inputs
        Returns:
            :obj:`gpytorch.lazy.LazyTensor`: A LazyTensor representing the predictive posterior covariance of the
                                               test points
        """
        if settings.fast_pred_var.on():
            self._last_test_train_covar = test_train_covar

        if settings.skip_posterior_variances.on():
            return ZeroLazyTensor(*test_test_covar.size())

        if settings.fast_pred_var.off():
            super().exact_predictive_covar(test_test_covar, test_train_covar)
        else:
            features_xstar = test_train_covar.evaluate_kernel().get_root(dim=-2)

            # compute J^T Cache as our root tensor
            j_star_covar = features_xstar.t() @ self.covar_cache

            covar_expanded = RootLazyTensor(j_star_covar)
            return self.noise * covar_expanded
