# Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import torch

from finite_ntk.lazy import flatten
import numpy as np


#### these utils from
# https://github.com/wjmaddox/drbayes/blob/master/subspace_inference/posteriors/utils.py

def extract_parameters(model):
    params = []	
    for module in model.modules():	
        for name in list(module._parameters.keys()):	
            if module._parameters[name] is None:	
                continue	
            param = module._parameters[name]	
            params.append((module, name, param.size()))	
            module._parameters.pop(name)	
    return params

def set_weights(params, w, device):	
    offset = 0
    for module, name, shape in params:
        size = np.prod(shape)	       
        value = w[offset:offset + size]
        setattr(module, name, value.view(shape).to(device))	
        offset += size

class LinearizedVINet(torch.nn.Module):
    def __init__(self, net, prior_std=1., linearize_idx=(3,), eps=1e-6):
        super(LinearizedVINet, self).__init__()
        self.base_model = net
        print(linearize_idx)

        self.eps = 1e-6

        # this is NOT a general purpose vi method!!!
        # specifically, it's an adaptation of SVI for the specific setting of the linearized networks
        # defined in Mu et al, ICLR 2020
        numparams = 0
        self.base_params = {}
        self.linear_means = {}
        self.linear_inv_softplus_sigma = {}
        if 1 in linearize_idx: 
            numparams += sum([p.numel() for p in self.base_model.hnet.linear1.parameters()])
            flattened_pars = flatten(self.base_model.hnet.linear1.parameters())
            self.base_params['linear1'] = extract_parameters(self.base_model.hnet.linear1)

            self.linear_means_linear1 = torch.nn.Parameter(
                torch.zeros(flattened_pars.shape[0], device=flattened_pars.device)
            )
            self.linear_inv_softplus_sigma_linear1 = torch.nn.Parameter(
                -5. * torch.ones(flattened_pars.shape[0], device=flattened_pars.device)
            )
            self.linear_means_linear1.requires_grad = True
            self.linear_inv_softplus_sigma_linear1.requires_grad = True
            
        if 2 in linearize_idx: 
            numparams += sum([p.numel() for p in self.base_model.hnet.linear2.parameters()])
            flattened_pars = flatten(self.base_model.hnet.linear2.parameters())
            self.base_params['linear2'] = extract_parameters(self.base_model.hnet.linear2)


            self.linear_means_linear2 = torch.nn.Parameter(
                torch.zeros(flattened_pars.shape[0], device=flattened_pars.device)
            )
            self.linear_inv_softplus_sigma_linear2 = torch.nn.Parameter(
                -5. * torch.ones(flattened_pars.shape[0], device=flattened_pars.device)
            )
            self.linear_means_linear2.requires_grad = True
            self.linear_inv_softplus_sigma_linear2.requires_grad = True

        if 3 in linearize_idx: 
            numparams += sum([p.numel() for p in self.base_model.hnet.linear3.parameters()])
            flattened_pars = flatten(self.base_model.hnet.linear3.parameters())
            self.base_params['linear3'] = extract_parameters(self.base_model.hnet.linear3)


            self.linear_means_linear3 = torch.nn.Parameter(
                torch.zeros(flattened_pars.shape[0], device=flattened_pars.device)
            )
            self.linear_inv_softplus_sigma_linear3 = torch.nn.Parameter(
                -5. * torch.ones(flattened_pars.shape[0], device=flattened_pars.device)
            )
            self.linear_means_linear3.requires_grad = True
            self.linear_inv_softplus_sigma_linear3.requires_grad = True

        self.linearize_idx = linearize_idx
        self.prior_std = prior_std

    def sample_and_set_weights(self, param, mean, inv_sigma):
        sigma = torch.nn.functional.softplus(inv_sigma) + self.eps
        
        if self.train:
            z = mean + torch.randn_like(mean) * sigma
        else:
            z = mean

        device = sigma.device
        set_weights(param, z, device)

    def _kldiv(self, mean, inv_sigma):
        prior_dist = torch.distributions.Normal(torch.zeros_like(mean), torch.ones_like(mean) * self.prior_std)
        q_dist = torch.distributions.Normal(mean, torch.nn.functional.softplus(inv_sigma) + self.eps)
        return torch.distributions.kl.kl_divergence(q_dist, prior_dist)

    def kl_divergence(self):
        kldiv = 0.
        if 1 in self.linearize_idx: 
            kldiv = kldiv + self._kldiv(self.linear_means_linear1, self.linear_inv_softplus_sigma_linear1).sum()
        if 2 in self.linearize_idx: 
            kldiv = kldiv + self._kldiv(self.linear_means_linear2, self.linear_inv_softplus_sigma_linear2).sum()
        if 3 in self.linearize_idx: 
            kldiv = kldiv + self._kldiv(self.linear_means_linear3, self.linear_inv_softplus_sigma_linear3).sum()
        return kldiv

        
    def forward(self, *args, **kwargs):
        if 1 in self.linearize_idx: 
            self.sample_and_set_weights(self.base_params['linear1'],
                    self.linear_means_linear1, self.linear_inv_softplus_sigma_linear1)
        if 2 in self.linearize_idx: 
            self.sample_and_set_weights(self.base_params['linear2'],
                    self.linear_means_linear2, self.linear_inv_softplus_sigma_linear2)
        if 3 in self.linearize_idx: 
            self.sample_and_set_weights(self.base_params['linear3'],
                    self.linear_means_linear3, self.linear_inv_softplus_sigma_linear3)

        return self.base_model(*args, **kwargs)
