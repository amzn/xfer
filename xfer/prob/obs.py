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
import mxnet.ndarray as nd

from .prob_base import Probability


class Likelihood(Probability):
    def __init__(self, ctx):
        super(Likelihood, self).__init__(None, ctx)


class Categorical(Likelihood):
    def __init__(self, ctx):
        super(Categorical, self).__init__(ctx)

    def set_unnormalized_mean(self, unnormalized_mean):
        self.unnormalized_mean = unnormalized_mean

    def log_pdf(self, y):
        return nd.sum(nd.nansum(y * nd.log_softmax(self.unnormalized_mean), axis=0, exclude=True))
