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

## copied from https://github.com/fmu2/gradfeat20

import torch
import torch.nn as nn
from util import *

LEAK = 0.01   # for Jenson-Shannon BiGAN

class FeatureNet(nn.Module):
  """Network section parametrized by theta_1"""
  """standard parametrization, used by both baseline and proposed models."""
  def __init__(self):
    super().__init__()
    self.conv1 = nn.Conv2d(3, 32, 5, 1)
    self.conv2 = nn.Conv2d(32, 64, 4, 2)

  def freeze(self):
    for p in self.parameters():
      p.requires_grad = False

  def thaw(self):
    for p in self.parameters():
      p.requires_grad = True

  def forward(self, x):
    y1 = F.leaky_relu(self.conv1(x), LEAK, inplace=True)
    y2 = F.leaky_relu(self.conv2(y1), LEAK, inplace=True)
    return y2


class STDHeadNet(nn.Module):
  """Network section parametrized by theta_2"""
  """standard parametrization, used by baseline"""
  def __init__(self):
    super().__init__()
    self.conv1 = nn.Conv2d(64, 128, 4, 1)
    self.conv2 = nn.Conv2d(128, 256, 4, 2)
    self.conv3 = nn.Conv2d(256, 512, 4, 1)

  def reinit(self, idx):
    pass

  def freeze(self, idx):
    if 1 in idx:
      for p in self.conv1.parameters():
        p.requires_grad = False
  
    if 2 in idx:
      for p in self.conv2.parameters():
        p.requires_grad = False
  
    if 3 in idx:
      for p in self.conv3.parameters():
        p.requires_grad = False

  def thaw(self):
    for p in self.parameters():
      p.requires_grad = True

  def linearize(self, idx):
    pass

  def forward(self, x):
    y1 = F.leaky_relu(self.conv1(x), LEAK, inplace=True)
    y2 = F.leaky_relu(self.conv2(y1), LEAK, inplace=True)
    y3 = F.leaky_relu(self.conv3(y2), LEAK, inplace=True)
    #print(y3.shape)
    return y3, None


class NTKHeadNet(nn.Module):
  """Network section parametrized by theta_2"""
  """NTK parametrization, used by proposed model"""
  def __init__(self):
    super().__init__()
    self.conv1 = NTKConv2d(64, 128, 4, 1)
    self.conv2 = NTKConv2d(128, 256, 4, 2)
    self.conv3 = NTKConv2d(256, 512, 4, 1)
    self.linear1 = self.linear2 = self.linear3 = None

  def reinit(self, idx):
    if 1 in idx:    self.conv1.init()
    if 2 in idx:    self.conv2.init()
    if 3 in idx:    self.conv3.init()

  def freeze(self, idx=(1, 2)):
    if 1 in idx:    self.conv1.freeze()
    if 2 in idx:    self.conv2.freeze()
    if 3 in idx:    self.conv3.freeze()

  def thaw(self):
    for p in self.parameters():
      p.requires_grad = True

  def linearize(self, idx=(3,)):
    self.freeze(idx)
    if 1 in idx:    self.linear1 = NTKConv2d(64, 128, 4, 1, zero_init=True)
    if 2 in idx:    self.linear2 = NTKConv2d(128, 256, 4, 2, zero_init=True)
    if 3 in idx:    self.linear3 = NTKConv2d(256, 512, 4, 1, zero_init=True)

  def forward(self, x):
    y1 = F.leaky_relu(self.conv1(x), LEAK, inplace=True)
    y2 = F.leaky_relu(self.conv2(y1), LEAK, inplace=True)
    y3 = F.leaky_relu(self.conv3(y2), LEAK, inplace=True)
    
    jvp3 = None
    if self.linear3 is not None:
      jvp3 = self.linear3(y2)
      if self.linear2 is not None:
        jvp2 = self.linear2(y1)
        if self.linear1 is not None:
          jvp1 = self.linear1(x) * ((y1 > 0).float() + (y1 < 0).float() * LEAK)
          jvp2 = self.conv2(jvp1, add_bias=False) + jvp2
        jvp2 = jvp2 * ((y2 > 0).float() + (y2 < 0).float() * LEAK)
        jvp3 = self.conv3(jvp2, add_bias=False) + jvp3
      jvp3 = jvp3 * ((y3 > 0).float() + (y3 < 0).float() * LEAK)
    return y3, jvp3
      

class STDClassifier(nn.Module):
  """Logistic regressor parametrized by omega"""
  """standard parametrization, used by baseline"""
  def __init__(self, nclasses):
    super().__init__()
    self.nclasses = nclasses
    self.avgpool = nn.AdaptiveAvgPool2d(1)
    self.fc = nn.Linear(512, nclasses)

  def reinit(self):
    self.fc = nn.Linear(512, self.nclasses)

  def freeze(self):
    for p in self.parameters():
      p.requires_grad = False

  def thaw(self):
    for p in self.parameters():
      p.requires_grad = True

  def linearize(self):
    pass

  def forward(self, x, jvp=None):
    x = self.avgpool(x).flatten(1)
    logits = self.fc(x)
    return logits


class NTKClassifier(nn.Module):
  """Logistic regressor parametrized by omega"""
  """NTK parametrization, used by proposed model"""
  def __init__(self, nclasses):
    super().__init__()
    self.nclasses = nclasses
    self.avgpool = nn.AdaptiveAvgPool2d(1)
    self.fc = NTKLinear(512, nclasses)
    self.linear = None

  def reinit(self):
    self.fc.init()

  def freeze(self):
    self.fc.freeze()

  def thaw(self):
    self.fc.thaw()

  def linearize(self, static=True):
    if static:
      self.linear = NTKLinear(512, self.nclasses)
      self.linear.weight.data = self.fc.weight.data
      self.linear.bias.data = self.fc.bias.data
      self.linear.freeze()
    else:
      self.fc.thaw()
      self.linear = self.fc

  def forward(self, x, jvp=None):
    x = self.avgpool(x).flatten(1)
    logits = self.fc(x)
  
    if jvp is not None:
      assert self.linear is not None
      jvp = self.avgpool(jvp).flatten(1)
      jvp = self.linear(jvp, add_bias=False)

    return logits, jvp


class Net(nn.Module):
  """Network as either baseline or proposed method"""
  def __init__(self, nclasses):
    super().__init__()
    self.fnet = FeatureNet()
    self.hnet = STDHeadNet()
    self.clf = STDClassifier(nclasses)

  def load_fnet(self, fnet, freeze=True):
    self.fnet = fnet
    self.fnet.thaw()
    if freeze:
      self.fnet.freeze()

  def load_hnet(self, hnet, reinit_idx, freeze_idx, linearize_idx):
    self.hnet = hnet
    self.hnet.thaw()
    self.hnet.reinit(reinit_idx)
    self.hnet.freeze(freeze_idx)
    self.hnet.linearize(linearize_idx)

  def load_clf(self, clf, reinit=False, linearize=False, static=True):
    self.clf = clf
    self.clf.thaw()
    if reinit:      self.clf.reinit()
    if linearize:   self.clf.linearize(static)

  def forward(self, x):
    x, jvp = self.hnet(self.fnet(x))
    return self.clf(x, jvp)