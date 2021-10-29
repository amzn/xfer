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

## copied from https://github.com/fmu2/gradfeat20/src/

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.utils.data as data
from torchvision.datasets import SVHN, CIFAR10, CIFAR100
from torchvision import transforms


class NTKConv2d(nn.Module):
  """Conv2d layer under NTK parametrization."""
  def __init__(self, in_channels, out_channels, kernel_size, stride=1, 
    padding=0, bias=True, zero_init=False):
    super().__init__()
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.kernel_size = kernel_size
    self.stride = stride
    self.padding = padding

    self.bias = None
    self.weight = nn.Parameter(torch.Tensor(
      out_channels, in_channels, kernel_size, kernel_size))
    if bias:
      self.bias = nn.Parameter(torch.Tensor(out_channels))
    self.init(zero_init)

  def init(self, zero_init=False):
    if zero_init:
      nn.init.constant_(self.weight, 0.)
      if self.bias is not None:
        nn.init.constant_(self.bias, 0.)
    else:
      nn.init.normal_(self.weight, 0., 1.)
      if self.bias is not None:
        nn.init.normal_(self.bias, 0., 1.)

  def freeze(self):
    for p in self.parameters():
      p.requires_grad = False

  def thaw(self):
    for p in self.parameters():
      p.requires_grad = True

  def forward(self, x, add_bias=True):
    weight = np.sqrt(1. / self.out_channels) * self.weight
    if add_bias and self.bias is not None:
      bias = np.sqrt(.1) * self.bias
      return F.conv2d(x, weight, bias, self.stride, self.padding)
    else:
      return F.conv2d(x, weight, None, self.stride, self.padding)


class NTKLinear(nn.Module):
  """Linear layer under NTK parametrization."""
  def __init__(self, in_features, out_features, bias=True, zero_init=False):
    super().__init__()
    self.in_features = in_features
    self.out_features = out_features

    self.bias = None
    self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
    if bias:
      self.bias = nn.Parameter(torch.Tensor(out_features))
    self.init(zero_init)

  def init(self, zero_init=False):
    if zero_init:
      nn.init.constant_(self.weight, 0.)
      if self.bias is not None:
        nn.init.constant_(self.bias, 0.)
    else:
      nn.init.normal_(self.weight, 0., 1.)
      if self.bias is not None:
        nn.init.normal_(self.bias, 0., 1.)

  def freeze(self):
    for p in self.parameters():
      p.requires_grad = False

  def thaw(self):
    for p in self.parameters():
      p.requires_grad = True

  def forward(self, x, add_bias=True):
    weight = np.sqrt(1. / self.out_features) * self.weight
    if add_bias and self.bias is not None:
      bias = np.sqrt(.1) * self.bias
      return F.linear(x, weight, bias)
    else:
      return F.linear(x, weight, None)


def std_to_ntk_conv2d(conv2d):
  """STD Conv2d -> NTK Conv2d"""
  if isinstance(conv2d, NTKConv2d):
    return conv2d
  bias = True if conv2d.bias is not None else False
  ntk_conv2d = NTKConv2d(conv2d.in_channels, conv2d.out_channels, 
    conv2d.kernel_size[0], conv2d.stride, conv2d.padding, bias=bias)
  # parameter rescaling (see Jacot et al. NeurIPS 18)
  ntk_conv2d.weight.data = conv2d.weight.data / np.sqrt(1. / conv2d.out_channels)
  if bias:
    ntk_conv2d.bias.data = conv2d.bias.data / np.sqrt(.1)
  return ntk_conv2d


def ntk_to_std_conv2d(conv2d):
  """NTK Conv2d -> STD Conv2d"""
  if isinstance(conv2d, nn.Conv2d):
    return conv2d
  bias = True if conv2d.bias is not None else False
  std_conv2d = nn.Conv2d(conv2d.in_channels, conv2d.out_channels, 
    conv2d.kernel_size[0], conv2d.stride, conv2d.padding, bias=bias)
  # parameter rescaling (see Jacot et al. NeurIPS 18)
  std_conv2d.weight.data = conv2d.weight.data * np.sqrt(1. / conv2d.out_channels)
  if bias:
    std_conv2d.bias.data = conv2d.bias.data * np.sqrt(.1)
  return std_conv2d


def std_to_ntk_linear(fc):
  """STD Linear -> NTK Linear"""
  if isinstance(fc, NTKLinear):
    return fc
  bias = True if fc.bias is not None else False
  ntk_fc = NTKLinear(fc.in_features, fc.out_features)
  # parameter rescaling (see Jacot et al. NeurIPS 18)
  ntk_fc.weight.data = fc.weight.data / np.sqrt(1. / fc.out_features)
  if bias:
    ntk_fc.bias.data = fc.bias.data / np.sqrt(.1)
  return ntk_fc


def ntk_to_std_linear(fc):
  """NTK Linear -> STD Linear"""
  if isinstance(fc, NTKLinear):
    return fc
  bias = True if fc.bias is not None else False
  std_fc = NTKLinear(fc.in_features, fc.out_features)
  # parameter rescaling (see Jacot et al. NeurIPS 18)
  std_fc.weight.data = fc.weight.data * np.sqrt(1. / fc.out_features)
  if bias:
    std_fc.bias.data = fc.bias.data * np.sqrt(.1)
  return std_fc


def merge_batchnorm(conv2d, batchnorm):
  """Folds BatchNorm2d into Conv2d."""
  if isinstance(batchnorm, nn.Identity):
    return conv2d
  mean = batchnorm.running_mean
  sigma = torch.sqrt(batchnorm.running_var + batchnorm.eps)
  beta = batchnorm.weight
  gamma = batchnorm.bias

  w = conv2d.weight
  if conv2d.bias is not None:
    b = conv2d.bias
  else:
    b = torch.zeros_like(mean)

  w = w * (beta / sigma).view(conv2d.out_channels, 1, 1, 1)
  b = (b - mean) / sigma * beta + gamma

  fused_conv2d = nn.Conv2d(
    conv2d.in_channels, conv2d.out_channels, conv2d.kernel_size, 
    conv2d.stride, conv2d.padding)
  fused_conv2d.weight.data = w
  fused_conv2d.bias.data = b

  return fused_conv2d


def load_data(dataset, path, batch_size=64, normalize=False):
  if normalize:
    # Wasserstein BiGAN is trained on normalized data.
    transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
  else:
    # BiGAN is trained on unnormalized data (see Dumoulin et al. ICLR 16).
    transform = transforms.ToTensor()

  if dataset == 'svhn':
    train_set = SVHN(path, split='extra', transform=transform, download=True)
    val_set = SVHN(path, split='test', transform=transform, download=True)

  elif dataset == 'cifar10':
    train_set = CIFAR10(path, train=True, transform=transform, download=True)
    val_set = CIFAR10(path, train=False, transform=transform, download=True)

  elif dataset == 'cifar100':
    train_set = CIFAR100(path, train=True, transform=transform, download=True)
    val_set = CIFAR100(path, train=False, transform=transform, download=True)

  train_loader = data.DataLoader(
    train_set, batch_size, shuffle=True, num_workers=12)
  val_loader = data.DataLoader(
    val_set, 1, shuffle=False, num_workers=1, pin_memory=True)
  return train_loader, val_loader
