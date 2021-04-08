# Transfer Learning via Linearized Neural Networks

This repository contains a GPyTorch implementation of finite width neural tangent kernels from the paper [(link)](https://arxiv.org/abs/2103.01439) 

*Fast Adaptation with Linearized Neural Networks*

by Wesley Maddox, Shuai Tang, Pablo Garcia Moreno, Andrew Gordon Wilson, and Andreas Damianou,

which appeared at AISTATS 2021. Please note that this is a revised and expanded version of the workshop paper [On Transfer Learning with Linearised Neural Networks](http://metalearning.ml/2019/papers/metalearn2019-maddox.pdf), which appeared at the 3rd MetaLearning Workshop at NeurIPS, 2019.

## Introduction


Please cite our work if you find it useful:
```
@inproceedings{maddox2021fast,
  title={Fast Adaptation with Linearized Neural Networks},
  author={Maddox, Wesley and Tang, Shuai and Moreno, Pablo and Wilson, Andrew Gordon and Damianou, Andreas},
  booktitle={International Conference on Artificial Intelligence and Statistics},
  pages={2737--2745},
  year={2021},
  organization={PMLR}
}
```

## Installation:

```bash
python setup.py develop
```

See requirements.txt file for requirements that came from our setup. We use Pytorch 1.3.1 and Python 3.6+ in our experiments.

Unless otherwise described, all experiments were run on a single GPU.

## Minimal Example

```python
import torch
import gpytorch
import finite_ntk

data = torch.randn(300, 1)
response = torch.randn(300, 1)

# randomly initialize a neural network
model = torch.nn.Sequential(torch.nn.Linear(1, 30), 
                            torch.nn.ReLU(),
                            torch.nn.BatchNorm1d(),
                            torch.nn.Linear(30, 1))

class ExactGPModel(gpytorch.models.ExactGP):
    # exact RBF Gaussian process class
    def __init__(self, train_x, train_y, likelihood, model):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = finite_ntk.lazy.NTK(
            model=model)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

gp_lh = gpytorch.likelihoods.GaussianLikelihood()
gp_model = ExactGPModel(data, response, gp_lh, model)

# draw a sample from the GP with kernel given by Jacobian of model
zeromean_pred = gp_lh(gp_model(data)).sample()
```


## References for Code Base

GPyTorch: [Pytorch repo](https://github.com/cornellius-gp/gpytorch); this is the origin of the codebase.

Adam Paszke's gist for the [Rop](https://gist.github.com/apaszke/c7257ac04cb8debb82221764f6d117ad)

We'd like to thank Max Balandat for providing us a cleaned version of the malaria data files from [Balandat et al, 2019](https://arxiv.org/abs/1910.06403) and Jacob Gardner and Marc Finzi for 
help with the Fisher vector products.

The Malaria Global Atlas data file can be downloaded at: https://wjmaddox.github.io/assets/data/malaria_df.hdf5   

## Authors  
Wesley Maddox
