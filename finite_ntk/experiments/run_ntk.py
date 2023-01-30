import numpy as np
import torch
import copy
import torch.nn as nn
import argparse
import sys

import gpytorch
import pickle

import finite_ntk

sys.path.append('../cifar')
from utils import train_epoch

class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        corr_shape = [x.shape[0], *self.shape]
        return x.view(*corr_shape)

class NTKGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, model, fisher=False):
        super(NTKGP, self).__init__(train_x, train_y, likelihood)

        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = finite_ntk.lazy.NTK(
            model=model, use_linearstrategy=fisher, used_dims=0
        )

    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)

def make_network():
    return torch.nn.Sequential(
            Reshape(1, 45, 45),
            torch.nn.Conv2d(1, 20, kernel_size=5),
            torch.nn.ReLU(True),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.Conv2d(20, 50, kernel_size=5),
            torch.nn.ReLU(True),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.Flatten(),
            torch.nn.Linear(3200, 500), 
            torch.nn.ReLU(True), 
            torch.nn.Linear(500, 2)
        )

def gaussian_loglikelihood(input, target, eps=1e-5):
    r"""
    heteroscedastic Gaussian likelihood where we parameterize the variance
    with the 1e-5 + softplus(network)
    input: tensor (batch + two-d, presumed to be output from model)
    target: tensor
    eps (1e-5): a nugget style term to ensure that the variance doesnt go to 0
    """
    dist = torch.distributions.Normal(
        input[:, 0], torch.nn.functional.softplus(input[:, 1]) + eps
    )
    res = dist.log_prob(target.view(-1))
    return res.mean()

def main(args):
    torch.random.manual_seed(args.seed)

    ## SET UP DATA ##
    dataset = np.load("./dataset/rotated_faces_data_withids.npz")
    train_images = dataset['train_images']
    train_targets = dataset['train_angles']

    test_images = dataset['test_images']
    test_targets = dataset['test_angles']
    #test_ids = dataset['test_people_ids']

    train_images = torch.from_numpy(train_images).float()
    train_targets = torch.from_numpy(train_targets).float()

    ### prepare adaptation and validation dataset
    num_to_keep = int(args.prop * test_images.shape[0])
    adapt_indices = np.random.permutation(test_images.shape[0])
    adapt_people = adapt_indices[:num_to_keep]
    val_people = adapt_indices[num_to_keep:]

    adapt_images = torch.from_numpy(test_images[adapt_people]).float()
    adapt_targets = torch.from_numpy(test_targets[adapt_people]).float()

    val_images = torch.from_numpy(test_images[val_people]).float()
    val_targets = torch.from_numpy(test_targets[val_people]).float()

    ##### standardize targets
    train_mean = train_targets.mean()
    train_std = train_targets.std()

    train_targets = (train_targets - train_mean) / train_std
    val_targets = (val_targets - train_mean) / train_std
    adapt_targets = (adapt_targets - train_mean) / train_std

    ##### set up data loaders
    # we have to reshape the inputs so that gpytorch internals can stack
    trainloader = torch.utils.data.DataLoader(
                    torch.utils.data.TensorDataset(train_images.reshape(train_images.shape[0], -1),
                        train_targets), batch_size=32, 
                    shuffle=True)

    adaptloader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(adapt_images.reshape(adapt_images.shape[0], -1),
                                    adapt_targets), batch_size=32, 
                                         shuffle=False)

    ###### make the network and set up optimizer
    net = make_network()
    net.cuda()  

    optimizer = torch.optim.Adam(net.parameters(), lr = 1e-3, amsgrad=True)

    lossfn = gaussian_loglikelihood

    ###### now train the network
    for i in range(args.epochs):
        #train_epoch(trainloader, net, gaussian_loglikelihood, optimizer)
        for input, target in adaptloader:
            input, target = input.cuda(), target.cuda()
        
            outputs = net(input)
            loss = -lossfn(outputs, target).sum()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()    
        
        if i % 10 is 0:
            with torch.no_grad():
                val_inputs = val_images.reshape(val_images.shape[0], -1).cuda()
                outputs = net(val_inputs)[:,0].cpu()
                rmse = torch.sqrt(torch.mean((outputs - val_targets)**2))
                print('Epoch: ', i, 'Test RMSE: ', rmse.item())
    
    ###### construct the GP model
    # we have to reshape the inputs so that gpytorch internals can stack
    likelihood = gpytorch.likelihoods.GaussianLikelihood().cuda()
    likelihood.noise = rmse.item()**2
    model = NTKGP(adapt_images.reshape(adapt_images.shape[0], -1), adapt_targets, 
                  likelihood, net, fisher=args.fisher).cuda() 

    # set in eval mode  
    likelihood.eval()
    model.eval()
    
    # compute predictive mean
    with gpytorch.settings.fast_pred_var(True):
        results = model(val_images.reshape(val_images.shape[0], -1).cuda())
    gp_predictions = results.mean.detach().cpu()
    gp_rmse = torch.sqrt(torch.mean((gp_predictions - val_targets)**2))
    print('GP RMSE: ', gp_rmse)

    # testing RMSE: 
    output = net(val_images.reshape(val_images.shape[0], -1).cuda())[:,0].detach()
    network_predictions = output.cpu()
    net_rmse = torch.sqrt(torch.mean((network_predictions - val_targets)**2))
    print('Final Net RMSE: ', net_rmse)

    ########## now we create a copy to re-train the last layer
    # which is a poor baseline here
    frozen_net = copy.deepcopy(net)
    parlength = len(list(frozen_net.named_parameters()))
    for i, (n, p) in enumerate(frozen_net.named_parameters()):
        if i < (parlength - 2):
            p.requires_grad = False
        else:
            # print out the name of the layer to verify we are only using the last layer
            print(n)

    #### train the last layer
    f_optimizer = torch.optim.Adam(frozen_net.parameters(), lr = 1e-3, amsgrad=True)
    for i in range(args.adapt_epochs):
        #train_epoch(adaptloader, frozen_net, gaussian_loglikelihood, f_optimizer)
        for input, target in adaptloader:
            input, target = input.cuda(), target.cuda()
        
            outputs = net(input)
            loss = -lossfn(outputs, target).sum()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        if i % 10 is 0:
            with torch.no_grad():
                val_inputs = val_images.reshape(val_images.shape[0], -1).cuda()
                outputs = frozen_net(val_inputs)[:,0].cpu()
                rmse = torch.sqrt(torch.mean((outputs - val_targets)**2))
                print('Epoch: ', i, 'Test RMSE: ', rmse.item())

    # testing RMSE: 
    output = frozen_net(val_images.reshape(val_images.shape[0], -1).cuda())[:,0].detach()
    frozen_network_predictions = output.cpu()
    fnet_rmse = torch.sqrt(torch.mean((frozen_network_predictions - val_targets)**2))
    print('Frozen Net RMSE: ', fnet_rmse)

    ##### and finally we "fine-tune" the whole net
    for i in range(args.adapt_epochs):
        #train_epoch(trainloader, net, gaussian_loglikelihood, optimizer)
        for input, target in adaptloader:
            input, target = input.cuda(), target.cuda()
        
            outputs = net(input)
            loss = -lossfn(outputs, target).sum()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        if i % 10 is 0:
            with torch.no_grad():
                val_inputs = val_images.reshape(val_images.shape[0], -1).cuda()
                outputs = net(val_inputs)[:,0].cpu()
                rmse = torch.sqrt(torch.mean((outputs - val_targets)**2))
                print('Epoch: ', i, 'Test RMSE: ', rmse.item())

    # testing RMSE: 
    output = net(val_images.reshape(val_images.shape[0], -1).cuda())[:,0].detach()
    rt_network_predictions = output.cpu()
    rt_net_rmse = torch.sqrt(torch.mean((rt_network_predictions - val_targets)**2))
    print('Retrained Net RMSE: ', rt_net_rmse)

    output_dict = {'targets': val_targets,
        'init_rmse': net_rmse, 
        'gp_rmse': gp_rmse,
        'frozen_rmse': fnet_rmse,
        'retrain_rmse': rt_net_rmse,
        'gp_preds': gp_predictions,
        'rt_preds': rt_network_predictions,
        'frozen_preds': frozen_network_predictions
    }    
    with open(args.output_file, "wb") as handle:
        pickle.dump(output_dict, handle, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--epochs", help="(int) number of epochs to train for", default=150, type=int
    )
    parser.add_argument(
        "--adapt_epochs", help="(int) number of epochs to train for", default=15, type=int
    )
    parser.add_argument(
        "--prop", help="(float) proportion of validation points", default=0.5, type=float
    )
    parser.add_argument(
        "--fisher", action="store_true", help="fisher basis usage flag (default: off)"
    )
    parser.add_argument("--seed", help="random seed", type=int, default=10)
    parser.add_argument("--output_file", type=str)
    args = parser.parse_args()
    main(args)
       
