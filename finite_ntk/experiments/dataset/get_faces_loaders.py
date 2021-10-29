import torch
import numpy as np


def get_faces_loaders(batch_size=128, test=True, data_path="./data/"):
    """
    returns the train (and test if selected) loaders for the olivetti
    rotated faces dataset
    """

    dat = np.load(data_path + "rotated_faces_data.npz")
    train_images = torch.FloatTensor(dat['train_images'])
    train_targets = torch.FloatTensor(dat['train_angles'])

    traindata = torch.utils.data.TensorDataset(train_images, train_targets)
    trainloader = torch.utils.data.DataLoader(traindata, batch_size=batch_size,
                                              shuffle=True)

    if test:
        test_images = torch.FloatTensor(dat['test_images'])
        test_targets = torch.FloatTensor(dat['test_angles'])

        testdata = torch.utils.data.TensorDataset(test_images, test_targets)
        testloader = torch.utils.data.DataLoader(testdata, batch_size=batch_size)

        return trainloader, testloader

    return trainloader
