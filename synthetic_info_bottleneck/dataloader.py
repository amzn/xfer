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

import os
import torch
import torch.utils.data as data
import PIL.Image as Image
import numpy as np
import json

from torchvision import transforms
from torchvision.datasets import ImageFolder


def PilLoaderRGB(imgPath) :
    return Image.open(imgPath).convert('RGB')


class EpisodeSampler():
    """
    Dataloader to sample a task/episode.
    In case of 5-way 1-shot: nSupport = 1, nClsEpisode = 5.

    :param string imgDir: image directory, each category is in a sub file;
    :param int nClsEpisode: number of classes in each episode;
    :param int nSupport: number of support examples;
    :param int nQuery: number of query examples;
    :param transform: image transformation/data augmentation;
    :param bool useGPU: whether to use gpu or not;
    :param int inputW: input image size, dimension W;
    :param int inputH: input image size, dimension H;
    """
    def __init__(self, imgDir, nClsEpisode, nSupport, nQuery, transform, useGPU, inputW, inputH):
        self.imgDir = imgDir
        self.clsList = os.listdir(imgDir)
        self.nClsEpisode = nClsEpisode
        self.nSupport = nSupport
        self.nQuery = nQuery
        self.transform = transform

        floatType = torch.cuda.FloatTensor if useGPU else torch.FloatTensor
        intType = torch.cuda.LongTensor if useGPU else torch.LongTensor

        self.tensorSupport = floatType(nClsEpisode * nSupport, 3, inputW, inputH)
        self.labelSupport = intType(nClsEpisode * nSupport)
        self.tensorQuery = floatType(nClsEpisode * nQuery, 3, inputW, inputH)
        self.labelQuery = intType(nClsEpisode * nQuery)
        self.imgTensor = floatType(3, inputW, inputH)

    def getEpisode(self):
        """
        Return an episode

        :return dict: {'SupportTensor': 1 x nSupport x 3 x H x W,
                       'SupportLabel': 1 x nSupport,
                       'QueryTensor': 1 x nQuery x 3 x H x W,
                       'QueryLabel': 1 x nQuery}
        """
        # labels {0, ..., nClsEpisode-1}
        for i in range(self.nClsEpisode) :
            self.labelSupport[i * self.nSupport : (i+1) * self.nSupport] = i
            self.labelQuery[i * self.nQuery : (i+1) * self.nQuery] = i

        # select nClsEpisode from clsList
        clsEpisode = np.random.choice(self.clsList, self.nClsEpisode, replace=False)
        for i, cls in enumerate(clsEpisode) :
            clsPath = os.path.join(self.imgDir, cls)
            imgList = os.listdir(clsPath)

            # in total nQuery+nSupport images from each class
            imgCls = np.random.choice(imgList, self.nQuery + self.nSupport, replace=False)

            for j in range(self.nSupport) :
                img = imgCls[j]
                imgPath = os.path.join(clsPath, img)
                I = PilLoaderRGB(imgPath)
                self.tensorSupport[i * self.nSupport + j] = self.imgTensor.copy_(self.transform(I))

            for j in range(self.nQuery) :
                img = imgCls[j + self.nSupport]
                imgPath = os.path.join(clsPath, img)
                I = PilLoaderRGB(imgPath)
                self.tensorQuery[i * self.nQuery + j] = self.imgTensor.copy_(self.transform(I))

        ## Random permutation. Though this is not necessary in our approach
        permSupport = torch.randperm(self.nClsEpisode * self.nSupport)
        permQuery = torch.randperm(self.nClsEpisode * self.nQuery)

        return {'SupportTensor':self.tensorSupport[permSupport],
                'SupportLabel':self.labelSupport[permSupport],
                'QueryTensor':self.tensorQuery[permQuery],
                'QueryLabel':self.labelQuery[permQuery]
                }


class BatchSampler():
    """
    Dataloader to sample a task/episode.
    In case of 5-way 1-shot: nSupport = 1, nClsEpisode = 5.

    :param string imgDir: image directory, each category is in a sub file;
    :param int nClsEpisode: number of classes in each episode;
    :param int nSupport: number of support examples;
    :param int nQuery: number of query examples;
    :param transform: image transformation/data augmentation;
    :param bool useGPU: whether to use gpu or not;
    :param int inputW: input image size, dimension W;
    :param int inputH: input image size, dimension H;
    :param int batchSize: batch size (number of episode in each batch).
    """
    def __init__(self, imgDir, nClsEpisode, nSupport, nQuery, transform, useGPU, inputW, inputH, batchSize):
        self.episodeSampler = EpisodeSampler(imgDir, nClsEpisode, nSupport, nQuery,
                                             transform, useGPU, inputW, inputH)

        floatType = torch.cuda.FloatTensor if useGPU else torch.FloatTensor
        intType = torch.cuda.LongTensor if useGPU else torch.LongTensor

        self.tensorSupport = floatType(batchSize, nClsEpisode * nSupport, 3, inputW, inputH)
        self.labelSupport = intType(batchSize, nClsEpisode * nSupport)
        self.tensorQuery = floatType(batchSize, nClsEpisode * nQuery, 3, inputW, inputH)
        self.labelQuery = intType(batchSize, nClsEpisode * nQuery)

        self.batchSize = batchSize

    def getBatch(self):
        """
        Return an episode

        :return dict: {'SupportTensor': B x nSupport x 3 x H x W,
                       'SupportLabel': B x nSupport,
                       'QueryTensor': B x nQuery x 3 x H x W,
                       'QueryLabel': B x nQuery}
        """
        for i in range(self.batchSize) :
            episode = self.episodeSampler.getEpisode()
            self.tensorSupport[i] = episode['SupportTensor']
            self.labelSupport[i] = episode['SupportLabel']
            self.tensorQuery[i] = episode['QueryTensor']
            self.labelQuery[i] = episode['QueryLabel']

        return {'SupportTensor':self.tensorSupport,
                'SupportLabel':self.labelSupport,
                'QueryTensor':self.tensorQuery,
                'QueryLabel':self.labelQuery
                }


class ValImageFolder(data.Dataset):
    """
    To make validation results comparable, we fix 2000 episodes for validation.

    :param string episodeJson: ./data/Dataset/val1000Episode_K_way_N_shot.json
    :param string imgDir: image directory, each category is in a sub file;
    :param int inputW: input image size, dimension W;
    :param int inputH: input image size, dimension H;
    :param valTransform: image transformation/data augmentation;
    :param bool useGPU: whether to use gpu or not;
    """
    def __init__(self, episodeJson, imgDir, inputW, inputH, valTransform, useGPU):
        with open(episodeJson, 'r') as f :
            self.episodeInfo = json.load(f)

        self.imgDir = imgDir
        self.nEpisode = len(self.episodeInfo)
        self.nClsEpisode = len(self.episodeInfo[0]['Support'])
        self.nSupport = len(self.episodeInfo[0]['Support'][0])
        self.nQuery = len(self.episodeInfo[0]['Query'][0])
        self.transform = valTransform
        floatType = torch.cuda.FloatTensor if useGPU else torch.FloatTensor
        intType = torch.cuda.LongTensor if useGPU else torch.LongTensor

        self.tensorSupport = floatType(self.nClsEpisode * self.nSupport, 3, inputW, inputH)
        self.labelSupport = intType(self.nClsEpisode * self.nSupport)
        self.tensorQuery = floatType(self.nClsEpisode * self.nQuery, 3, inputW, inputH)
        self.labelQuery = intType(self.nClsEpisode * self.nQuery)

        self.imgTensor = floatType(3, inputW, inputH)
        for i in range(self.nClsEpisode) :
            self.labelSupport[i * self.nSupport : (i+1) * self.nSupport] = i
            self.labelQuery[i * self.nQuery : (i+1) * self.nQuery] = i


    def __getitem__(self, index):
        """
        Return an episode

        :param int index: index of data example
        :return dict: {'SupportTensor': 1 x nSupport x 3 x H x W,
                       'SupportLabel': 1 x nSupport,
                       'QueryTensor': 1 x nQuery x 3 x H x W,
                       'QueryLabel': 1 x nQuery}
        """
        for i in range(self.nClsEpisode) :
            for j in range(self.nSupport) :
                imgPath = os.path.join(self.imgDir, self.episodeInfo[index]['Support'][i][j])
                I = PilLoaderRGB(imgPath)
                self.tensorSupport[i * self.nSupport + j] = self.imgTensor.copy_(self.transform(I))

            for j in range(self.nQuery) :
                imgPath = os.path.join(self.imgDir, self.episodeInfo[index]['Query'][i][j])
                I = PilLoaderRGB(imgPath)
                self.tensorQuery[i * self.nQuery + j] = self.imgTensor.copy_(self.transform(I))

        return {'SupportTensor':self.tensorSupport,
                'SupportLabel':self.labelSupport,
                'QueryTensor':self.tensorQuery,
                'QueryLabel':self.labelQuery
                }

    def __len__(self):
        """
        Number of episodes
        """
        return self.nEpisode


def ValLoader(episodeJson, imgDir, inputW, inputH, valTransform, useGPU) :
    dataloader = data.DataLoader(ValImageFolder(episodeJson, imgDir, inputW, inputH,
                                                valTransform, useGPU),
                                 shuffle=False)
    return dataloader


def TrainLoader(batchSize, imgDir, trainTransform) :
    dataloader = data.DataLoader(ImageFolder(imgDir, trainTransform),
                                 batch_size=batchSize, shuffle=True, drop_last=True)
    return dataloader


if __name__ == '__main__' :
    import torchvision.transforms as transforms
    mean = [x/255.0 for x in [120.39586422,  115.59361427, 104.54012653]]
    std = [x/255.0 for x in [70.68188272,  68.27635443,  72.54505529]]
    normalize = transforms.Normalize(mean=mean, std=std)
    trainTransform = transforms.Compose([
                                         transforms.RandomCrop(80, padding=8),
                                         transforms.RandomHorizontalFlip(),
                                         lambda x: np.asarray(x),
                                         transforms.ToTensor(),
                                         normalize
                                        ])

    TrainEpisodeSampler = EpisodeSampler(imgDir = '../data/Mini-ImageNet/train_train/',
                                        nClsEpisode = 5,
                                        nSupport = 5,
                                        nQuery = 14,
                                        transform = trainTransform,
                                        useGPU = True,
                                        inputW = 80,
                                        inputH = 80)
    data = TrainEpisodeSampler.getEpisode()
    print (data['SupportLabel'])

