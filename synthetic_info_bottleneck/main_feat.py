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

import itertools
import json
import os
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import MultiStepLR

from utils.outils import progress_bar, AverageMeter, accuracy, getCi
from utils.utils import to_device

from dataset import dataset_setting
from networks import get_featnet
from dataloader import TrainLoader, ValLoader

randomSeed = 123
torch.backends.cudnn.deterministic = True
torch.manual_seed(randomSeed)


#############################################################################################
class ClassifierEval(nn.Module):
    '''
    There is nothing to be learned in this classifier
    it is only used to evaluate netFeat episodically
    '''
    def __init__(self, nKnovel, nFeat):
        super(ClassifierEval, self).__init__()

        self.nKnovel = nKnovel
        self.nFeat = nFeat

        # bias & scale of classifier
        self.bias = nn.Parameter(torch.FloatTensor(1).fill_(0), requires_grad=False)
        self.scale_cls = nn.Parameter(torch.FloatTensor(1).fill_(10), requires_grad=False)

    def apply_classification_weights(self, features, cls_weights):
        '''
        (B x n x nFeat, B x nKnovel x nFeat) -> B x n x nKnovel
        (B x n x nFeat, B x nKnovel*nExamplar x nFeat) -> B x n x nKnovel*nExamplar if init_type is nn
        '''
        features = F.normalize(features, p=2, dim=features.dim()-1, eps=1e-12)
        cls_weights = F.normalize(cls_weights, p=2, dim=cls_weights.dim()-1, eps=1e-12)
        cls_scores = self.scale_cls * torch.baddbmm(1.0, self.bias.view(1, 1, 1), 1.0, features, cls_weights.transpose(1,2))
        return cls_scores

    def forward(self, features_supp, features_query):
        '''
        features_supp: (B, nKnovel * nExamplar, nFeat)
        features_query: (B, nKnovel * nTest, nFeat)
        '''
        B = features_supp.size(0)

        weight = features_supp.view(B, self.nKnovel, -1, self.nFeat).mean(2)
        cls_scores = self.apply_classification_weights(features_query, weight)

        return cls_scores


class ClassifierTrain(nn.Module):
    def __init__(self, nCls, nFeat=640, scaleCls = 10.):
        super(ClassifierTrain, self).__init__()

        self.scaleCls =  scaleCls
        self.nFeat =  nFeat
        self.nCls =  nCls

        # weights of base categories
        self.weight = torch.FloatTensor(nFeat, nCls).normal_(0.0, np.sqrt(2.0/nFeat)) # Dimension nFeat * nCls
        self.weight = nn.Parameter(self.weight, requires_grad=True)

        # bias
        self.bias = nn.Parameter(torch.FloatTensor(1, nCls).fill_(0), requires_grad=True) # Dimension 1 * nCls

        # Scale of cls (Heat Parameter)
        self.scaleCls = nn.Parameter(torch.FloatTensor(1).fill_(scaleCls), requires_grad=True)

        # Method
        self.applyWeight = self.applyWeightCosine

    def getWeight(self):
        return self.weight, self.bias, self.scaleCls

    def applyWeightCosine(self, feature, weight, bias, scaleCls):
        batchSize, nFeat =feature.size()

        feature = F.normalize(feature, p=2, dim=1, eps=1e-12) ## Attention: normalized along 2nd dimension!!!
        weight = F.normalize(weight, p=2, dim=0, eps=1e-12)## Attention: normalized along 1st dimension!!!

        clsScore = scaleCls * (torch.mm(feature, weight) )#+ bias)
        return clsScore

    def forward(self, feature):
        weight, bias, scaleCls = self.getWeight()
        clsScore = self.applyWeight(feature, weight, bias, scaleCls)
        return clsScore


class BaseTrainer:
    def __init__(self, trainLoader, valLoader, nbCls, nClsEpisode, nFeat,
                 outDir, milestones=[50], inputW=80, inputH=80, cuda=False):

        self.trainLoader = trainLoader
        self.valLoader = valLoader
        self.outDir = outDir
        self.milestones = milestones
        if not os.path.isdir(self.outDir):
            os.mkdir(self.outDir)

        # Define model
        self.netFeat, nFeat = get_featnet('WRN_28_10', inputW, inputH)
        self.netClassifier = ClassifierTrain(nbCls)
        self.netClassifierVal = ClassifierEval(nClsEpisode, nFeat)

        # GPU setting
        self.device = torch.device('cuda' if cuda else 'cpu')
        if cuda:
            self.netFeat.cuda()
            self.netClassifier.cuda()
            self.netClassifierVal.cuda()

        self.criterion = nn.CrossEntropyLoss()
        self.bestAcc = 0

    def LrWarmUp(self, totalIter, lr):
        msg = '\nLearning rate warming up'
        print(msg)

        self.optimizer = torch.optim.SGD(
                itertools.chain(*[self.netFeat.parameters(),
                                self.netClassifier.parameters()]),
                                1e-7,
                                momentum=0.9,
                                weight_decay=5e-4,
                                nesterov=True)

        nbIter = 0
        lrUpdate = lr
        valTop1 = 0

        while nbIter < totalIter :
            self.netFeat.train()
            self.netClassifier.train()
            losses = AverageMeter()
            top1 = AverageMeter()
            top5 = AverageMeter()

            for batchIdx, (inputs, targets) in enumerate(self.trainLoader):
                nbIter += 1
                if nbIter == totalIter:
                    break

                lrUpdate = nbIter / float(totalIter) * lr
                for g in self.optimizer.param_groups:
                    g['lr'] = lrUpdate

                inputs = to_device(inputs, self.device)
                targets = to_device(targets, self.device)

                self.optimizer.zero_grad()
                outputs = self.netFeat(inputs)
                outputs = self.netClassifier(outputs)
                loss = self.criterion(outputs, targets)

                loss.backward()
                self.optimizer.step()

                acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
                losses.update(loss.item(), inputs.size()[0])
                top1.update(acc1[0].item(), inputs.size()[0])
                top5.update(acc5[0].item(), inputs.size()[0])

                msg = 'Loss: {:.3f} | Lr : {:.5f} | Top1: {:.3f}% | Top5: {:.3f}%'.format(
                        losses.avg, lrUpdate, top1.avg, top5.avg)
                progress_bar(batchIdx, len(self.trainLoader), msg)

        with torch.no_grad():
            valTop1 = self.test(0)

        self.optimizer = torch.optim.SGD(
                itertools.chain(*[self.netFeat.parameters(),
                                self.netClassifier.parameters()]),
                                lrUpdate,
                                momentum=0.9,
                                weight_decay=5e-4,
                                nesterov=True)

        self.lrScheduler = MultiStepLR(self.optimizer, milestones=self.milestones, gamma=0.1)
        return valTop1

    def train(self, epoch):
        msg = '\nTrain at Epoch: {:d}'.format(epoch)
        print (msg)

        self.netFeat.train()
        self.netClassifier.train()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        for batchIdx, (inputs, targets) in enumerate(self.trainLoader):

            inputs = to_device(inputs, self.device)
            targets = to_device(targets, self.device)

            self.optimizer.zero_grad()
            outputs = self.netFeat(inputs)
            outputs = self.netClassifier(outputs)
            loss = self.criterion(outputs, targets)

            loss.backward()
            self.optimizer.step()

            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
            losses.update(loss.item(), inputs.size()[0])
            top1.update(acc1[0].item(), inputs.size()[0])
            top5.update(acc5[0].item(), inputs.size()[0])

            msg = 'Loss: {:.3f} | Top1: {:.3f}% | Top5: {:.3f}%'.format(losses.avg, top1.avg, top5.avg)
            progress_bar(batchIdx, len(self.trainLoader), msg)

        return losses.avg, top1.avg, top5.avg

    def test(self, epoch):
        msg = '\nTest at Epoch: {:d}'.format(epoch)
        print (msg)

        self.netFeat.eval()
        self.netClassifierVal.eval()

        top1 = AverageMeter()

        for batchIdx, data in enumerate(self.valLoader):
            data = to_device(data, self.device)

            SupportTensor, SupportLabel, QueryTensor, QueryLabel = \
                data['SupportTensor'].squeeze(0), data['SupportLabel'].squeeze(0), \
                data['QueryTensor'].squeeze(0), data['QueryLabel'].squeeze(0)

            SupportFeat, QueryFeat = self.netFeat(SupportTensor), self.netFeat(QueryTensor)
            SupportFeat, QueryFeat = SupportFeat.unsqueeze(0), QueryFeat.unsqueeze(0)

            clsScore = self.netClassifierVal(SupportFeat, QueryFeat)
            clsScore = clsScore.view(QueryFeat.size()[1], -1)

            acc1 = accuracy(clsScore, QueryLabel, topk=(1, ))
            top1.update(acc1[0].item(), clsScore.size()[0])
            msg = 'Top1: {:.3f}%'.format(top1.avg)
            progress_bar(batchIdx, len(self.valLoader), msg)

        ## Save checkpoint.
        acc = top1.avg
        if acc > self.bestAcc:
            print ('Saving Best')
            torch.save(self.netFeat.state_dict(), os.path.join(self.outDir, 'netFeatBest.pth'))
            torch.save(self.netClassifier.state_dict(), os.path.join(self.outDir, 'netClsBest.pth'))
            self.bestAcc = acc

        print('Saving Last')
        torch.save(self.netFeat.state_dict(), os.path.join(self.outDir, 'netFeatLast.pth'))
        torch.save(self.netClassifier.state_dict(), os.path.join(self.outDir, 'netClsLast.pth'))

        msg = 'Best Performance: {:.3f}'.format(self.bestAcc)
        print(msg)
        return top1.avg


#############################################################################################
## Parameters
parser = argparse.ArgumentParser(description='Base/FeatureNet Classification')
parser.add_argument('--lr', default=1e-1, type=float, help='learning rate')
parser.add_argument('--weightDecay', default=5e-4, type=float, help='weight decay')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--outDir', type=str, help='output directory')
parser.add_argument('--batchSize', type = int, default = 64, help='batch size')
parser.add_argument('--nbEpoch', type = int, default = 120, help='nb epoch')
parser.add_argument('--cuda', action='store_true', help='whether to use gpu')
parser.add_argument('--resumeFeatPth', type = str, help='resume feature Path')
parser.add_argument('--resumeClassifierPth', type = str, help='resume classifier Path')
parser.add_argument('--dataset', type=str, default='miniImageNet', choices=['miniImageNet', 'CUB', 'Cifar', 'tieredImageNet'], help='Which dataset? Should modify normalization parameter')
# Lr WarmUp
parser.add_argument('--totalIter', type = int, default=6000, help='total iterations for learning rate warm')
# Validation
parser.add_argument('--nFeat', type = int, default=640, help='feature dimension')

args = parser.parse_args()
print (args)


#############################################################################################
## datasets
trainTransform, valTransform, inputW, inputH, \
        trainDir, valDir, testDir, episodeJson, nbCls = \
        dataset_setting(args.dataset, 1)

trainLoader = TrainLoader(args.batchSize, trainDir, trainTransform)
valLoader = ValLoader(episodeJson, valDir, inputW, inputH, valTransform, args.cuda)

with open(episodeJson, 'r') as f:
    episodeInfo = json.load(f)

args.nClsEpisode = len(episodeInfo[0]['Support'])
args.nSupport = len(episodeInfo[0]['Support'][0])
args.nQuery = len(episodeInfo[0]['Query'][0])


#############################################################################################
## model

#milestones=[50, 80, 100]
milestones = [100] if args.dataset == 'CUB' else [50] # More epochs for CUB since less iterations / epoch

baseModel = BaseTrainer(trainLoader, valLoader, nbCls,
                        args.nClsEpisode, args.nFeat, args.outDir, milestones,
                        inputW, inputH,
                        args.cuda)

## Load pretrained model if there is
if args.resumeFeatPth :
    baseModel.netFeat.load_state_dict(torch.load(args.resumeFeatPth))
    msg = 'Loading weight from {}'.format(args.resumeFeatPth)
    print (msg)

if args.resumeClassifierPth :
    baseModel.netClassifier.load_state_dict(torch.load(args.resumeClassifierPth))
    msg = 'Loading weight from {}'.format(args.resumeClassifierPth)
    print (msg)



#############################################################################################
## main
valTop1 = baseModel.LrWarmUp(args.totalIter, args.lr)

testAccLog = []
trainAccLog = []

history = {'trainTop1':[], 'valTop1':[], 'trainTop5':[], 'trainLoss':[]}

for epoch in range(args.nbEpoch):
    trainLoss, trainTop1, trainTop5 = baseModel.train(epoch)
    with torch.no_grad() :
        valTop1 = baseModel.test(epoch)
    history['trainTop1'].append(trainTop1)
    history['trainTop5'].append(trainTop5)
    history['trainLoss'].append(trainLoss)
    history['valTop1'].append(valTop1)

    with open(os.path.join(args.outDir, 'history.json'), 'w') as f :
        json.dump(history, f)
    baseModel.lrScheduler.step()

## Finish training!!!
msg = 'mv {} {}'.format(os.path.join(args.outDir, 'netFeatBest.pth'), os.path.join(args.outDir, 'netFeatBest{:.3f}.pth'.format(baseModel.bestAcc)))
print (msg)
os.system(msg)

msg = 'mv {} {}'.format(os.path.join(args.outDir, 'netFeatLast.pth'), os.path.join(args.outDir, 'netFeatLast{:.3f}.pth'.format(valTop1)))
print (msg)
os.system(msg)

msg = 'mv {} {}'.format(os.path.join(args.outDir, 'netClsBest.pth'), os.path.join(args.outDir, 'netClsBest{:.3f}.pth'.format(baseModel.bestAcc)))
print (msg)
os.system(msg)

msg = 'mv {} {}'.format(os.path.join(args.outDir, 'netClsLast.pth'), os.path.join(args.outDir, 'netClsLast{:.3f}.pth'.format(valTop1)))
print (msg)
os.system(msg)

msg = 'mv {} {}'.format(args.outDir, '{}_{:.3f}'.format(args.outDir, valTop1))
print (msg)
os.system(msg)

