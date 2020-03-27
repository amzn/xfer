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
import itertools
import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from utils.outils import progress_bar, AverageMeter, accuracy, getCi
from utils.utils import to_device

class Algorithm:
    """
    Algorithm logic is implemented here with training and validation functions etc.

    :param args: experimental configurations
    :type args: EasyDict
    :param logger: logger
    :param netFeat: feature network
    :type netFeat: class `WideResNet` or `ConvNet_4_64`
    :param netSIB: Classifier/decoder
    :type netSIB: class `ClassifierSIB`
    :param optimizer: optimizer
    :type optimizer: torch.optim.SGD
    :param criterion: loss
    :type criterion: nn.CrossEntropyLoss
    """
    def __init__(self, args, logger, netFeat, netSIB, optimizer, criterion):
        self.netFeat = netFeat
        self.netSIB = netSIB
        self.optimizer = optimizer
        self.criterion = criterion

        self.nbIter = args.nbIter
        self.nStep = args.nStep
        self.outDir = args.outDir
        self.nFeat = args.nFeat
        self.batchSize = args.batchSize
        self.nEpisode = args.nEpisode
        self.momentum = args.momentum
        self.weightDecay = args.weightDecay

        self.logger = logger
        self.device = torch.device('cuda' if args.cuda else 'cpu')

        # Load pretrained model
        if args.resumeFeatPth :
            if args.cuda:
                param = torch.load(args.resumeFeatPth)
            else:
                param = torch.load(args.resumeFeatPth, map_location='cpu')
            self.netFeat.load_state_dict(param)
            msg = '\nLoading netFeat from {}'.format(args.resumeFeatPth)
            self.logger.info(msg)

        if args.test:
            self.load_ckpt(args.ckptPth)


    def load_ckpt(self, ckptPth):
        """
        Load checkpoint from ckptPth.

        :param ckptPth: the path to the ckpt
        :type ckptPth: string
        """
        param = torch.load(ckptPth)
        self.netFeat.load_state_dict(param['netFeat'])
        self.netSIB.load_state_dict(param['SIB'])
        lr = param['lr']
        self.optimizer = torch.optim.SGD(itertools.chain(*[self.netSIB.parameters(),]),
                                         lr,
                                         momentum=self.momentum,
                                         weight_decay=self.weightDecay,
                                         nesterov=True)
        msg = '\nLoading networks from {}'.format(ckptPth)
        self.logger.info(msg)


    def compute_grad_loss(self, clsScore, QueryLabel):
        """
        Compute the loss between true gradients and synthetic gradients.
        """
        # register hooks
        def require_nonleaf_grad(v):
            def hook(g):
                v.grad_nonleaf = g
            h = v.register_hook(hook)
            return h
        handle = require_nonleaf_grad(clsScore)

        loss = self.criterion(clsScore, QueryLabel)
        loss.backward(retain_graph=True) # need to backward again

        # remove hook
        handle.remove()

        gradLogit = self.netSIB.dni(clsScore) # B * n x nKnovel
        gradLoss = F.mse_loss(gradLogit, clsScore.grad_nonleaf.detach())

        return loss, gradLoss


    def validate(self, valLoader, lr=None, mode='val'):
        """
        Run one epoch on val-set.

        :param valLoader: the dataloader of val-set
        :type valLoader: class `ValLoader`
        :param float lr: learning rate for synthetic GD
        :param string mode: 'val' or 'train'
        """
        if mode == 'test':
            nEpisode = self.nEpisode
            self.logger.info('\n\nTest mode: randomly sample {:d} episodes...'.format(nEpisode))
        elif mode == 'val':
            nEpisode = len(valLoader)
            self.logger.info('\n\nValidation mode: pre-defined {:d} episodes...'.format(nEpisode))
            valLoader = iter(valLoader)
        else:
            raise ValueError('mode is wrong!')

        episodeAccLog = []
        top1 = AverageMeter()

        self.netFeat.eval()
        #self.netSIB.eval() # set train mode, since updating bn helps to estimate better gradient

        if lr is None:
            lr = self.optimizer.param_groups[0]['lr']

        #for batchIdx, data in enumerate(valLoader):
        for batchIdx in range(nEpisode):
            data = valLoader.getEpisode() if mode == 'test' else next(valLoader)
            data = to_device(data, self.device)

            SupportTensor, SupportLabel, QueryTensor, QueryLabel = \
                    data['SupportTensor'].squeeze(0), data['SupportLabel'].squeeze(0), \
                    data['QueryTensor'].squeeze(0), data['QueryLabel'].squeeze(0)

            with torch.no_grad():
                SupportFeat, QueryFeat = self.netFeat(SupportTensor), self.netFeat(QueryTensor)
                SupportFeat, QueryFeat, SupportLabel = \
                        SupportFeat.unsqueeze(0), QueryFeat.unsqueeze(0), SupportLabel.unsqueeze(0)

            clsScore = self.netSIB(SupportFeat, SupportLabel, QueryFeat, lr)
            clsScore = clsScore.view(QueryFeat.shape[0] * QueryFeat.shape[1], -1)
            QueryLabel = QueryLabel.view(-1)
            acc1 = accuracy(clsScore, QueryLabel, topk=(1,))
            top1.update(acc1[0].item(), clsScore.shape[0])

            msg = 'Top1: {:.3f}%'.format(top1.avg)
            progress_bar(batchIdx, nEpisode, msg)
            episodeAccLog.append(acc1[0].item())

        mean, ci95 = getCi(episodeAccLog)
        self.logger.info('Final Perf with 95% confidence intervals: {:.3f}%, {:.3f}%'.format(mean, ci95))
        return mean, ci95


    def train(self, trainLoader, valLoader, lr=None, coeffGrad=0.0) :
        """
        Run one epoch on train-set.

        :param trainLoader: the dataloader of train-set
        :type trainLoader: class `TrainLoader`
        :param valLoader: the dataloader of val-set
        :type valLoader: class `ValLoader`
        :param float lr: learning rate for synthetic GD
        :param float coeffGrad: deprecated
        """
        bestAcc, ci = self.validate(valLoader, lr)
        self.logger.info('Acc improved over validation set from 0% ---> {:.3f} +- {:.3f}%'.format(bestAcc,ci))

        self.netSIB.train()
        self.netFeat.eval()

        losses = AverageMeter()
        top1 = AverageMeter()
        history = {'trainLoss' : [], 'trainAcc' : [], 'valAcc' : []}

        for episode in range(self.nbIter):
            data = trainLoader.getBatch()
            data = to_device(data, self.device)

            with torch.no_grad() :
                SupportTensor, SupportLabel, QueryTensor, QueryLabel = \
                        data['SupportTensor'], data['SupportLabel'], data['QueryTensor'], data['QueryLabel']
                nC, nH, nW = SupportTensor.shape[2:]

                SupportFeat = self.netFeat(SupportTensor.reshape(-1, nC, nH, nW))
                SupportFeat = SupportFeat.view(self.batchSize, -1, self.nFeat)

                QueryFeat = self.netFeat(QueryTensor.reshape(-1, nC, nH, nW))
                QueryFeat = QueryFeat.view(self.batchSize, -1, self.nFeat)

            if lr is None:
                lr = self.optimizer.param_groups[0]['lr']

            self.optimizer.zero_grad()

            clsScore = self.netSIB(SupportFeat, SupportLabel, QueryFeat, lr)
            clsScore = clsScore.view(QueryFeat.shape[0] * QueryFeat.shape[1], -1)
            QueryLabel = QueryLabel.view(-1)

            if coeffGrad > 0:
                loss, gradLoss = self.compute_grad_loss(clsScore, QueryLabel)
                loss = loss + gradLoss * coeffGrad
            else:
                loss = self.criterion(clsScore, QueryLabel)

            loss.backward()
            self.optimizer.step()

            acc1 = accuracy(clsScore, QueryLabel, topk=(1, ))
            top1.update(acc1[0].item(), clsScore.shape[0])
            losses.update(loss.item(), QueryFeat.shape[1])
            msg = 'Loss: {:.3f} | Top1: {:.3f}% '.format(losses.avg, top1.avg)
            if coeffGrad > 0:
                msg = msg + '| gradLoss: {:.3f}%'.format(gradLoss.item())
            progress_bar(episode, self.nbIter, msg)

            if episode % 1000 == 999 :
                acc, _ = self.validate(valLoader, lr)

                if acc > bestAcc :
                    msg = 'Acc improved over validation set from {:.3f}% ---> {:.3f}%'.format(bestAcc , acc)
                    self.logger.info(msg)

                    bestAcc = acc
                    self.logger.info('Saving Best')
                    torch.save({
                                'lr': lr,
                                'netFeat': self.netFeat.state_dict(),
                                'SIB': self.netSIB.state_dict(),
                                'nbStep': self.nStep,
                                }, os.path.join(self.outDir, 'netSIBBest.pth'))

                self.logger.info('Saving Last')
                torch.save({
                            'lr': lr,
                            'netFeat': self.netFeat.state_dict(),
                            'SIB': self.netSIB.state_dict(),
                            'nbStep': self.nStep,
                            }, os.path.join(self.outDir, 'netSIBLast.pth'))

                msg = 'Iter {:d}, Train Loss {:.3f}, Train Acc {:.3f}%, Val Acc {:.3f}%'.format(
                        episode, losses.avg, top1.avg, acc)
                self.logger.info(msg)
                history['trainLoss'].append(losses.avg)
                history['trainAcc'].append(top1.avg)
                history['valAcc'].append(acc)

                losses = AverageMeter()
                top1 = AverageMeter()

        return bestAcc, acc, history
