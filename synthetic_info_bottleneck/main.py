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

import torch
import torch.nn as nn
import random
import itertools
import json
import os

from algorithm import Algorithm
from networks import get_featnet
from sib import ClassifierSIB
from dataset import dataset_setting
from dataloader import BatchSampler, ValLoader, EpisodeSampler
from utils.config import get_config
from utils.utils import get_logger, set_random_seed

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

#############################################################################################
## Read hyper-parameters
args = get_config()

# Setup logging to file and stdout
logger = get_logger(args.logDir, args.expName)

# Fix random seed to reproduce results
set_random_seed(args.seed)
logger.info('Start experiment with random seed: {:d}'.format(args.seed))
logger.info(args)

# GPU setup
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
if args.gpu != '':
    args.cuda = True
device = torch.device('cuda' if args.cuda else 'cpu')

#############################################################################################
## Datasets
trainTransform, valTransform, inputW, inputH, \
        trainDir, valDir, testDir, episodeJson, nbCls = \
        dataset_setting(args.dataset, args.nSupport)

trainLoader = BatchSampler(imgDir = trainDir,
                           nClsEpisode = args.nClsEpisode,
                           nSupport = args.nSupport,
                           nQuery = args.nQuery,
                           transform = trainTransform,
                           useGPU = args.cuda,
                           inputW = inputW,
                           inputH = inputH,
                           batchSize = args.batchSize)

valLoader = ValLoader(episodeJson,
                      valDir,
                      inputW,
                      inputH,
                      valTransform,
                      args.cuda)

testLoader = EpisodeSampler(imgDir = testDir,
                            nClsEpisode = args.nClsEpisode,
                            nSupport = args.nSupport,
                            nQuery = args.nQuery,
                            transform = valTransform,
                            useGPU = args.cuda,
                            inputW = inputW,
                            inputH = inputH)


#############################################################################################
## Networks
netFeat, args.nFeat = get_featnet(args.architecture, inputW, inputH)
netFeat = netFeat.to(device)
netSIB = ClassifierSIB(args.nClsEpisode, args.nFeat, args.nStep)
netSIB = netSIB.to(device)

## Optimizer
optimizer = torch.optim.SGD(itertools.chain(*[netSIB.parameters(),]),
                            args.lr,
                            momentum=args.momentum,
                            weight_decay=args.weightDecay,
                            nesterov=True)

## Loss
criterion = nn.CrossEntropyLoss()

## Algorithm class
alg = Algorithm(args, logger, netFeat, netSIB, optimizer, criterion)


#############################################################################################
## Training
if not args.test:
    bestAcc, lastAcc, history = alg.train(trainLoader, valLoader, coeffGrad=args.coeffGrad)

    ## Finish training!!!
    msg = 'mv {} {}'.format(os.path.join(args.outDir, 'netSIBBest.pth'),
                            os.path.join(args.outDir, 'netSIBBest{:.3f}.pth'.format(bestAcc)))
    logger.info(msg)
    os.system(msg)

    msg = 'mv {} {}'.format(os.path.join(args.outDir, 'netSIBLast.pth'),
                            os.path.join(args.outDir, 'netSIBLast{:.3f}.pth'.format(lastAcc)))
    logger.info(msg)
    os.system(msg)

    with open(os.path.join(args.outDir, 'history.json'), 'w') as f :
        json.dump(history, f)

    msg = 'mv {} {}'.format(args.outDir, '{}_{:.3f}'.format(args.outDir, bestAcc))
    logger.info(msg)
    os.system(msg)


#############################################################################################
## Testing
logger.info('Testing model {}...'.format(args.ckptPth if args.test else 'LAST'))
mean, ci95 = alg.validate(testLoader, mode='test')

if not args.test:
    logger.info('Testing model BEST...')
    alg.load_ckpt(os.path.join('{}_{:.3f}'.format(args.outDir, bestAcc),
                               'netSIBBest{:.3f}.pth'.format(bestAcc)))
    mean, ci95 = alg.validate(testLoader, mode='test')
