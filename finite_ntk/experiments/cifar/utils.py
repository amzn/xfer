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
import torch
import os

import numpy as np
import tqdm


def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr


def save_checkpoint(dir, epoch, name="checkpoint", **kwargs):
    state = {"epoch": epoch}
    state.update(kwargs)
    filepath = os.path.join(dir, "%s-%d.pt" % (name, epoch))
    torch.save(state, filepath)


def train_epoch(
    loader, model, criterion, optimizer, cuda=True, verbose=False, subset=None,
):
    loss_sum = 0.0
    verb_stage = 0

    num_objects_current = 0
    num_batches = len(loader)

    if subset is not None:
        num_batches = int(num_batches * subset)
        loader = itertools.islice(loader, num_batches)

    if verbose:
        loader = tqdm.tqdm(loader, total=num_batches)

    for i, (input, target) in enumerate(loader):
        if cuda:
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

        loss, _ = criterion(model, input, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_sum += loss.data.item() * input.size(0)

        num_objects_current += input.size(0)

        if verbose and 10 * (i + 1) / num_batches >= verb_stage + 1:
            print(
                "Stage {:d}/10. Loss {:12.4f}.".format(verb_stage + 1, loss_sum / num_objects_current,)
            )
            verb_stage += 1

    return {
        "loss": loss_sum / num_objects_current,
    }


def eval(loader, model, criterion, cuda=True, verbose=False):
    loss_sum = 0.0
    correct = 0.0
    num_objects_total = len(loader.dataset)

    with torch.no_grad():
        if verbose:
            loader = tqdm.tqdm(loader)
        for i, (input, target) in enumerate(loader):
            if cuda:
                input = input.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)

            loss, output = criterion(model, input, target)

            loss_sum += loss.item() * input.size(0)

            pred = output.data.argmax(1, keepdim=True)
            correct += pred.eq(target.data.view_as(pred)).sum().item()

    return {
        "loss": loss_sum / num_objects_total,
        "accuracy": correct / num_objects_total * 100,
    }


def predict(loader, model, criterion, verbose=False):
    predictions = list()
    targets = list()

    if verbose:
        loader = tqdm.tqdm(loader)

    with torch.no_grad():
        for input, target in loader:
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            _, batch_logprobs = criterion(model, input, target, return_predictions=True)
            batch_predictions = torch.nn.functional.softmax(batch_logprobs, dim=1)
            predictions.append(batch_predictions.cpu().numpy())
            targets.append(target.cpu().numpy())

    return {"predictions": np.vstack(predictions), "targets": np.concatenate(targets)}
