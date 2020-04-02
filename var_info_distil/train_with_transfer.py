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

# This file trains a student model (following the wide residual network architecture) with knowledge transfer from the
# pretrained teacher model. Available knowledge transfer methods are variational information distillation (Ahn et al.,
# http://openaccess.thecvf.com/content_CVPR_2019/papers/Ahn_Variational_Information_Distillation_for_Knowledge_Transfer_CVPR_2019_paper.pdf
# ) and knowledge distillation (Hinton et al., https://arxiv.org/abs/1503.02531).

import os
import argparse
import torch
import ignite

from functools import partial
from ignite.engine import Engine, Events

from cifar10 import get_cifar10_loaders
from wide_residual_network import (
    BATCH_SIZE,
    LEARNING_RATE,
    MOMENTUM,
    WEIGHT_DECAY,
    LEARNING_RATE_DECAY_MILESTONES,
    LEARNING_RATE_DECAY_FACTOR,
    TeacherWideResidualNetwork,
    StudentWideResidualNetwork,
)
from loss import TemperatureScaledKLDivLoss, GaussianLoss, EnsembleKnowledgeTransferLoss
from util import (
    BatchUpdaterWithTransfer,
    BatchEvaluator,
    prepare_batch,
    LearningRateUpdater,
    MetricLogger,
    attach_pbar_and_metrics,
)


def main(
    student_depth,
    student_width,
    teacher_depth,
    teacher_width,
    max_epochs,
    variational_information_distillation_factor,
    knowledge_distillation_factor,
    knowledge_distillation_temperature,
    state_dict_path,
    teacher_state_dict_path,
    device,
    data_dir,
    num_workers,
):
    """
    This function constructs and trains a student model with knowledge transfer from the pretrained teacher model. 
    
    :param int student_depth: factor for controlling the depth of the student model.
    :param int student_width: factor for controlling the width of the student model.
    :param int teacher_depth: factor for controlling the depth of the teacher model. 
    :param int teacher_width: factor for controlling the width of the teacher model.
    :param int max_epochs: maximum number of epochs for training the student model.
    :param float variational_information_distillation_factor: scaling factor for variational information distillation.
    :param float knowledge_distillation_factor: scaling factor for knowledge distillation.
    :param float knowledge_distillation_temperature: degree of smoothing on distributions for computing the Kuback-Leibler 
    divergence for knowledge distillation. 
    :param string state_dict_path: path to save the student model.
    :param string teacher_state_dict_path: path to load the teacher model from.
    :param int device: device to use for training the model
    :param string data_dir: directory to save and load the dataset.
    :param int num_workers: number of workers to use for loading the dataset.
    """

    # Define the device for training the model.
    device = torch.device(device)

    # Get data loaders for the CIFAR-10 dataset.
    train_loader, validation_loader, test_loader = get_cifar10_loaders(
        data_dir, batch_size=BATCH_SIZE, num_workers=num_workers
    )

    # Construct the student model to be trained.
    model = StudentWideResidualNetwork(depth=student_depth, width=student_width, teacher_width=teacher_width)
    model = model.to(device)

    # Construct and load the teacher model for guiding the student model.
    teacher_model = TeacherWideResidualNetwork(
        depth=teacher_depth, width=teacher_width, load_path=teacher_state_dict_path
    )
    teacher_model = teacher_model.to(device)

    # Define optimizer and learning rate scheduler
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=LEARNING_RATE_DECAY_MILESTONES, gamma=LEARNING_RATE_DECAY_FACTOR
    )

    # Construct the loss function to be used for training.
    label_criterion = torch.nn.CrossEntropyLoss()
    teacher_logit_criterion = TemperatureScaledKLDivLoss(temperature=knowledge_distillation_temperature)
    teacher_feature_criterion = GaussianLoss()
    criterion = EnsembleKnowledgeTransferLoss(
        label_criterion=label_criterion,
        teacher_logit_criterion=teacher_logit_criterion,
        teacher_feature_criterion=teacher_feature_criterion,
        teacher_logit_factor=knowledge_distillation_factor,
        teacher_feature_factor=variational_information_distillation_factor,
    )

    # Define the ignite engines for training and evaluation.
    batch_updater = BatchUpdaterWithTransfer(
        model=model, teacher_model=teacher_model, optimizer=optimizer, criterion=criterion, device=device
    )
    batch_evaluator = BatchEvaluator(model=model, device=device)
    trainer = Engine(batch_updater)
    evaluator = Engine(batch_evaluator)

    # Define and attach the progress bar, loss metric, and the accuracy metrics.
    attach_pbar_and_metrics(trainer, evaluator)

    # The training engine updates the learning rate schedule at end of each epoch.
    lr_updater = LearningRateUpdater(lr_scheduler=lr_scheduler) 
    trainer.on(Events.EPOCH_COMPLETED(every=1))(lr_updater)

    # The training engine logs the training and the evaluation metrics at end of each epoch.
    metric_logger = MetricLogger(evaluator=evaluator, eval_loader=validation_loader)
    trainer.on(Events.EPOCH_COMPLETED(every=1))(metric_logger)

    # Train the model
    trainer.run(train_loader, max_epochs=max_epochs)

    # Save the model to pre-defined path. We move the model to CPU which is desirable as the default device
    # for loading the model.
    model.cpu()
    state_dict_dir = "/".join(state_dict_path.split("/")[:-1])
    os.makedirs(state_dict_dir, exist_ok=True)
    torch.save(model.state_dict(), state_dict_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--student-depth", type=int, default=16, help="factor for controlling the depth of the student model."
    )
    parser.add_argument(
        "--student-width", type=int, default=1, help="factor for controlling the width of the student model."
    )
    parser.add_argument(
        "--teacher-depth", type=int, default=40, help="factor for controlling the depth of the teacher model."
    )
    parser.add_argument(
        "--teacher-width", type=int, default=2, help="factor for controlling the width of the teacher model."
    )
    parser.add_argument(
        "--max-epochs", type=int, default=200, help="maximum number of epochs for training the student model."
    )
    parser.add_argument(
        "--variational-information-distillation-factor",
        type=float,
        default=0.1,
        help="scaling factor for variational information distillation.",
    )
    parser.add_argument(
        "--knowledge-distillation-factor", type=float, default=1.0, help="scaling factor for knowledge distillation"
    )
    parser.add_argument(
        "--knowledge-distillation-temperature",
        type=float,
        default=2.0,
        help="temperature factor for knowledge distillation",
    )
    parser.add_argument(
        "--state-dict-path",
        type=str,
        default="./state_dict/student_with_transfer.th",
        help="path to save the student model",
    )
    parser.add_argument(
        "--teacher-state-dict-path",
        type=str,
        default="./state_dict/teacher.th",
        help="path to load the teacher model from",
    )
    parser.add_argument("--device", type=int, default=0, help="device to use for training the model.")
    parser.add_argument("--data-dir", type=str, default="./data/", help="directory to save and load the dataset.")
    parser.add_argument("--num-workers", type=int, default=4, help="number of workers to use for loading the dataset.")
    args = parser.parse_args()

    main(
        student_depth=args.student_depth,
        student_width=args.student_width,
        teacher_depth=args.teacher_depth,
        teacher_width=args.teacher_width,
        max_epochs=args.max_epochs,
        variational_information_distillation_factor=args.variational_information_distillation_factor,
        knowledge_distillation_factor=args.knowledge_distillation_factor,
        knowledge_distillation_temperature=args.knowledge_distillation_temperature,
        state_dict_path=args.state_dict_path,
        teacher_state_dict_path=args.teacher_state_dict_path,
        device=args.device,
        data_dir=args.data_dir,
        num_workers=args.num_workers,
    )
