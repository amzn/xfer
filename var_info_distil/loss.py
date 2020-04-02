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

class TemperatureScaledKLDivLoss(nn.Module):
    """
    Temperature scaled Kullback-Leibler divergence loss for knowledge distillation (Hinton et al., 
    https://arxiv.org/abs/1503.02531)
    
    :param float temperature: parameter for softening the distribution to be compared.
    """

    def __init__(self, temperature):
        super(TemperatureScaledKLDivLoss, self).__init__()
        self.temperature = temperature
        self.kullback_leibler_divergence = nn.KLDivLoss(reduction="batchmean")

    def forward(self, y_pred, y):
        """
        Output the temperature scaled Kullback-Leibler divergence loss for given the prediction and the target.
        :param torch.Tensor y_pred: unnormalized prediction for logarithm of the target.
        :param torch.Tensor y: probabilities representing the target.
        """
        log_p = torch.log_softmax(y_pred / self.temperature, dim=1)
        q = torch.softmax(y / self.temperature, dim=1)

        # Note that the Kullback-Leibler divergence is re-scaled by the squared temperature parameter.
        loss = (self.temperature ** 2) * self.kullback_leibler_divergence(log_p, q)
        return loss


class GaussianLoss(nn.Module):
    """
    Gaussian loss for transfer learning with variational information distillation.    
    """

    def forward(self, y_pred, y):
        """
        Output the Gaussian loss given the prediction and the target.
        :param tuple(torch.Tensor, torch.Tensor) y_pred: predicted mean and variance for the Gaussian 
        distribution.
        :param torch.Tensor y: target for the Gaussian distribution.
        """
        y_pred_mean, y_pred_var = y_pred
        loss = torch.mean(0.5 * ((y_pred_mean - y) ** 2 / y_pred_var + torch.log(y_pred_var)))
        return loss


class EnsembleKnowledgeTransferLoss(nn.Module):
    """
    Knowledge transfer loss as an ensemble of individual knowledge transfer losses defined on predicting the label, 
    logits of the teacher model, and features of the teacher model. 
    
    :param torch.nn.Module label_criterion: criterion for predicting the labels.
    :param torch.nn.Module teacher_logit_criterion: criterion for predicting the logit of the teacher model.
    :param torch.nn.Module teacher_feature_criterion: criterion for predicting the feature of the teacher model.
    :param float teacher_logit_factor: scaling factor for predicting the logit of the teacher model.
    :param float teacher_feature_factor: scaling factor for predicting the feature of the teacher model.
    """

    def __init__(
        self,
        label_criterion,
        teacher_logit_criterion,
        teacher_feature_criterion,
        teacher_logit_factor,
        teacher_feature_factor,
    ):
        super(EnsembleKnowledgeTransferLoss, self).__init__()
        self.label_criterion = label_criterion
        self.teacher_logit_criterion = teacher_logit_criterion
        self.teacher_feature_criterion = teacher_feature_criterion

        self.teacher_logit_factor = teacher_logit_factor
        self.teacher_feature_factor = teacher_feature_factor

    def forward(self, logit, label, teacher_feature_preds, teacher_logit, teacher_features):
        """
        Output the ensemble of knowledge transfer losses given the predictions and the targets.
        :param torch.Tensor logit: logit of the student model for predicting the label and logit of the teacher model. 
        :param torch.Tensor label: target label of the image.
        :param tuple(tuple(torch.Tensor)) teacher_feature_preds: predictions of the student model made on features of 
        the teacher model.
        :param torch.Tensor teacher_logit: logit of the teacher model to predict from the the student model.
        :param tuple(torch.Tensor) teacher_features: features of the teacher model to predict from the student model.
        """
        label_loss = self.label_criterion(logit, label)
        teacher_logit_loss = self.teacher_logit_criterion(logit, teacher_logit)
        teacher_feature_losses = [
            self.teacher_feature_criterion(pred, feature) for pred, feature in zip(teacher_feature_preds, teacher_features)
        ]
        loss = (
            label_loss
            + self.teacher_logit_factor * teacher_logit_loss
            + self.teacher_feature_factor * sum(teacher_feature_losses)
        )

        return loss
