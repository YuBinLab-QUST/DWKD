import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.blocks import Convolution
from monai.networks.layers import Act, Norm
import pdb

from utils.utils import dynamic_kd_loss, cross_entropy, kl_div, kd_loss_f


class UncertaintyTeacherKDForSequenceClassification(nn.Module):
    def __init__(self,
                 kd_alpha=0.5,
                 ce_alpha=0.5,
                 en_alpha=0.,
                 t=0.9,
                 loss_func=None,
                 temperature=5.0,
                 student=None,
                 ende="en",
                 dy_loss: bool = True
                 ):
        super().__init__()
        self.student = student
        self.kd_alpha = kd_alpha
        self.ce_alpha = ce_alpha
        self.en_alpha = en_alpha
        self.temperature = temperature
        self.loss_func = loss_func
        self.ende = ende
        self.dy_loss = dy_loss
        self.t = t

    def forward(self, inputs=None, labels=None, teacher_logits=None):
        loss = 0.
        if self.training:
            student_logits = self.student(inputs)
        else:
            student_logits = self.student(inputs)
            return student_logits
        if self.dy_loss:
            kd_loss = dynamic_kd_loss(student_logits, teacher_logits, self.temperature)
        else:
            kd_loss = kd_loss_f(student_logits, teacher_logits, self.temperature)
        entropy_loss = cross_entropy(student_logits, torch.sigmoid(teacher_logits), self.t).mean()
        dice_loss = self.loss_func(student_logits, labels)
        if self.en_alpha != 0.:
            loss += self.en_alpha * entropy_loss
        loss += self.ce_alpha * dice_loss
        if teacher_logits is not None:
            loss += self.kd_alpha * kd_loss
        return loss, kd_loss, dice_loss, entropy_loss
