# Copyright 2020 - 2022 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os

import torch
import numpy as np
from torch import nn
import pdb
import torch.nn.functional as F
from monai.networks.blocks import Convolution, ResidualUnit

from utils import vecquantile
import nibabel as nib

def dice(x, y):
    intersect = np.sum(np.sum(np.sum(x * y)))
    y_sum = np.sum(np.sum(np.sum(y)))
    if y_sum == 0:
        return 0.0
    x_sum = np.sum(np.sum(np.sum(x)))
    return 2 * intersect / (x_sum + y_sum)


class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = np.where(self.count > 0,
                            self.sum / self.count,
                            self.sum)


def distributed_all_gather(tensor_list,
                           valid_batch_size=None,
                           out_numpy=False,
                           world_size=None,
                           no_barrier=False,
                           is_valid=None):
    if world_size is None:
        world_size = torch.distributed.get_world_size()
    if valid_batch_size is not None:
        valid_batch_size = min(valid_batch_size, world_size)
    elif is_valid is not None:
        is_valid = torch.tensor(bool(is_valid), dtype=torch.bool, device=tensor_list[0].device)
    if not no_barrier:
        torch.distributed.barrier()
    tensor_list_out = []
    with torch.no_grad():
        if is_valid is not None:
            is_valid_list = [torch.zeros_like(is_valid) for _ in range(world_size)]
            torch.distributed.all_gather(is_valid_list, is_valid)
            is_valid = [x.item() for x in is_valid_list]
        for tensor in tensor_list:
            gather_list = [torch.zeros_like(tensor) for _ in range(world_size)]
            torch.distributed.all_gather(gather_list, tensor)
            if valid_batch_size is not None:
                gather_list = gather_list[:valid_batch_size]
            elif is_valid is not None:
                gather_list = [g for g, v in zip(gather_list, is_valid_list) if v]
            if out_numpy:
                gather_list = [t.cpu().numpy() for t in gather_list]
            tensor_list_out.append(gather_list)
    return tensor_list_out


# version 1: use torch.autograd
class LabelSmoothSoftmaxCEV1(nn.Module):
    '''
    This is the autograd version, you can also try the LabelSmoothSoftmaxCEV2 that uses derived gradients
    '''

    def __init__(self, lb_smooth=0.1, reduction='mean', ignore_index=-100):
        super(LabelSmoothSoftmaxCEV1, self).__init__()
        self.lb_smooth = lb_smooth
        self.reduction = reduction
        self.lb_ignore = ignore_index
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, logits, label):
        '''
        Same usage method as nn.CrossEntropyLoss:
            >>> criteria = LabelSmoothSoftmaxCEV1()
            >>> logits = torch.randn(8, 19, 384, 384) # nchw, float/half
            >>> lbs = torch.randint(0, 19, (8, 384, 384)) # nhw, int64_t
            >>> loss = criteria(logits, lbs)
        '''
        # overcome ignored label
        logits = logits.float()  # use fp32 to avoid nan
        with torch.no_grad():
            num_classes = logits.size(1)
            label = label.clone().detach()
            ignore = label.eq(self.lb_ignore)
            n_valid = ignore.eq(0).sum()
            label[ignore] = 0
            lb_pos, lb_neg = 1. - self.lb_smooth, self.lb_smooth / num_classes
            lb_one_hot = torch.empty_like(logits).fill_(
                lb_neg).scatter_(1, label.unsqueeze(1), lb_pos).detach()

        logs = self.log_softmax(logits)
        loss = -torch.sum(logs * lb_one_hot, dim=1)
        loss[ignore] = 0
        if self.reduction == 'mean':
            loss = loss.sum() / n_valid
        if self.reduction == 'sum':
            loss = loss.sum()

        return loss

class HookTool:
    def __init__(self):
        self.fea = None

    def hook_fun(self, module, fea_in, fea_out):
        '''
        注意用于处理feature的hook函数必须包含三个参数[module, fea_in, fea_out]，参数的名字可以自己起，但其意义是
        固定的，第一个参数表示torch里的一个子module，比如Linear,Conv2d等，第二个参数是该module的输入，其类型是
        tuple；第三个参数是该module的输出，其类型是tensor。注意输入和输出的类型是不一样的，切记。
        '''
        self.fea = fea_out


def get_feas_by_hook(model):
    """
    提取Conv2d后的feature，我们需要遍历模型的module，然后找到Conv2d，把hook函数注册到这个module上；
    这就相当于告诉模型，我要在Conv2d这一层，用hook_fun处理该层输出的feature.
    由于一个模型中可能有多个Conv2d，所以我们要用hook_feas存储下来每一个Conv2d后的feature
    """
    fea_hooks = []
    for n, m in model.named_modules():
        if isinstance(m, Convolution):
            if not m.is_transposed and hasattr(m,"adn"):
                cur_hook = HookTool()
                m.register_forward_hook(cur_hook.hook_fun)
                fea_hooks.append(cur_hook)

    return fea_hooks

def quantile_threshold(features):

    print("calculating quantile threshold")
    quant = vecquantile.QuantileVector(depth=features.shape[1], seed=1)
    batch_size = 1
    for i in range(0, features.shape[0], batch_size):
        batch = features[i:i + batch_size]
        batch = np.transpose(batch.cpu().detach().numpy(), axes=(0, 2, 3, 4, 1)).reshape(-1, features.shape[1])
        quant.add(batch)
    ret = quant.readout(1000)[:, int(1000 * (1-0.005)-1)]
    return ret
def load_images( ID, PATH_DATA='./', DIM=(192, 224, 160), VALID_SET=True):
    img1 = os.path.join(PATH_DATA, ID, ID + '_flair.nii.gz')
    img2 = os.path.join(PATH_DATA, ID, ID + '_t1.nii.gz')
    img3 = os.path.join(PATH_DATA, ID, ID + '_t1ce.nii.gz')
    img4 = os.path.join(PATH_DATA, ID, ID + '_t2.nii.gz')

    # combine the four imaging modalities (flair, t1, t1ce, t2)
    imgs_input = nib.concat_images([img1, img2, img3, img4],).get_fdata()

    imgs_preprocess = np.zeros((DIM[0], DIM[1], DIM[2], 4))  # (5, 192, 224, 160)
    if VALID_SET:
        for i in range(imgs_preprocess.shape[-1]):
            imgs_preprocess[:, :, :, i] = crop_image_brats(imgs_input[:, :, :, i],OUT_SHAPE=DIM)
            imgs_preprocess[:, :, :, i] = norm_image(imgs_preprocess[:, :, :, i])

    return imgs_preprocess[np.newaxis, ...]

def crop_image_brats(img, OUT_SHAPE=(192, 224, 160)):
    # manual cropping
    input_shape = np.array(img.shape)
    # center the cropped image
    offset = np.array((input_shape - OUT_SHAPE) / 2).astype(np.int8)
    offset[offset < 0] = 0
    x, y, z = offset
    crop_img = img[x:x + OUT_SHAPE[0], y:y + OUT_SHAPE[1], z:z + OUT_SHAPE[2]]
    # pad the preprocessed image
    padded_img = np.zeros(OUT_SHAPE)
    x, y, z = np.array((OUT_SHAPE - np.array(crop_img.shape)) / 2).astype(np.int8)
    padded_img[x:x + crop_img.shape[0], y:y + crop_img.shape[1], z:z + crop_img.shape[2]] = crop_img
    return padded_img

def norm_image(img, NORM_TYP="norm"):
    if NORM_TYP == "standard_norm": # standarization, same dataset
        img_mean = img.mean()
        img_std = img.std()
        img_std = 1 if img.std()==0 else img.std()
        img = (img - img_mean) / img_std
    elif NORM_TYP == "norm": # different datasets
        img = (img - np.min(img))/(np.ptp(img)) # (np.max(img) - np.min(img))
    elif NORM_TYP == "norm_slow": # different datasets
        img_ptp = 1 if np.ptp(img)== 0 else np.ptp(img)
        img = (img - np.min(img))/img_ptp
    return img

def cross_entropy(x, y, t=0.9):
    student_probs = torch.sigmoid(x)
    student_entropy = - y * torch.log(student_probs + 1e-10)  # student entropy, (bsz, )
    _y = torch.ones_like(y)
    _y[y >= t] = 0.
    student_entropy += - _y * torch.log((1 - student_probs) + 1e-10)

    return student_entropy


def kl_div(x, y):
    input = (torch.sigmoid(x) + 1e-10)
    input = torch.cat([input, 1 - input], dim=1)
    target = torch.sigmoid(y)
    target = torch.cat([target, 1 - target], dim=1)
    kl = F.kl_div(input, target, reduction="none", log_target=True)
    return kl


def dynamic_kd_loss(student_logits, teacher_logits, temperature=3.0) -> torch.Tensor:
    loss = 0.
    with torch.no_grad():
        student_probs = torch.sigmoid(student_logits)
        teacher_probs = torch.sigmoid(teacher_logits)
        student_entropy = - teacher_probs * torch.log(student_probs + 1e-10)  # student entropy, (bsz, )
        student_entropy += - (1 - teacher_probs) * torch.log((1 - student_probs) + 1e-10)  # student entropy, (bsz, )
        # normalized entropy score by student uncertainty:
        # i.e.,  entropy / entropy_upper_bound
        # higher uncertainty indicates the student is more confusing about this instance
        instance_weight = student_entropy / torch.max(student_entropy)

    batch_loss = kl_div(student_logits / temperature, teacher_logits / temperature) * temperature ** 2
    loss += torch.mean(batch_loss * torch.cat([instance_weight, instance_weight], dim=1))
    return loss


def kd_loss_f(student_logits, teacher_logits, temperature=3.0) -> torch.Tensor:
    loss = 0.
    batch_loss = kl_div(student_logits / temperature, teacher_logits / temperature) * temperature ** 2
    loss += torch.mean(batch_loss)
    return loss
