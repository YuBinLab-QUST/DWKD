# -*- coding = utf-8 -*-
# @Time : 2023/7/11 20:33
# @Author：dianlong
# -*- coding = utf-8 -*-
# @Time : 2023/7/10 16:00
# @Author：dianlong
import argparse
import os

import numpy as np
import torch.utils.data.distributed
from monai.networks.nets import UNet, AttentionUnet

from models.student import UncertaintyTeacherKDForSequenceClassification
from utils.utils import get_feas_by_hook, quantile_threshold
from utils.data_utils import get_loader

parser = argparse.ArgumentParser(description='Swin UNETR segmentation pipeline for BRATS Challenge')
parser.add_argument('--checkpoint', default=None, help='start training from saved checkpoint')
parser.add_argument('--logdir', default='test', type=str, help='directory to save the tensorboard logs')
parser.add_argument('--fold', default=-1, type=int, help='data fold')
parser.add_argument('--data_dir', type=str, default="/opt/data/private/fengyan/2021/datasets/MICCAI_BraTS2020_TrainingData",
                    help='dataset directory')
parser.add_argument('--json_list', type=str, default="./jsons/brats20_folds.json", help='dataset json file')
parser.add_argument('--save_checkpoint', action='store_true', help='save checkpoint during training')
parser.add_argument('--max_epochs', default=1000, type=int, help='max number of training epochs')
parser.add_argument('--workers', default=0, type=int, help='number of workers')
parser.add_argument('--in_channels', default=4, type=int, help='number of input channels')
parser.add_argument('--out_channels', default=3, type=int, help='number of output channels')
parser.add_argument('--a_min', default=-175.0, type=float, help='a_min in ScaleIntensityRanged')
parser.add_argument('--a_max', default=250.0, type=float, help='a_max in ScaleIntensityRanged')
parser.add_argument('--b_min', default=0.0, type=float, help='b_min in ScaleIntensityRanged')
parser.add_argument('--b_max', default=1.0, type=float, help='b_max in ScaleIntensityRanged')
parser.add_argument('--space_x', default=1.5, type=float, help='spacing in x direction')
parser.add_argument('--space_y', default=1.5, type=float, help='spacing in y direction')
parser.add_argument('--space_z', default=2.0, type=float, help='spacing in z direction')
parser.add_argument('--roi_x', default=128, type=int, help='roi size in x direction')
parser.add_argument('--roi_y', default=128, type=int, help='roi size in y direction')
parser.add_argument('--roi_z', default=128, type=int, help='roi size in z direction')
parser.add_argument('--dropout_rate', default=0.0, type=float, help='dropout rate')
parser.add_argument('--dropout_path_rate', default=0.0, type=float, help='drop path rate')
parser.add_argument('--RandFlipd_prob', default=0.2, type=float, help='RandFlipd aug probability')
parser.add_argument('--RandRotate90d_prob', default=0.2, type=float, help='RandRotate90d aug probability')
parser.add_argument('--RandScaleIntensityd_prob', default=0.1, type=float, help='RandScaleIntensityd aug probability')
parser.add_argument('--RandShiftIntensityd_prob', default=0.1, type=float, help='RandShiftIntensityd aug probability')
parser.add_argument('--infer_overlap', default=0.5, type=float, help='sliding window inference overlap')
parser.add_argument('--resume_ckpt', action='store_true', help='resume training from pretrained checkpoint')
parser.add_argument('--smooth_dr', default=1e-6, type=float, help='constant added to dice denominator to avoid nan')
parser.add_argument('--smooth_nr', default=0.0, type=float, help='constant added to dice numerator to avoid zero')
parser.add_argument('--use_checkpoint', action='store_true', help='use gradient checkpointing to save memory')
parser.add_argument('--spatial_dims', default=3, type=int, help='spatial dimension of input data')
parser.add_argument('--pretrained_dir', type=str,
                    default='./pretrained_models/2020/AttentionUnetKD8-16_fold-1_0.2_CE0.8_en0.0_t5_ep1000_SKD_1',
                    help='pretrained checkpoint directory')
parser.add_argument('--pretrained_model_name', type=str, default="model_epoch799.pt", help='pretrained model name')
parser.add_argument('--squared_dice', action='store_true', help='use squared Dice')
parser.add_argument('--temperature', default=3.0, type=float, help='KL loss temperature')
parser.add_argument('--CUDA_VISIBLE_DEVICES', type=str, default="0,1,2", help='uncertainty mode')
parser.add_argument('--channels', type=int, nargs='+', help='uncertainty mode')
parser.add_argument('--strides', type=int, nargs='+', help='uncertainty mode')
parser.add_argument('--distributed', action='store_true', help='uncertainty mode')
parser.add_argument('--kd', action='store_true', default=True, help='uncertainty mode')
parser.add_argument('--batch_size', type=int, default=1, help='uncertainty mode')

args = parser.parse_args()
args.test_mode = False


def up_activate_maps_to_mask(activate_maps: list):
    activate_maps: torch.Tensor = torch.cat([hook.fea for hook in activate_maps], dim=1)
    activate_maps: torch.Tensor = torch.relu(activate_maps)
    activate_maps.resize((activate_maps.size()[0], activate_maps.size()[1], -1))
    activate_maps = torch.nn.functional.interpolate(activate_maps, size=(128, 128, 128), mode="trilinear")

    return activate_maps


if __name__ == '__main__':
    loader = get_loader(args)
    # region student
    student_model = AttentionUnet(spatial_dims=3,
                                  in_channels=4,
                                  out_channels=3,
                                  channels=(8, 16),
                                  strides=(2, 2,)
                                  )
    if args.kd:
        student_model = UncertaintyTeacherKDForSequenceClassification(
            student=student_model,
        )
    pretrained_dir = args.pretrained_dir
    model_name = args.pretrained_model_name
    pretrained_pth = os.path.join(pretrained_dir, args.pretrained_model_name)
    model_dict = torch.load(pretrained_pth)["state_dict"]
    student_model.load_state_dict(model_dict)
    student_model.cuda()
    if args.kd:
        student_model = student_model.student
    train_loader = loader[0]
    fea_hooks = get_feas_by_hook(student_model)
    targets = []
    feas = []
    intersections = torch.zeros((56, 3)).cuda()
    unions = torch.zeros((56, 3)).cuda()
    for idx, batch_data in enumerate(train_loader):
        print(idx)
        if isinstance(batch_data, list):
            data, target = batch_data
        else:
            data, target = batch_data['image'], batch_data['label']
        data, target = data.cuda(), target.cuda()

        student_logits = student_model(data)
        # student_logits = student_model.student(data)
        # 1.calculate threshold
        feas0 = torch.cat([fea_hooks[0].fea, fea_hooks[1].fea, fea_hooks[2].fea], dim=1)
        feas1 = torch.cat([fea_hooks[3].fea, fea_hooks[4].fea], dim=1)
        thresholds0 = quantile_threshold(torch.relu(feas0))
        thresholds1 = quantile_threshold(torch.relu(feas1))


        # 2.resize the activation map to match the label size.
        feas1 = torch.nn.functional.interpolate(feas1, (128, 128, 128), mode="trilinear", align_corners=True)
        feas0 = torch.cat([feas0, feas1], dim=1)
        thresholds = np.hstack((thresholds0, thresholds1))

        for i in range(feas0.size()[1]):
            # 3.Generate a one-hot segmentation map based on a threshold.
            indexes = feas0[:, i] > thresholds[i]
            # 4.Calculate the intersection and union of the activation map with WT, TC, and ET separately.
            y_o = torch.sum(target[0], dim=(1, 2, 3))  #
            y_pred_o = torch.sum(indexes, dim=(1, 2, 3))
            y_pred_o = torch.cat([y_pred_o, y_pred_o, y_pred_o], dim=0)
            intersection = torch.sum(torch.cat([indexes, indexes, indexes], dim=0) * target[0], dim=(1, 2, 3))
            union = y_o + y_pred_o - intersection
            # 5.Save the calculated intersection and union
            intersections[i] += intersection
            unions[i] += union
    # 6.Calculate the Intersection over Union (IoU) for each activation map across the entire dataset
    IoU = intersections / unions
    is_detector = IoU > 0.04
    detector_sum = is_detector.sum(dim=0)
    print(detector_sum)
    print(args.pretrained_dir, args.pretrained_model_name)
