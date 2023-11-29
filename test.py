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

import argparse
import os
from functools import partial

import nibabel as nib
import numpy as np
import torch
from monai.inferers import sliding_window_inference
from monai.networks.nets import UNet, AttentionUnet
# from models import SwinUNETR
from models.student import UncertaintyTeacherKDForSequenceClassification
from utils.data_utils import get_loader

# from models.unetr import UNETR
parser = argparse.ArgumentParser(description='Swin UNETR segmentation pipeline')
parser.add_argument('--data_dir', default=r'F:\dataset\MICCAI_BraTS_2018_Data_Validation', type=str,
                    help='dataset directory')
parser.add_argument('--exp_name', default='', type=str, help='experiment name')
parser.add_argument('--json_list', default='./jsons/brats23_folds_v.json', type=str, help='dataset json file')
parser.add_argument('--fold', default=-1, type=int, help='data fold')
parser.add_argument('--pretrained_model_name', default='model.pt', type=str, help='pretrained model name')
parser.add_argument('--feature_size', default=16, type=int, help='feature size')
parser.add_argument('--infer_overlap', default=0.6, type=float, help='sliding window inference overlap')
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
parser.add_argument('--distributed', action='store_true', help='start distributed training')
parser.add_argument('--workers', default=0, type=int, help='number of workers')
parser.add_argument('--RandFlipd_prob', default=0.2, type=float, help='RandFlipd aug probability')
parser.add_argument('--RandRotate90d_prob', default=0.2, type=float, help='RandRotate90d aug probability')
parser.add_argument('--RandScaleIntensityd_prob', default=0.1, type=float, help='RandScaleIntensityd aug probability')
parser.add_argument('--RandShiftIntensityd_prob', default=0.1, type=float, help='RandShiftIntensityd aug probability')
parser.add_argument('--spatial_dims', default=3, type=int, help='spatial dimension of input data')
parser.add_argument('--use_checkpoint', action='store_true', help='use gradient checkpointing to save memory')
parser.add_argument('--pretrained_dir',
                    default='./pretrained_models/2021/MUnet32-64-128-256-512_ep1000_1/',
                    type=str,
                    help='pretrained checkpoint directory')
parser.add_argument('--device', default=0, type=int, help='spatial dimension of input data')
parser.add_argument('--type', default="voter", type=str, help='spatial dimension of input data')
parser.add_argument('--channels', type=int, nargs='+', help='uncertainty mode')
parser.add_argument('--strides', type=int, nargs='+', help='uncertainty mode')
parser.add_argument('--n', default=5, type=int, help='uncertainty mode')
parser.add_argument('--num_res_units', default=2, type=int, help='uncertainty mode')
parser.add_argument('--size', default=32, type=int, help='uncertainty mode')
parser.add_argument('--num_layers', default=6, type=int, help='uncertainty mode')
parser.add_argument('--tta', action='store_true', help='start distributed training')
parser.add_argument('--kd', action='store_true', help='start distributed training')
parser.add_argument('--data', default="2021", type=str, help='year of dataset')
dataname = {
    "2018": "Brats18",
    "2019": "BraTS19",
    "2020": "BraTS20",
    "2021": "BraTS20",
}
data_dir = {
    "2018": "../MICCAI_BraTS_2018_Data_Validation",
    "2019": "../MICCAI_BraTS_2019_Data_Validation",
    "2020": "../MICCAI_BraTS2020_ValidationData",
    "2021": r"E:\Datasets\ASNR-MICCAI-BraTS2023-GLI-Challenge-ValidationData",
}
json_list = {
    "2018": "./jsons/brats18_folds_v.json",
    "2019": "./jsons/brats19_folds_v.json",
    "2020": "./jsons/brats20_folds_v.json",
    "2021": "./jsons/brats23_folds_v.json",
}


def mean(model_inferer_tests, image, tta: bool = True):
    prob = torch.sigmoid(model_inferer_tests(image))  # 000
    if tta:
        prob += torch.sigmoid(model_inferer_tests(image.flip(dims=(2,))).flip(dims=(2,)))
        prob += torch.sigmoid(model_inferer_tests(image.flip(dims=(3,))).flip(dims=(3,)))
        prob += torch.sigmoid(model_inferer_tests(image.flip(dims=(4,))).flip(dims=(4,)))
        prob += torch.sigmoid(model_inferer_tests(image.flip(dims=(2, 3))).flip(dims=(2, 3)))
        prob += torch.sigmoid(model_inferer_tests(image.flip(dims=(2, 4))).flip(dims=(2, 4)))
        prob += torch.sigmoid(model_inferer_tests(image.flip(dims=(3, 4))).flip(dims=(3, 4)))
        prob += torch.sigmoid(model_inferer_tests(image.flip(dims=(2, 3, 4))).flip(dims=(2, 3, 4)))
        prob = (prob / 8.0)
    seg = prob[0].detach().cpu().numpy()
    seg = (seg > 0.5).astype(np.int8)
    return seg


def main():
    args = parser.parse_args()
    args.test_mode = True
    output_directory = './outputs/' + args.exp_name
    args.data_dir = data_dir[args.data]
    args.json_list = json_list[args.data]
    for key, val in args._get_kwargs():
        print("=====>" + key + ' : ' + str(val))
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    test_loader = get_loader(args)
    pretrained_dir = args.pretrained_dir
    model = UNet(spatial_dims=3,
                 in_channels=4,
                 out_channels=3,
                 channels=(32,64,128,256,512),
                 strides=(2,2,2,2),
                 num_res_units=2
                 )
    if args.kd:
        model = UncertaintyTeacherKDForSequenceClassification(
            student=model
        )
    model_dict = torch.load(os.path.join(pretrained_dir, args.pretrained_model_name))["state_dict"]
    model.load_state_dict(model_dict)
    model.eval()
    model.cuda(args.device)

    model_inferer_test = partial(
        sliding_window_inference,
        roi_size=[args.roi_x, args.roi_y, args.roi_z],
        sw_batch_size=1,
        predictor=model,
        overlap=args.infer_overlap,
    )
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            image = batch["image"].cuda(args.device)
            affine = batch['image_meta_dict']['original_affine'][0].numpy()
            num1 = batch['image_meta_dict']['filename_or_obj'][0].split('/')[-1].split('-')[-2]
            num2 = batch['image_meta_dict']['filename_or_obj'][0].split('/')[-1].split('-')[-3]
            if args.data == "2021":
                img_name = 'BraTS-GLI-{}-{}.nii.gz'.format( num2,num1 )
            else:
                img_name = '{}_{}_{}_1.nii.gz'.format(dataname[args.data], num1, num2)
            print("Inference on case {}".format(img_name))
            seg = mean(model_inferer_test, image, args.tta)

            seg_out = np.zeros((seg.shape[1], seg.shape[2], seg.shape[3]))
            seg_out[seg[1] == 1] = 2
            seg_out[seg[0] == 1] = 1
            seg_out[seg[2] == 1] = 3
            nib.save(nib.Nifti1Image(seg_out.astype(np.uint8), affine),
                     os.path.join(output_directory, img_name))
    print("Finished inference!")


if __name__ == '__main__':
    main()
