# -*- coding = utf-8 -*-
# @Time : 2023/7/26 21:33
# @Author：dianlong
import warnings

import monai
from monai.networks.nets import UNet,AttentionUnet

from models.student import UncertaintyTeacherKDForSequenceClassification
from utils.utils import load_images

warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
import torch

DIMENSION = "3d"
MODALITY = "FLAIR"
XAI_MODE = "segmentation"
CLASS_IDs = [0, 1, 2]
ID = "BraTS20_Training_362"
SLICE_ID = 50
XAI = "GCAM"
CLASS_ID = 2  # np.argmax(predictions[0])
TUMOR_LABEL = "all"  # for grad-CAM
# Segmentation model parameters
DATASET_PATH = r'./datasets'  # RT
IMG_SHAPE = (192, 224, 160)

io_imgs = load_images(ID, PATH_DATA=DATASET_PATH, DIM=IMG_SHAPE)
im_orig = io_imgs[:, :, :, SLICE_ID, 0]  # 2D FLAIR
io_imgs = monai.data.meta_tensor.MetaTensor(io_imgs, dtype=torch.float32).permute(0, 4, 1, 2, 3)
model_path = "./pretrained_models/model.pt"
# https://pytorch.org/vision/stable/auto_examples/plot_visualization_utils.html
is_KD =False
model = UNet(
    spatial_dims=3,
    in_channels=4,
    out_channels=3,
    channels=(16, 32),
    strides=[2],
    num_res_units=2
)
#
# model = AttentionUnet(
#     spatial_dims=3,
#     in_channels=4,
#     out_channels=3,
#     channels=(8, 16),
#     strides=[2],
# )
if is_KD:
    model = UncertaintyTeacherKDForSequenceClassification(
        student=model
    )
model_dict = torch.load(model_path)["state_dict"]
model.load_state_dict(model_dict)
if "KD" in model_path:
    model = model.student.eval()
# model = model.eval()
input_tensor = io_imgs
if torch.cuda.is_available():
    model = model  # .cuda()
    input_tensor = input_tensor  # .cuda()


class SegmentationModelOutputWrapper(torch.nn.Module):
    def __init__(self, model):
        super(SegmentationModelOutputWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)  # ["out"]



from pytorch_grad_cam import GradCAM


class SemanticSegmentationTarget:
    def __init__(self, category, mask):
        self.category = category
        self.mask = torch.from_numpy(mask)
        if torch.cuda.is_available():
            self.mask = self.mask  # .cuda()

    def __call__(self, model_output):
        return (model_output[self.category, :, :] * self.mask).sum()


# 获取需要可视化的层
target_layers = [model.model.__getitem__(-1).__getitem__(0)]
# car_mask_float保存的是car的分割图，只有0和1，SemanticSegmentationTarget则是为了获取
# targets = [SemanticSegmentationTarget(car_category, car_mask_float)]
with GradCAM(model=model,
             target_layers=target_layers,
             use_cuda=False) as cam:
    grayscale_cam = cam(input_tensor=input_tensor)
    # input_tensor = input_tensor / input_tensor.max()
    # cam_image = show_cam_on_image(input_tensor[0, 0, :, :, 80],
    #                               grayscale_cam[70, :, :], use_rgb=True)

# Image.fromarray(cam_image)

import matplotlib.pyplot as plt

plt.title(model_path)
plt.imshow(grayscale_cam)
plt.show()
