import os

import cv2
import numpy as np
import torch
import ttach as tta
from typing import Callable, List, Tuple

from matplotlib import pyplot as plt
from torch import autograd
from skimage.transform import resize
from pytorch_grad_cam.activations_and_gradients import ActivationsAndGradients
from pytorch_grad_cam.utils.svd_on_activations import get_2d_projection
from pytorch_grad_cam.utils.image import scale_cam_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.visualize import visualize_tensor, show_gray_image, overlay_gradcam, show_heatmap, overlay_grad, show_image, \
    overlay_pred, norm_image


class BaseCAM:
    def __init__(self,
                 model: torch.nn.Module,
                 target_layers: List[torch.nn.Module],
                 use_cuda: bool = False,
                 reshape_transform: Callable = None,
                 compute_input_gradient: bool = False,
                 uses_gradients: bool = True,
                 ) -> None:
        self.model = model.eval()
        self.target_layers = target_layers
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()
        self.reshape_transform = reshape_transform
        self.compute_input_gradient = compute_input_gradient
        self.uses_gradients = uses_gradients
        self.activations_and_grads = ActivationsAndGradients(
            self.model, target_layers, reshape_transform)

    """ Get a vector of weights for every channel in the target layer.
        Methods that return weights channels,
        will typically need to only implement this function. """

    def get_cam_weights(self,
                        input_tensor: torch.Tensor,
                        target_layers: List[torch.nn.Module],
                        targets: List[torch.nn.Module],
                        activations: torch.Tensor,
                        grads: torch.Tensor) -> np.ndarray:
        raise Exception("Not Implemented")

    def get_cam_image(self,
                      input_tensor: torch.Tensor,
                      target_layer: torch.nn.Module,
                      targets: List[torch.nn.Module],
                      activations: torch.Tensor,
                      grads: torch.Tensor,
                      eigen_smooth: bool = False) -> np.ndarray:

        weights = self.get_cam_weights(input_tensor,
                                       target_layer,
                                       targets,
                                       activations,
                                       grads)
        weighted_activations = weights[:, :, None, None, None] * activations
        if eigen_smooth:
            cam = get_2d_projection(weighted_activations)
        else:
            cam = weighted_activations.sum(axis=1)
        return cam

    def forward(self,
                input_tensor: torch.Tensor,
                targets: List[torch.nn.Module],
                eigen_smooth: bool = False,
                CLASS_IDs=None,
                SLICE_ID=77) -> np.ndarray:
        if CLASS_IDs is None:
            CLASS_IDs = [0, 1, 2]
        modality_dict = {"FLAIR": 0, "T1": 1, "T1CE": 2, "T2": 3}

        if self.cuda:
            input_tensor = input_tensor.cuda()  # input_tensor 134 172 142

        if self.compute_input_gradient:
            input_tensor = torch.autograd.Variable(input_tensor,
                                                   requires_grad=True)
        # conv_outputs, predictions = self.model(input_tensor)
        # loss = predictions[:, :, :, :, CLASS_IDs[0]]
        for param in self.target_layers[0].parameters():
            param.requires_grad_()
        # outputs = self.activations_and_grads(input_tensor)  # model output

        if self.uses_gradients:
            grads = self.get_grad_cam(input_tensor, 0)
            for c_id in CLASS_IDs[0:]:
                grads += self.get_grad_cam(input_tensor, c_id)
        heatmap = visualize_tensor(grads)
        im_orig = input_tensor[0, modality_dict["FLAIR"], :, :, SLICE_ID, ]  # (1, 192, 224, 160, 4)

        overlay = overlay_gradcam(im_orig.detach().numpy(), heatmap)

        return overlay

    def get_grad_cam(self, input_tensor, class_id, eps=1e-5):
        modality_dict = {"FLAIR": 0, "T1": 1, "T1CE": 2, "T2": 3}

        predictions = self.activations_and_grads(input_tensor)

        loss = predictions[:, class_id, :, :, :]
        conv_outputs = self.activations_and_grads.out
        output = conv_outputs[0]
        grads = autograd.grad(loss.sum(), conv_outputs)[0][0]

        norm_grads = torch.div(grads, torch.mean(torch.square(grads)) + eps)

        weights = torch.mean(norm_grads, dim=(1, 2, 3))

        # Average gradients spatially
        # Build a ponderated map of filters according to gradients importance
        cam = torch.sum(torch.multiply(weights, output.permute(1, 2, 3, 0)), dim=-1)

        # Apply ReLU
        grad_cam = np.maximum(cam.cpu().detach().numpy(), 0)
        # Resize heatmap to be the same size as the input
        if np.max(grad_cam) > 0:
            grad_cam = grad_cam / np.max(grad_cam)

        # Resize to the output layer's shape
        new_shape = input_tensor.shape[2:]
        grad_cam = resize(grad_cam, new_shape)
        self.model.zero_grad()
        for param in self.model.parameters(): param.grad = None
        return grad_cam

    def get_target_width_height(self,
                                input_tensor: torch.Tensor) -> Tuple[int, int, int]:
        width, height, Length = input_tensor.size(-1), input_tensor.size(-2), input_tensor.size(-3)
        return width, height, Length

    def compute_cam_per_layer(
            self,
            input_tensor: torch.Tensor,
            targets: List[torch.nn.Module],
            eigen_smooth: bool) -> np.ndarray:
        activations_list = [a.cpu().data.numpy()
                            for a in self.activations_and_grads.activations]
        grads_list = [g.cpu().data.numpy()
                      for g in self.activations_and_grads.gradients]
        target_size = self.get_target_width_height(input_tensor)  # input_tensor 134 172 142

        cam_per_target_layer = []
        # Loop over the saliency image from every layer
        for i in range(len(self.target_layers)):
            target_layer = self.target_layers[i]
            layer_activations = None
            layer_grads = None
            if i < len(activations_list):
                layer_activations = activations_list[i]
            if i < len(grads_list):
                layer_grads = grads_list[i]

            cam = self.get_cam_image(input_tensor,
                                     target_layer,
                                     targets,
                                     layer_activations,
                                     layer_grads,
                                     eigen_smooth)
            cam = np.maximum(cam, 0)
            scaled = scale_cam_image(cam, target_size)
            cam_per_target_layer.append(scaled[:, None, :])

        return cam_per_target_layer

    def aggregate_multi_layers(
            self,
            cam_per_target_layer: np.ndarray) -> np.ndarray:
        cam_per_target_layer = np.concatenate(cam_per_target_layer, axis=1)
        cam_per_target_layer = np.maximum(cam_per_target_layer, 0)
        result = np.mean(cam_per_target_layer, axis=1)
        return scale_cam_image(result)

    def forward_augmentation_smoothing(self,
                                       input_tensor: torch.Tensor,
                                       targets: List[torch.nn.Module],
                                       eigen_smooth: bool = False) -> np.ndarray:
        transforms = tta.Compose(
            [
                tta.HorizontalFlip(),
                tta.Multiply(factors=[0.9, 1, 1.1]),
            ]
        )
        cams = []
        for transform in transforms:
            augmented_tensor = transform.augment_image(input_tensor)
            cam = self.forward(augmented_tensor,
                               targets,
                               eigen_smooth)

            # The ttach library expects a tensor of size BxCxHxW
            cam = cam[:, None, :, :]
            cam = torch.from_numpy(cam)
            cam = transform.deaugment_mask(cam)

            # Back to numpy float32, HxW
            cam = cam.numpy()
            cam = cam[:, 0, :, :]
            cams.append(cam)

        cam = np.mean(np.float32(cams), axis=0)
        return cam

    def overlay_gradcam(self, img, cam3):
        img = np.uint8(255 * norm_image(img))
        cam3 = np.uint8(255 * cam3)

        cam3 = cv2.applyColorMap(cam3, cv2.COLORMAP_JET)
        cam3 = cv2.cvtColor(cam3, cv2.COLOR_BGR2RGB)

        img = img[..., np.newaxis]

        new_img = 0.3 * cam3 + 0.5 * img
        return (new_img * 255.0 / new_img.max()).astype("uint8")

    def __call__(self,
                 input_tensor: torch.Tensor,
                 targets: List[torch.nn.Module] = None,
                 aug_smooth: bool = False,
                 eigen_smooth: bool = False) -> np.ndarray:

        # Smooth the CAM result with test time augmentation
        if aug_smooth is True:
            return self.forward_augmentation_smoothing(
                input_tensor, targets, eigen_smooth)

        return self.forward(input_tensor,
                            targets, eigen_smooth)

    def __del__(self):
        self.activations_and_grads.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.activations_and_grads.release()
        if isinstance(exc_value, IndexError):
            # Handle IndexError here...
            print(
                f"An exception occurred in CAM with block: {exc_type}. Message: {exc_value}")
            return True
