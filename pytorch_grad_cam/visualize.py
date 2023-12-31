DIMENSION = "2d"
import matplotlib.pyplot as plt
import numpy as np
import cv2


# Visualization methods
def show_image(im, TITLE='', AX=None):
    if AX is None:
        plt.figure()
    plt.axis('off')
    plt.imshow(im)
    plt.title(TITLE)


def show_gray_image(im, TITLE='', AX=None):
    if AX is None:
        plt.figure()
    plt.axis('off')
    plt.imshow(im, cmap=plt.cm.gray, vmin=0, vmax=1)
    plt.title(TITLE)


def show_heatmap(im, TITLE='', AX=None, CMAP="inferno"):
    if AX is None:
        plt.figure()
    plt.axis('off')
    plt.imshow(im, cmap=CMAP)
    plt.title(TITLE)


def visualize_tensor(large_image, PERCENTILE=99):
    r"""Returns a 3D tensor as a grayscale 2D tensor.
    This method sums a 3D tensor across the absolute value of axis=-1, and then
    clips values at a given percentile.
    """
    new_image = np.sum(np.abs(large_image), axis=-1)

    vmax = np.percentile(new_image, PERCENTILE)
    vmin = np.min(new_image)

    return np.clip((new_image - vmin) / (vmax - vmin), 0, 1)


def visualize_tensor_negatives(large_image, PERCENTILE=99):
    r"""Returns a 3D tensor as a 2D tensor with positive and negative values.
    """
    new_image = np.sum(large_image, axis=-1)

    span = abs(np.percentile(new_image, PERCENTILE))
    vmin = -span
    vmax = span

    return np.clip((new_image - vmin) / (vmax - vmin), -1, 1)


def overlay_gradcam(img, cam3):
    img = np.uint8(255 * norm_image(img))
    cam3 = np.uint8(255 * cam3)

    cam3 = cv2.applyColorMap(cam3, cv2.COLORMAP_JET)
    cam3 = cv2.cvtColor(cam3, cv2.COLOR_BGR2RGB)

    img = img[..., np.newaxis]
    mask = np.ones((img.shape[0], img.shape[1], img.shape[2]))
    mask[img == 0] = 0
    new_img = 0.3 * cam3 * mask + 0.5 * img
    return (new_img * 255.0 / new_img.max()).astype("uint8")


def overlay_grad(img, grad, DIM=DIMENSION):
    img = np.uint8(255 * norm_image(img))
    grad = np.uint8(255 * grad)

    grad = cv2.applyColorMap(grad, cv2.COLORMAP_BONE)  # cv2.COLORMAP_JET
    grad = cv2.cvtColor(grad, cv2.COLOR_BGR2RGB)

    if DIM == "3d":
        img = img[..., np.newaxis]

    new_img = 0.5 * grad + 0.3 * img
    return (new_img * 255.0 / new_img.max()).astype("uint8")


def overlay_pred(img, pred, DIM=DIMENSION):
    if DIM == "3d":
        img = img[..., np.newaxis]

    new_img = 0.3 * pred + img
    return (new_img * 255.0 / new_img.max()).astype("uint8")


def norm_image(img, NORM_TYP="norm"):
    if NORM_TYP == "standard_norm":  # standarization, same dataset
        img_mean = img.mean()
        img_std = img.std()
        img_std = 1 if img.std() == 0 else img.std()
        img = (img - img_mean) / img_std
    elif NORM_TYP == "norm":  # different datasets
        img = (img - np.min(img)) / (np.ptp(img))  # (np.max(img) - np.min(img))
    elif NORM_TYP == "norm_slow":  # different datasets
        img_ptp = 1 if np.ptp(img) == 0 else np.ptp(img)
        img = (img - np.min(img)) / img_ptp
    return img
