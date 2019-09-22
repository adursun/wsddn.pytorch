import chainercv.transforms as T
import numpy as np


def hflip(img, boxes, gt_boxes=None):
    """Flips image and related boxes horizontally."""
    img = T.flip(img, y_flip=False, x_flip=True, copy=True)
    boxes = T.flip_bbox(boxes, img[0].shape, y_flip=False, x_flip=True)
    if gt_boxes is None:
        return img, boxes, None
    gt_boxes = T.flip_bbox(gt_boxes, img[0].shape, y_flip=False, x_flip=True)
    return img, boxes, gt_boxes


def scale(img, boxes, max_dim, gt_boxes=None):
    """Scales image and related boxes."""
    initial_size = img[0].shape
    scaled_img = T.scale(img, max_dim, fit_short=False)
    boxes = T.resize_bbox(boxes, initial_size, scaled_img[0].shape)
    if gt_boxes is None:
        return scaled_img, boxes, None
    gt_boxes = T.resize_bbox(gt_boxes, initial_size, scaled_img[0].shape)
    return scaled_img, boxes, gt_boxes


def swap_axes(boxes):
    """Swaps x and y axes."""
    boxes = boxes.copy()
    boxes = np.stack((boxes[:, 1], boxes[:, 0], boxes[:, 3], boxes[:, 2]), axis=1)
    return boxes


def np2gpu(arr, device):
    """Creates torch array from numpy one."""
    arr = np.expand_dims(arr, axis=0)
    return torch.from_numpy(arr).to(device)
