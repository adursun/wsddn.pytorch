import chainercv.transforms as T
import numpy as np
import torch


def unique_boxes(boxes, scale=1.0):
    """Returns indices of unique boxes."""
    v = np.array([1, 1e3, 1e6, 1e9])
    hashes = np.round(boxes * scale).dot(v)
    _, index = np.unique(hashes, return_index=True)
    return np.sort(index)


def filter_small_boxes(boxes, min_size):
    """Filters out small boxes."""
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]
    keep = np.where((w >= min_size) & (h > min_size))[0]
    return keep


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
