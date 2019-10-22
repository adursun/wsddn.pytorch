import chainercv.transforms as T
import numpy as np
import torch
from chainercv.evaluations import eval_detection_voc
from PIL import Image
from torchvision import transforms
from torchvision.ops import nms, roi_pool

# this is duplicate
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


TRANSFORMS = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def prepare(img, boxes, max_dim=None, xflip=False, gt_boxes=None):
    img = np.asarray(img, dtype=np.float32)  # use numpy array for augmentation
    img = np.transpose(img, (2, 0, 1))  # convert img to CHW

    # convert boxes into (ymin, xmin, ymax, xmax) format
    boxes = swap_axes(boxes)

    if gt_boxes is not None:
        gt_boxes = swap_axes(gt_boxes)

    # scale
    if max_dim:
        img, boxes, gt_boxes = scale(img, boxes, max_dim, gt_boxes)

    # horizontal flip
    if xflip:
        img, boxes, gt_boxes = hflip(img, boxes, gt_boxes)

    # convert boxes back to (xmin, ymin, xmax, ymax) format
    boxes = swap_axes(boxes)

    if gt_boxes is not None:
        gt_boxes = swap_axes(gt_boxes)

    # convert img from CHW to HWC
    img = Image.fromarray(np.transpose(img, (1, 2, 0)).astype(np.uint8), mode="RGB")
    img = TRANSFORMS(img)  # convert pillow image to normalized tensor

    return img, boxes, gt_boxes


def evaluate(net, scales, dataloader):
    """Evaluates network."""
    with torch.no_grad():
        net.eval()

        aps = []
        maps = []

        for max_dim in scales:

            for xflip in [True, False]:
                total_pred_boxes = []
                total_pred_scores = []
                total_pred_labels = []
                total_gt_boxes = []
                total_gt_labels = []

                for (img, boxes, scores, gt_boxes, gt_labels) in dataloader:
                    boxes, scores, gt_boxes, gt_labels = (
                        boxes.numpy(),
                        scores.numpy(),
                        gt_boxes.numpy(),
                        gt_labels.numpy(),
                    )

                    keep = unique_boxes(boxes)
                    boxes = boxes[keep, :]

                    keep = filter_small_boxes(boxes, 2)
                    boxes = boxes[keep, :]

                    p_img, p_boxes, p_gt_boxes = prepare(
                        img, boxes, max_dim, xflip, gt_boxes
                    )

                    batch_imgs, batch_boxes, batch_scores, batch_gt_boxes, batch_gt_labels = (
                        np2gpu(p_img, DEVICE),
                        np2gpu(p_boxes, DEVICE),
                        np2gpu(scores, DEVICE),
                        np2gpu(p_gt_boxes, DEVICE),
                        np2gpu(gt_labels, DEVICE),
                    )
                    combined_scores, pred_boxes = net(
                        batch_imgs, batch_boxes, batch_scores
                    )
                    # pred_scores, pred_labels = torch.max(combined_scores, dim=1)

                    img_thresh = torch.sort(combined_scores.view(-1), descending=True)[
                        0
                    ][300]

                    batch_pred_boxes = []
                    batch_pred_scores = []
                    batch_pred_labels = []

                    for i in range(20):
                        region_scores = combined_scores[:, i]
                        filtered_indices = region_scores > img_thresh

                        filtered_region_scores = region_scores[filtered_indices]
                        filtered_pred_boxes = pred_boxes[filtered_indices]

                        selected_indices = nms(
                            filtered_pred_boxes, filtered_region_scores, 0.4
                        )

                        batch_pred_boxes.append(
                            filtered_pred_boxes[selected_indices].cpu().numpy()
                        )
                        batch_pred_scores.append(
                            filtered_region_scores[selected_indices].cpu().numpy()
                        )
                        batch_pred_labels.append(
                            np.full(len(selected_indices), i, dtype=np.int32)
                        )

                    total_pred_boxes.append(np.concatenate(batch_pred_boxes, axis=0))
                    total_pred_scores.append(np.concatenate(batch_pred_scores, axis=0))
                    total_pred_labels.append(np.concatenate(batch_pred_labels, axis=0))
                    total_gt_boxes.append(batch_gt_boxes[0].cpu().numpy())
                    total_gt_labels.append(batch_gt_labels[0].cpu().numpy())

                result = eval_detection_voc(
                    total_pred_boxes,
                    total_pred_labels,
                    total_pred_scores,
                    total_gt_boxes,
                    total_gt_labels,
                    iou_thresh=0.5,
                    use_07_metric=True,
                )
                aps.append(result["ap"])
                maps.append(result["map"])

        aps = np.stack(aps)
        maps = np.array(maps)

        print("Avg ap:", np.mean(aps, axis=0))
        print("Avg map:", np.mean(maps))

        net.train()


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
