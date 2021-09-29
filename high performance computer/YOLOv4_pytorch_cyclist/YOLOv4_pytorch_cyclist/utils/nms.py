import sys
import warnings
import numpy as np
import torch
import torchvision
if torchvision.__version__ >= '0.3.0':
    _nms = torchvision.ops.nms
else:
    warnings.warn('No NMS is available. Please upgrade torchvision to 0.3.0+')
    sys.exit(-1)


def nms(boxes, scores, nms_thresh):
    """ Performs non-maximum suppression, run on GPU or CPU according to
    boxes's device.
    Args:
        boxes(Tensor[N, 4]): boxes in (x1, y1, x2, y2) format, use absolute coordinates(or relative coordinates)
        scores(Tensor[N]): scores
        nms_thresh(float): thresh
    Returns:
        indices kept.
    """
    keep = _nms(boxes, scores, nms_thresh)
    return keep


def batched_nms(boxes, scores, idxs, iou_threshold, type='nms'):
    """
    Performs non-maximum suppression in a batched fashion.

    Each index value correspond to a category, and NMS
    will not be applied between elements of different categories.

    Parameters
    ----------
    boxes : Tensor[N, 4]
        boxes where NMS will be performed. They
        are expected to be in (x1, y1, x2, y2) format
    scores : Tensor[N]
        scores for each one of the boxes
    idxs : Tensor[N]
        indices of the categories for each one of the boxes.
    iou_threshold : float
        discards all overlapping boxes
        with IoU < iou_threshold

    Returns
    -------
    keep : Tensor
        int64 tensor with the indices of
        the elements that have been kept by NMS, sorted
        in decreasing order of scores
    """
    if type == 'soft_nms':
        if boxes.numel() == 0:
            return torch.empty((0,), dtype=torch.int64, device=boxes.device)
        max_coordinate = boxes.max()
        offsets = idxs.to(boxes) * (max_coordinate + 1)
        boxes_for_nms = boxes + offsets[:, None]
        keep = soft_nms(boxes_for_nms, scores)
    else:
        if boxes.numel() == 0:
            return torch.empty((0,), dtype=torch.int64, device=boxes.device)
        # strategy: in order to perform NMS independently per class.
        # we add an offset to all the boxes. The offset is dependent
        # only on the class idx, and is large enough so that boxes
        # from different classes do not overlap
        max_coordinate = boxes.max()
        offsets = idxs.to(boxes) * (max_coordinate + 1)
        boxes_for_nms = boxes + offsets[:, None]
        keep = nms(boxes_for_nms, scores, iou_threshold)

    return keep

def soft_nms(boxes, sc, Nt=0.5, sigma=0.5, thresh=0.003, method=0):
    """
    Args:
        boxes: boxes 坐标矩阵 [N,4] [x1,y1,x2,y2]
        sc: 对应分数[N]
        Nt: iou交叠阈值
        sigma: gaussian函数的方差
        thresh: 最后分数阈值
        method: 使用的方法

    Returns:
            留下的boxes的index
    """
    #indexes concatenate boxes with the last column
    boxes = boxes.cpu().numpy()
    sc = sc.cpu().numpy()
    N = boxes.shape[0]
    indexes = np.array([np.arange(N)])
    boxes = np.concatenate((boxes, indexes.T),axis=1)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    scores = sc.copy()
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    for i in range(N):
        tBD = boxes[i, :].copy()
        tscore = scores[i].copy()
        tarea = areas[i].copy()
        pos = i + 1

        #
        if i != N-1:
            maxscore = np.max(scores[pos:], axis=0)
            maxpos = np.argmax(scores[pos:], axis=0)
        else:
            maxscore = scores[-1]
            maxpos = 0
        if tscore < maxscore:
            boxes[i, :] = boxes[maxpos + i + 1, :]
            boxes[maxpos + i +1, :] = tBD
            tBD = boxes[i, :]

            scores[i] = scores[maxpos + i + 1]
            scores[maxpos + i + 1] = tscore
            tscore = scores[i]

            areas[i] = areas[maxpos + i + 1]
            areas[maxpos + i + 1] = tarea
            tarea = areas[i]

        #IOu calculate
        yy1 = np.maximum(boxes[i, 1], boxes[pos:, 1])
        xx1 = np.maximum(boxes[i, 0], boxes[pos:, 0])
        xx2 = np.minimum(boxes[i, 2], boxes[pos:, 2])
        yy2 = np.minimum(boxes[i, 3], boxes[pos:, 3])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter/(areas[i] + areas[pos:] - inter)

        # Three methods : 1.Linear 2.gaussian 3.original NMS
        if method ==1: #linear
            weight = np.ones(ovr.shape)
            weight[ovr > Nt] = weight[ovr > Nt] - ovr[ovr > Nt]

        elif method ==2: #gaussian
            weight = np.exp(-(ovr * ovr)/sigma)
        else:
            weight = np.ones(ovr.shape)
            weight[ovr > Nt] = 0

        scores[pos:] = weight * scores[pos:]

        #select the boxes and keep the corresponding indexes
        inds = boxes[:, 4][scores > thresh]
        keep = inds.astype(int)
        return keep


