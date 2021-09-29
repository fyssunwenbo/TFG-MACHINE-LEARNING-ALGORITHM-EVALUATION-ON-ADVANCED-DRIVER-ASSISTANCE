from mmdet.apis import init_detector, inference_detector
import os
import numpy as np
import mmcv
import torch
import cv2

CLASS_NAME=["cyclist", "person"]

def show_result(
                img,
                result,
                score_thr=0.3,
                show=False,
                out_file=None):

    img = mmcv.imread(img)
    img = img.copy()
    if isinstance(result, tuple):
        bbox_result, segm_result = result
        if isinstance(segm_result, tuple):
            segm_result = segm_result[0]  # ms rcnn
    else:
        bbox_result, segm_result = result, None
    bboxes = np.vstack(bbox_result)
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(bbox_result)
    ]
    labels = np.concatenate(labels)
    # draw segmentation masks
    if segm_result is not None and len(labels) > 0:  # non empty
        segms = mmcv.concat_list(segm_result)
        inds = np.where(bboxes[:, -1] > score_thr)[0]
        np.random.seed(42)
        color_masks = [
            np.random.randint(0, 256, (1, 3), dtype=np.uint8)
            for _ in range(max(labels) + 1)
        ]
        for i in inds:
            i = int(i)
            color_mask = color_masks[labels[i]]
            sg = segms[i]
            if isinstance(sg, torch.Tensor):
                sg = sg.detach().cpu().numpy()
            mask = sg.astype(bool)
            img[mask] = img[mask] * 0.5 + color_mask * 0.5
    # if out_file specified, do not show image in window
    if out_file is not None:
        show = False
    # draw bounding boxes
    imshow_det_bboxes(img,bboxes,labels,CLASS_NAME,0.3,show=show,out_file=out_file)

    if not (show or out_file):
        return img

def imshow_det_bboxes(img,
                      bboxes,
                      labels,
                      class_names=None,
                      score_thr=0,
                      thickness=1,
                      font_scale=0.5,
                      show=False,
                      win_name='image',
                      wait_time=0,
                      out_file=None):
    assert bboxes.ndim == 2
    assert labels.ndim == 1
    assert bboxes.shape[0] == labels.shape[0]
    assert bboxes.shape[1] == 4 or bboxes.shape[1] == 5

    if score_thr > 0:
        assert bboxes.shape[1] == 5
        scores = bboxes[:, -1]
        inds = scores > score_thr
        bboxes = bboxes[inds, :]
        labels = labels[inds]

    img = np.ascontiguousarray(img)
    for bbox, label in zip(bboxes, labels):
        if label==0:
            bbox_color=(0,0,255)
            text_color=(0,0,255)
        else:
            bbox_color = (0, 255, 0)
            text_color = (0, 255, 0)
        bbox_int = bbox.astype(np.int32)
        left_top = (bbox_int[0], bbox_int[1])
        right_bottom = (bbox_int[2], bbox_int[3])
        cv2.rectangle(
            img, left_top, right_bottom, bbox_color, thickness=thickness)
        label_text = class_names[
            label] if class_names is not None else f'cls {label}'
        if len(bbox) > 4:
            label_text += f'|{bbox[-1]:.02f}'
        cv2.putText(img, label_text, (bbox_int[0], bbox_int[1] - 2),
                    cv2.FONT_HERSHEY_COMPLEX, font_scale, text_color)

    if show:
        cv2.imshow(win_name,img)
        cv2.waitKey(40)
    if out_file is not None:
        cv2.imwrite(out_file,img)
    return img

if __name__ =='__main__':
    config_file = 'work_dirs/faster_rcnn_r50_fpn.py'
    checkpoint_file = 'outputs/epoch_24.pth'

    model = init_detector(config_file, checkpoint_file, device='cuda:0')
    base_path = 'data/test'
    imgs = os.listdir(base_path)
    for name in imgs:
        img = os.path.join(base_path, name)
        print(img)
        
        result = inference_detector(model, img)
        # 保存结果
        show_result(img, result, out_file='data/results/{}.jpg'.format(name[:-4] + "_result"))