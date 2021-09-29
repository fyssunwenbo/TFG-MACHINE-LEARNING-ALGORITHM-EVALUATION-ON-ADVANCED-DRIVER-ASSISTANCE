import glob
import os
import time

import torch
from PIL import Image
from vizer.draw import draw_boxes

from ssd.config import cfg
from ssd.data.datasets import COCODataset, VOCDataset, CustomerDataSet
import argparse
import numpy as np
from timeit import default_timer as timer
from ssd.data.transforms import build_transforms
from ssd.modeling.detector import build_detection_model
from ssd.utils import mkdir
from ssd.utils.checkpoint import CheckPointer
import cv2

def Video_detection(cfg, ckpt,score_threshold, video_path, dataset_type, output_dir):

    #os.makedirs(output_dir,exist_ok=True)
    vid = cv2.VideoCapture(video_path)
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")
    video_FourCC = int(vid.get(cv2.CAP_PROP_FOURCC))
    video_fps = vid.get(cv2.CAP_PROP_FPS)
    video_size = (
        int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    # video_size = (512,512)
    isOutput = True if output_dir != "" else False
    if isOutput:
        print(
            "!!! TYPE:",
            type(output_dir),
            type(video_FourCC),
            type(video_fps),
            type(video_size),
        )
        out = cv2.VideoWriter(
            output_dir, video_FourCC, video_fps, video_size
        )
    accum_time = 0
    curr_fps = 0
    fps = "FPS: ??"
    prev_time = timer()
    if dataset_type == "voc":
        class_names = VOCDataset.class_names
    elif dataset_type == 'coco':
        class_names = COCODataset.class_names
    elif dataset_type == 'customer':
        class_names = CustomerDataSet.class_names
    else:
        raise NotImplementedError('Not implemented now.')
    device = torch.device(cfg.MODEL.DEVICE)

    model = build_detection_model(cfg)
    model = model.to(device)
    checkpointer = CheckPointer(model, save_dir=cfg.OUTPUT_DIR)
    checkpointer.load(ckpt, use_latest=ckpt is None)
    weight_file = ckpt if ckpt else checkpointer.get_checkpoint_file()
    print('Loaded weights from {}'.format(weight_file))
    transforms = build_transforms(cfg, is_train=False)
    cpu_device = torch.device("cpu")
    model.eval()
    while True:
        return_value, frame = vid.read()
        W, H, _ = frame.shape
        frame =cv2.resize(frame,(512,512),frame, interpolation=cv2.INTER_LINEAR)
        width, height, _ = frame.shape
        image = transforms(frame)[0].unsqueeze(0)
        result = model(image.to(device))[0]
        result = result.resize((width, height))
        boxes, labels, scores = result['boxes'], result['labels'], result['scores']
        boxes = boxes.cpu().detach().numpy()
        labels = labels.cpu().detach().numpy()
        scores = scores.cpu().detach().numpy()
        indices = scores > score_threshold
        boxes = boxes[indices]
        labels = labels[indices]
        scores = scores[indices]
        frame = draw_boxes(frame, boxes, labels, scores, class_names).astype(np.uint8)
        curr_time = timer()
        exec_time = curr_time - prev_time
        prev_time = curr_time
        accum_time = accum_time + exec_time
        curr_fps = curr_fps + 1
       # if accum_time > 1:
           # accum_time = accum_time - 1
          #  fps = "FPS: " + str(curr_fps)
           # curr_fps = 0
        fps = "FPS: "+str(round(1/exec_time,1))
        cv2.putText(
            frame,
            text=fps,
            org=(3, 15),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.50,
            color=(255, 0, 0),
            thickness=2,
        )
        cv2.namedWindow("result", cv2.WINDOW_NORMAL)
        cv2.imshow("result", frame)
        if isOutput:
            frame = cv2.resize(frame, video_size, frame, interpolation=cv2.INTER_LINEAR)
            out.write(frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

def main():
    parser = argparse.ArgumentParser(description="SSD Demo.")
    parser.add_argument(
        "--config-file",
        default="./configs/vgg_ssd512_customer712.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument("--ckpt", type=str, default=None, help="Trained weights.")
    parser.add_argument("--score_threshold", type=float, default=0.1)
    parser.add_argument(
        "--video_path", type=str, default="bag.avi", help="video file path"
    )
    parser.add_argument("--output_dir", default='demo/video_result', type=str, help='Specify a image dir to save predicted images.')
    parser.add_argument("--dataset_type", default="customer", type=str, help='Specify dataset type. Currently support voc and coco.')

    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    print(args)

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    print("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        print(config_str)
    print("Running with config:\n{}".format(cfg))

    Video_detection(cfg=cfg,
             ckpt=args.ckpt,
             score_threshold=args.score_threshold,
             video_path=args.video_path,
             output_dir=args.output_dir,
             dataset_type=args.dataset_type)


if __name__ == '__main__':
    main()
