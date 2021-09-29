import sys
from mmdet.apis import init_detector, inference_detector
import datetime
import os, cv2, json
import glob, math, time
import torch
from PIL import Image
from vizer.draw import draw_boxes
import warnings
import numpy as np
import torchvision
if torchvision.__version__ >= '0.3.0':
    _nms = torchvision.ops.nms
else:
    warnings.warn('No NMS is available. Please upgrade torchvision to 0.3.0+')
    sys.exit(-1)

def run_demo(config_file,checkpoint_file, score_threshold, images_dir, output_dir, ):
    model = init_detector(config_file, checkpoint_file, device='cuda:0')
    print('Loaded weights from {}'.format(checkpoint_file))
    res_img_dir = os.path.join(output_dir, 'res_img')
    os.makedirs(res_img_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    CLASSES = ('container_vessel', 'cruise_ship', 'sailboat',
               'sightseeing_boat', 'skiff', 'yacht')
    files = os.listdir(images_dir)
    s_time = datetime.datetime.now()
    for i, file in enumerate(files):
        #print('Total images:{},current:{}'.format(len(files), i))
        with open(os.path.join(output_dir,file[:-4]+".txt"), 'w') as f:
            image = cv2.imread(os.path.join(images_dir, file))[:, :, ::-1]  # RGB
            start = time.time()
            img_h, img_w, _ = image.shape
            load_time = time.time() - start
            start = time.time()
            result = inference_detector(model, image) #长度是6的list，每个位置表示一个类别。bboxshape=[n,5] 最后一位表示置信度
            bboxes = np.vstack(result)
            labels = [
                np.full(bbox.shape[0], i, dtype=np.int32)
                for i, bbox in enumerate(result)
            ]
            labels = np.concatenate(labels)
            scores = bboxes[:, -1]
            boxes = bboxes[:, :4]

            inference_time = time.time() - start
            if(labels.size==0):
                continue
            #boxes, labels, scores = result['boxes'], result['labels'], result['scores']

            indices = scores > score_threshold
            boxes = boxes[indices]
            labels = labels[indices]
            scores = scores[indices]
            # drawn_image = draw_boxes(image, boxes, labels, scores, class_names).astype(np.uint8)
            # Image.fromarray(drawn_image).save(os.path.join(res_img_dir, name+".jpg"))
            if labels.size == 0:
                continue


            meters = ' | '.join(
                [
                    'objects {:02d}'.format(len(boxes)),
                    'load {:03d}ms'.format(round(load_time * 1000)),
                    'inference {:03d}ms'.format(round(inference_time * 1000)),
                    'FPS {}'.format(round(1.0 / inference_time))
                ]
            )
            print('({:04d}/{:04d}) {}: {}'.format(i + 1, len(files), file, meters))
            for i, label in enumerate(labels):
                xmin, ymin, xmax, ymax = boxes[i]
                score = scores[i]
                name = CLASSES[label]
                bbox = name +" "+str(score)+" "+str(xmin)+" "+ str(ymin) +" "+ str(xmax)+" "+str(ymax)+"\n"
                f.write(bbox)

            # print(all_boxes.shape, all_labels.shape, all_scores.shape)
            # 绘制大图---------------
            drawn_image = draw_boxes(image, boxes, labels, scores, CLASSES).astype(np.uint8)
            Image.fromarray(drawn_image).save(os.path.join(res_img_dir, file[:-4] + ".jpg"))



def main():
    # Specify the path to model config and checkpoint file
    config_file = 'work_dirs/ship/faster_rcnn_r50_fpn.py'
    checkpoint_file = '/media/qingyuan/D/data/ubuntu_data/FasterRcnn_Ship/FasterRcnn_r50_FPN/latest.pth'
    #images_dir = 'data/ship_val/val/JPEGImages/'
    images_dir = '/media/qingyuan/D/data/ubuntu_data/FasterRcnn_Ship/test_ship/single/photo'
    output_dir = '/media/qingyuan/D/data/ubuntu_data/FasterRcnn_Ship/test_results/test_FPN_single/'
    run_demo(config_file = config_file,
             checkpoint_file =checkpoint_file,
             score_threshold=0.1,
             images_dir=images_dir,
             output_dir=output_dir,
          )


if __name__ == '__main__':
    main()