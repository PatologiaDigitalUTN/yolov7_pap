import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages, LoadImageFromOpencv
from utils.general import check_img_size, non_max_suppression, \
    scale_coords, set_logging, translate_coordinates
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel


def detect(img, weights, device ='cpu', imgsz=640, augment=True, conf_thres=0.25,
           iou_thres=0.45, patch_center=None, glob_img=None):
    # Initialize
    set_logging()
    device = select_device(device)

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    t0 = time.time()
    img, im0s = LoadImageFromOpencv(img) # problema acá
    img = torch.from_numpy(img).to(device)
    img = img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Warmup
    if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
        old_img_b = img.shape[0]
        old_img_h = img.shape[2]
        old_img_w = img.shape[3]
        for i in range(3):
            model(img, augment=augment)[0]

    # Inference
    t1 = time_synchronized()
    with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
        pred = model(img, augment=augment)[0]
    t2 = time_synchronized()

    # Apply NMS
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes=None, agnostic=True)
    t3 = time_synchronized()

    # Process detections
    for i, det in enumerate(pred):  # detections per image
        s, im0 = '', im0s 

        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
                s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

            # If the image to be detected is a patch, translate the coordinates
            # to the
            if patch_center is not None:
                det = translate_coordinates(patch_center, det, imgsz, imgsz)
                im0 = glob_img
            # Write results
            #for *xyxy, conf, cls in reversed(det):
            #    label = f'{names[int(cls)]} {conf:.2f}'
            #    plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)

        # Print time (inference + NMS)
        print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

        return det

        # Stream results
        # if view_img:
        #     cv2.imshow(str(p), im0)
        #     cv2.waitKey(1)  # 1 millisecond