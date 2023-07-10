import argparse
import time
from pathlib import Path
import os

# ADDING THESE
import torch
import torch.nn as nn
import logging
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
from torchvision import models
from classification.train_v3.model import build_model

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_second_stage_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel


# Please add mean and standard deviation of 2nd stage classifier's dataset
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

class_names2 = None
super_class = None                  # The class whose output is not supposed to go to image classifier

def create_model(n_classes,device):
    model = torch.hub.load('pytorch/vision:v0.6.0','resnext50_32x4d',pretrained=True)
    n_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(n_features, n_classes),
        nn.Softmax(dim=1)
    )
    return model.to(device)


def detect(save_img=False):
    frame_number = 0
    source, weights, view_img, save_txt, imgsz, trace, second_classifier = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace, opt.second_classifier
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
        model = TracedModel(model, device, opt.img_size)

    if half:
        model.half()  # to FP16

    class_names1 = model.module.names if hasattr(model, 'module') else model.names
    print(class_names1)

    # Second-stage classifier
    if second_classifier:
        # TODO: Provide the list of class names for 2nd model
        class_names2 = ['Normal', 'Altered']

        print(class_names2)

        # If YOLO is trained for more than one class, then we mention which class is sub_class
        if len(class_names1)>1:
            super_class = 1                                 # Index of class that does not have sub-classes, currently for 2 classes
                                                            # it can be made a list for multiple superclasses
            sub_class = 0                                   # Index of class that has sub-classes that image classifier will classify

        # If YOLO is trained for just one class of objects, which is to be further classified by image classifier            
        else:
            sub_class=0
            super_class = None

        modelc = build_model(num_classes=len(class_names2))
        # TODO: Provide the path to load the pre-trained image classifier model .pt file below
        checkpoint = torch.load('D:\\PatoUTN\\Entrenamientos\\effnetb0_2_clases.pt', map_location='cpu')
        modelc.load_state_dict(checkpoint)
        modelc.to(device)
        modelc.to(device).eval()  

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    if second_classifier:
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in class_names2]
    else:
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in class_names1]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        frame_number+=1
        original_image = img
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=opt.augment)[0]

        # Inference
        t1 = time_synchronized()
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=opt.augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t3 = time_synchronized()

        # Apply 2nd stage classifier
        if second_classifier:
            if pred[0].nelement()>0:
                print('PREDS ANTES DE 2COND STAGE')
                print(pred)
                pred = apply_second_stage_classifier(pred, modelc, original_image, device, mean, std)
                print('PRED DESPUES')
                print(pred)
        all_label = ""
        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class                    
                    # s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for elements in reversed(det):
                    print(elements)
                    print('DETECCIONES')
                    print(det)
                    xyxy,conf,*cls = elements[:4],elements[4],elements[5:]
                    print('CLASES', cls)

                    if save_txt:
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(f"Frame number {frame_number}  \n XYXY {xyxy}, conf {conf}, cls {cls}")

                    if save_img or view_img:  # Add bbox to image
                        # If there was no sub_class detection
                        if len(cls[0])==1:
                            label = f'{class_names1[int(cls[0].item())]} {conf:.2f}'
                            plot_one_box(xyxy, im0, label=label, color=colors[int(cls[0].item())], line_thickness=3)
                            all_label+=label+","

                        # else, we find super class and sub class
                        else:
                            all_classes = cls[0].tolist()
                            print('CLASES', all_classes, 'SUPER', super_class, 'SUBCLASS', sub_class)
                            
                            # In general.py, we appended a column to every detection 'det' tensor
                            # If super class exists in the tensor, then we just show its class name
                            if super_class and super_class in all_classes:
                                # name_ind = int(sub_classes[sub_class])
                                label = f'{class_names1[super_class]} {conf:.2f}'

                            # If subclass existed in detection 'det' tensor and superclass didnt
                            # then we show name of subclass
                            else:
                                name_ind = int(all_classes[-1])         # Last element is the sub_class from image classifier
                                label = f'{class_names2[name_ind]} {conf:.2f}'

                            all_label+=label+","
                            plot_one_box(xyxy, im0, label=label, color=colors[name_ind], line_thickness=3)

            # Print time (inference + NMS)
            print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                    print(f" The image with the result is saved in: {save_path}")
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        #print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    parser.add_argument('--second-classifier', action='store_true', help='Apply 2nd stage classifier on detected items')
    opt = parser.parse_args()
    print(opt)
    #check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov7.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
