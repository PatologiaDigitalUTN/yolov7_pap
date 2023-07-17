import cv2
import detect_from_patch as detect_from_patch
import numpy as np
from utils.plots import plot_one_box
from utils.general import extract_overlapped_patches, \
    non_max_suppression_patches, apply_classifier, non_max_suppression
from classification.train_v3.model import build_model
import torch
from torchvision import transforms
from utils.torch_utils import select_device
import random


def process_inference(cv_image):
    im0 = cv_image.copy()
    dets = torch.tensor([])

    device = select_device('cpu')

    # Check if cv_image resolution is bigger than 640 x 640
    if cv_image.shape[0] > 640 or cv_image.shape[1] > 640:
        patches = extract_overlapped_patches(0.1, cv_image)
        c = 0
        # Iterate over dict patches
        for pcenterxy, pimage in patches.items():
            # Detect cells
            dets = torch.cat((dets, detect_from_patch.detect(pimage, 'cellv1.pt',
                                    patch_center=pcenterxy, glob_img=cv_image)), 0)
            if(c == 0):
                print('Primer dets')
                print(dets)
            c += 1

    # Apply NMS to remove duplicate deteccions from cells in overlapped areas
    dets = non_max_suppression_patches(dets, 0.2)
    dets = [dets]
    #dets = nms.tolist()
    
    # Load classifier
    modelc = build_model(model='efficientnetb0', num_classes=2) # usando misma funcion que usamos para entrenar los modelos
    modelc.load_state_dict(torch.load("E:\\MLPathologyProject\\pap\\CRIC\\result\\clasificacion_efficientnetb0_2_clases\\model.pt", map_location=device))
    modelc.to(device).eval()

    transform = transforms.ToTensor()
    img = transform(cv_image)

    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    
    # Classify detected cells
    dets = apply_classifier(dets, modelc, img, im0)[0]

    names = ['Altered', 'Normal']
    colors = ([255, 0, 255], [255, 0, 0]) # BGR Altered = Magenta, Normal = Blue
    #colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Write results
    for *xyxy, conf, cls in reversed(dets):
        label = f'{names[int(cls)]} {conf:.2f}'
        plot_one_box(xyxy, cv_image, label=label, color=colors[int(cls)], line_thickness=1)


    # Save image
    cv2.imwrite("E:\\MLPathologyProject\\pap\\CRIC\\yolo_modelos\\probando.png", cv_image)