import detect_from_patch as detect_from_patch
from utils.plots import plot_one_box
from utils.general import extract_overlapped_patches, \
    non_max_suppression_patches, apply_classifier, max_label_detection
from classification.train_v3.model import build_model
import torch
from torchvision import transforms
from utils.torch_utils import select_device
from models.experimental import attempt_load
import pandas as pd
import os
import cv2
import numpy as np

from pathlib import Path


def bethesda_to_2_class_idx(name):
    if name == 'Negative for intraepithelial lesion':
            return 1
    else:
            return 0
    

  
def bethesda_to_2_class(name):
    if name == 'Negative for intraepithelial lesion':
            return 'Normal'
    else:
            return 'Altered'
    

def process_inference(device, model, modelc, cv_image):
    im0 = cv_image.copy()
    dets = np.empty((1,6))

    # Check if cv_image resolution is bigger than 640 x 640
    if cv_image.shape[0] > 640 or cv_image.shape[1] > 640:
        patches = extract_overlapped_patches(0.1, cv_image)
        c = 0
        # Iterate over dict patches
        for pcenterxy, pimage in patches.items():
            # Detect cells
            det = detect_from_patch.detect(pimage, model, 
                                    device, patch_center=pcenterxy,
                                    glob_img=cv_image).detach().cpu().numpy()

            dets = np.concatenate((dets, det), axis=0)

    # Apply NMS to remove duplicate deteccions from cells in overlapped areas
    dets = non_max_suppression_patches(dets, 0.2)
    dets = dets.to(device)
    dets = [dets]
    #dets = nms.tolist()

    transform = transforms.ToTensor()
    img = transform(cv_image)

    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    
    # Classify detected cells

    dets = apply_classifier(dets, modelc, img, im0)[0]

    return dets


filename = "/shared/PatoUTN/PAP/Datasets/original/CRIC_classifications.json"
f = open (filename, "r")
df = pd.read_json(f)

#test_path = "E:\\MLPathologyProject\\pap\\CRIC\\base_test"
test_path = "/shared/PatoUTN/PAP/Datasets/originales/6/particionado/test"
plotted_path = "/shared/PatoUTN/PAP/Datasets/test_2_step/Effnetb0"
# Create plotted path dir if it doesn't exist
# if not os.path.exists(plotted_path):
#    os.makedirs(plotted_path)       

yolo_weights = 'runs/train/640_1class_NoAugm_1/weights/best.pt' # Yolo weights path
clasif_model = 'efficientnetb0' # Classification model
cweights = "efnetb0v1.pt" # Classification model weights path

device = select_device("0")

model = attempt_load(yolo_weights, map_location='cuda:0')  # load FP32 model

# Load classifier
modelc = build_model(model=clasif_model, num_classes=2) # usando misma funcion que usamos para entrenar los modelos
modelc.load_state_dict(torch.load(cweights, map_location='cuda:0'))
modelc.to(device)
modelc.eval()

altered_tp, altered_fp = 0, 0

# Make an array containing the RGB color blue and light green
colors_labels = [[0, 0, 255], [ 88, 214, 33 ]]
# Blue >> Altered
# Light green >> Normal

# Make an array containing the RGB color pink and yellow
colors_dets = [[255, 0, 255], [ 241, 209, 24 ]]

metrics =	{
    "altered_tp": 0,
    "altered_fp": 0,
    "altered_tn": 0,
    "altered_fn": 0,
    "normal_tp": 0,
    "normal_fp": 0,
    "normal_tn": 0,
    "normal_fn": 0,
    "normal_gt": 0,
    "altered_gt": 0,
    'normal_d': 0,
    'altered_d': 0
}
cant = 0
# Iterate over test images
for image_name in os.listdir(test_path):
    labels = []
    cells = df.loc[df['image_name'] == image_name]['classifications'].values[0]
    cv_image = cv2.imread(os.path.join(test_path, image_name))
    dets = process_inference(device, model, modelc, cv_image)

    for det in dets:
         det = det.cpu().numpy()
         class_name = 'Altered' if det[5] == 0 else 'Normal'
         class_idx = int(det[5])
         print(det)
         plot_one_box((det[0],det[1],det[2],det[3]), cv_image, label=class_name + ' DET', color=colors_dets[class_idx], line_thickness=1)

    for cell in cells:
        # Top left coordinates of the cell
        x1 = float(cell['nucleus_x']) - 45
        y1 = float(cell['nucleus_y']) - 45
        # Bottom right coordinates of the cell
        x2 = float(cell['nucleus_x']) + 45
        y2 = float(cell['nucleus_y']) + 45
        conficence = 1
        class_idx = bethesda_to_2_class_idx(cell['bethesda_system'])
        class_name = bethesda_to_2_class(cell['bethesda_system'])
        plot_one_box((x1,y1,x2,y2), cv_image, label=class_name, color=colors_labels[class_idx], line_thickness=1)

        label = [x1, y1, x2, y2, conficence, class_idx]
        labels.append(label)
        
    cant += len(cells)

    # Plotear imagenes
    # cv2.imwrite(os.path.join(plotted_path, image_name), cv_image)
    
    # Compare detections with labels
    metrics = max_label_detection(metrics, dets, labels, 0.4)

altered_precision = metrics['altered_tp'] / (metrics['altered_tp'] + metrics['altered_fp'])
altered_recall = metrics['altered_tp'] / (metrics['altered_tp'] + metrics['altered_fn'])

altered_precision_n = metrics['altered_tp'] / (metrics['altered_d'])
altered_recall_n = metrics['altered_tp'] / (metrics['altered_gt'])

normal_precision = metrics['normal_tp'] / (metrics['normal_tp'] + metrics['normal_fp'])
normal_recall = metrics['normal_tp'] / (metrics['normal_tp'] + metrics['normal_fn'])

normal_precision_n = metrics['normal_tp'] / (metrics['normal_d'])
normal_recall_n = metrics['normal_tp'] / (metrics['normal_gt'])

precision = (altered_precision + normal_precision) / 2
recall = (altered_recall + normal_recall) / 2
    
"""print("Altered precision: ", altered_precision)
print("Altered precision gth: ", altered_precision_n)
print("Altered recall: ", altered_recall)
print('Altered recall gth', altered_recall_n)
print("Normal precision: ", normal_precision)
print("Normal precision gth: ", normal_precision_n)

print("Normal recall: ", normal_recall)
print('Normal recall gth', normal_recall_n)
print("Precision: ", precision)
print("Recall: ", recall)
print("LABELS", metrics["altered_gt"] + metrics["normal_gt"])
print('Cant', cant)"""
print('Normal TP: ', metrics['normal_tp'], '     Normal FP: ', metrics['normal_fp'])
print('Altered FP: ', metrics['altered_fp'], '     Altered TP: ', metrics['altered_tp'])
