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


def bethesda_to_2_class_idx(name):
    if name == 'Negative for intraepithelial lesion':
            return 1
    else:
            return 0
    

def process_inference(device, model, modelc, cv_image):
    im0 = cv_image.copy()
    dets = torch.tensor([])

    # Check if cv_image resolution is bigger than 640 x 640
    if cv_image.shape[0] > 640 or cv_image.shape[1] > 640:
        patches = extract_overlapped_patches(0.1, cv_image)
        c = 0
        # Iterate over dict patches
        for pcenterxy, pimage in patches.items():
            # Detect cells
            dets = torch.cat((dets, detect_from_patch.detect(pimage, model, 
                                    device, patch_center=pcenterxy,
                                    glob_img=cv_image)), 0)

    # Apply NMS to remove duplicate deteccions from cells in overlapped areas
    dets = non_max_suppression_patches(dets, 0.2)
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


filename = "E:\\MLPathologyProject\\pap\\CRIC\\classifications.json"
f = open (filename, "r")
df = pd.read_json(f)

#test_path = "E:\\MLPathologyProject\\pap\\CRIC\\base_test"
test_path = "E:\\MLPathologyProject\\pap\\CRIC\\muestras\\test_originales"

yolo_weights = 'cellv1.pt' # Yolo weights path
clasif_model = 'efficientnetb0' # Classification model
cweights = "E:\\MLPathologyProject\\pap\\CRIC\\result\\" \
    "clasificacion_efficientnetb0_2_clases\\model.pt" # Classification model weights path

device = select_device('cpu')

model = attempt_load(yolo_weights, map_location=device)  # load FP32 model

# Load classifier
modelc = build_model(model=clasif_model, num_classes=2) # usando misma funcion que usamos para entrenar los modelos
modelc.load_state_dict(torch.load(cweights, map_location=device))
modelc.to(device).eval()

altered_tp, altered_fp = 0, 0

metrics =	{
    "altered_tp": 0,
    "altered_fp": 0,
    "altered_tn": 0,
    "altered_fn": 0,
    "normal_tp": 0,
    "normal_fp": 0,
    "normal_tn": 0,
    "normal_fn": 0
}

# Iterate over test images
for image_name in os.listdir(test_path):
    labels = torch.tensor([])
    cells = df.loc[df['image_name'] == image_name]['classifications'].values[0]
    for cell in cells:
        # Top left coordinates of the cell
        x1 = float(cell['nucleus_x']) - 45
        y1 = float(cell['nucleus_y']) - 45
        # Bottom right coordinates of the cell
        x2 = float(cell['nucleus_x']) + 45
        y2 = float(cell['nucleus_y']) + 45
        conficence = 1
        class_idx = bethesda_to_2_class_idx(cell['bethesda_system'])

        label = torch.tensor([[x1, y1, x2, y2, conficence, class_idx]])
        labels = torch.cat((labels, label), 0)
    
    dets = process_inference(device, model, modelc, cv2.imread(os.path.join(test_path, image_name)))
    
    # Compare detections with labels
    metrics = max_label_detection(metrics, dets, labels, 0.65)
    
print("Altered precision: ", metrics['altered_tp'] / (metrics['altered_tp'] + metrics['altered_fp']))
print("Altered recall: ", metrics['altered_tp'] / (metrics['altered_tp'] + metrics['altered_fn']))
print("Normal precision: ", metrics['normal_tp'] / (metrics['normal_tp'] + metrics['normal_fp']))
print("Normal recall: ", metrics['normal_tp'] / (metrics['normal_tp'] + metrics['normal_fn']))
