import detect_from_patch as detect_from_patch
from plots import plot_one_box
import torch
from torchvision import transforms
import pandas as pd
import os
import cv2
import random

filename = "E:\\MLPathologyProject\\pap\\CRIC\\classifications.json"
f = open (filename, "r")
df = pd.read_json(f)

src_path = "E:\\MLPathologyProject\\pap\\CRIC\\base_test"
dest_path = "E:\\MLPathologyProject\\pap\\CRIC\\base_test_labels"
#Get a list of 6 random colors for the bounding boxes
colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(6)]
class_idx = {'Negative for intraepithelial lesion': 0, 'ASC-H': 1, 
             'ASC-US': 2, 'HSIL': 3, 'LSIL': 4, 'SCC': 5}

for image_name in os.listdir(src_path):
    labels = torch.tensor([])
    # Load image
    image_path = os.path.join(src_path, image_name)
    cv_image = cv2.imread(image_path)
    cells = df.loc[df['image_name'] == image_name]['classifications'].values[0]
    for cell in cells:
        # Top left coordinates of the cell
        x1 = float(cell['nucleus_x']) - 45
        y1 = float(cell['nucleus_y']) - 45
        # Bottom right coordinates of the cell
        x2 = float(cell['nucleus_x']) + 45
        y2 = float(cell['nucleus_y']) + 45
        conficence = 1
        class_name = cell['bethesda_system']

        plot_one_box((x1,y1,x2,y2), cv_image, label=class_name, color=colors[class_idx[class_name]], line_thickness=1)
    cv2.imwrite(os.path.join(dest_path, image_name), cv_image)