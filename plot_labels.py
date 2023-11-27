
from utils.plots import plot_one_box
import torch
from torchvision import transforms
import pandas as pd
import os
import cv2
import random


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


filename = "/shared/PatoUTN/PAP/Datasets/original/CRIC_classifications.json"
f = open (filename, "r")
df = pd.read_json(f)

src_path = "/shared/PatoUTN/PAP/Datasets/borrar"
dest_path = "/shared/PatoUTN/PAP/Datasets/paper"
#Get a list of 6 random colors for the bounding boxes
colors_labels = [[0, 0, 255], [ 88, 214, 33 ]]

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
        class_idx = bethesda_to_2_class_idx(cell['bethesda_system'])
        class_name = bethesda_to_2_class(cell['bethesda_system'])
        plot_one_box((x1,y1,x2,y2), cv_image, label=class_name, color=colors_labels[class_idx], line_thickness=2)
    cv2.imwrite(os.path.join(dest_path, image_name), cv_image)