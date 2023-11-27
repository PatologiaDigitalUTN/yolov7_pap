import pandas as pd
import os

json_path = "/shared/PatoUTN/PAP/Datasets/original/CRIC_classifications.json"
f = open (json_path, "r")
df = pd.read_json(f)

paths = ['/shared/PatoUTN/PAP/Datasets/originales/2/particionado_labeled/train',
         '/shared/PatoUTN/PAP/Datasets/originales/2/particionado_labeled/val',
         '/shared/PatoUTN/PAP/Datasets/originales/2/particionado_labeled/test']
img_width = 1376
img_height = 1020
bb_width = 90 # label's bounding box width
bb_height = 90

for path in paths:
    for image_name in os.listdir(path):
        cells = df.loc[df['image_name'] == image_name]['classifications'].values[0]

        file_name = os.path.join(path, f'{image_name.split(".")[0]}.txt')
        annotation_file = open(file_name, 'w')
        for cell in cells:
            if cell['bethesda_system'] == 'Negative for intraepithelial lesion':
                class_idx = 1
            else:
                class_idx = 0
            # class_idx = 0 # just 1 class
            
            line = f'{class_idx} {cell["nucleus_x"] / img_width} {cell["nucleus_y"] / img_height} {bb_width / img_width} {bb_height / img_height}\n'
            annotation_file.write(line)
        annotation_file.close()