import os
import shutil

# Get the list of all .txt files in the current directory.
path_names = '/shared/PatoUTN/PAP/Datasets/originales/1/yolo_idx_class/val'
path_new = '/shared/PatoUTN/PAP/Datasets/originales/1/yolo_idx_class_all_labeled/val'
txt_files = [f for f in os.listdir(path_names) if f.endswith('.txt')]

unlabeled = 0
labeled = 0

# Loop over each .txt file.
for txt_file in txt_files:
    is_labeled = True
    # Open the .txt file in read mode.
    with open(os.path.join(path_names, txt_file), 'r') as f:
        if os.stat(os.path.join(path_names, txt_file)).st_size != 0:
            labeled+=1
            is_labeled = True
            # file isn't empty
        else:
            unlabeled+=1
            is_labeled = False
            # file is empty
    f.close()
    if is_labeled:     
        shutil.copyfile(os.path.join(path_names, txt_file), os.path.join(path_new, txt_file))
        img_file = txt_file.split('.')[0] + '.png'
        shutil.copyfile(os.path.join(path_names, img_file), os.path.join(path_new, img_file))
print('Unlabeled: ', unlabeled)
print('Labeled: ', labeled)