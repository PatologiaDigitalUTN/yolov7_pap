import random
import os
import shutil

split_original_images = True

original_images_path = '/shared/PatoUTN/PAP/Datasets/SIPaKMeD/original/1/Original'
labels = '/shared/PatoUTN/PAP/Datasets/SIPaKMeD/original/1/Labels'
original_images_splitted_path = '/shared/PatoUTN/PAP/Datasets/SIPaKMeD/original/1/Particionado-Yolo'

# Split images into test/train/val
train = 0.8
val = 0.1
test = 0.1

if not os.path.exists(original_images_splitted_path):
    os.makedirs(original_images_splitted_path)

images = os.listdir(original_images_path)
random.Random(4).shuffle(images)

total = len(images)

train_images_len = int(total * train)
test_images_len = int(total * test)
val_images_len = int(total * val)

train_images = images[:train_images_len]
test_images = images[train_images_len:train_images_len+test_images_len]
val_images = images[train_images_len+test_images_len:]

if split_original_images:
    for image_type, image_names in [("train", train_images), ("test", test_images), ("val", val_images)]:
        for image_name in image_names:
            dst_folder_path = os.path.join(original_images_splitted_path, image_type)
            os.makedirs(dst_folder_path, exist_ok=True)
            dst_path = os.path.join(dst_folder_path, image_name)
            src_path = os.path.join(original_images_path, image_name)
            shutil.copy(src_path, dst_path)
            label = os.path.splitext(image_name)[0] + '.txt'
            dst_path = os.path.join(dst_folder_path, label)
            src_path = os.path.join(labels, label)
            shutil.copy(src_path, dst_path)