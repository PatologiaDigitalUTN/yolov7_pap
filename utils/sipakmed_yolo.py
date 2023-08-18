# Dataset utils and dataloaders

import os

from itertools import repeat
from multiprocessing.pool import ThreadPool
from pathlib import Path
from threading import Thread

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ExifTags
from torch.utils.data import Dataset
from tqdm import tqdm

from copy import deepcopy
#from pycocotools import mask as maskUtils
from torchvision.utils import save_image
from torchvision.ops import roi_pool, roi_align, ps_roi_pool, ps_roi_align

# Loads all orignal images (uncut) from a given folder. Returns CV2 Images
def load_images_from_folder(folder):
    images = []
    file_list = os.listdir(folder)
    bmp_files = [file for file in file_list if file.lower().endswith('.bmp')]
    for filename in bmp_files:
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images

# Loads all the crops of an image given the folder and the number of image
def load_cropped_from_images_folder(folder, number):
    images = []
    file_list = os.listdir(folder)
    bmp_files = [file for file in file_list if file.lower().endswith('.bmp') and file.startswith(number)]
    for filename in bmp_files:
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images

# Template matches crops into image returning coordinates for all crops in [topleftx, toplefty, bottomrightx, bottomrighty] fashion
def find_crops_image(image, crops):
    crop_coords = []

    for crop in crops:
        crop_height, crop_width = crop.shape[:2]
        result = cv2.matchTemplate(image, crop, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        top_left = max_loc
        #print(top_left)
        bottom_right = (top_left[0] + crop_width, top_left[1] + crop_height)
        crop = [top_left[0], top_left[1], bottom_right[0], bottom_right[1]]
        crop_coords.append(crop)
    
    return crop_coords

# Given an image folder and a crop folder, it returns all the crops coordinates for an image
def get_crop_coordinates_images(original_folder, crop_folder):
    coords = []
    
    # Get Images from folder
    file_list = os.listdir(original_folder)
    bmp_files = [file for file in file_list if file.lower().endswith('.bmp')]
    bmp_paths = [os.path.join(original_folder, bmp_file) for bmp_file in bmp_files]

    #For each image, load crops to coords
    for path in bmp_paths:
        coords.append(get_crop_coordinates_image(path, crop_folder))

    return coords

# Given an image path, return all crop coordinates for said image
def get_crop_coordinates_image(img_path, crop_folder):
    file_name = os.path.basename(img_path)
    img_number, file_extension = os.path.splitext(file_name)
    img = cv2.imread(img_path)
    
    crops = load_cropped_from_images_folder(crop_folder, img_number)

    # For each crop of an image, get crops
    img_crops = find_crops_image(img, crops)

    return img_crops

# Given an array of detection, transforms the format to match YOLO's format
def transform_coords_yolo_format(coords):

    yolo_coords = []

    for coord in coords:
        yolo_coords.push(get_yolo_format(coord))


    return yolo_coords

# Given the top-left and bottom-right x, y coordinates, returns center XY, width and height of the bounding box (normalized by sipakmed image dimensions as default)
# First element in the array is the class, 0 means it is a cell
def get_yolo_format(coord, img_width=2048, img_height=1536):

    centerx = ((coord[0] + coord[2]) / 2) / img_width
    centery = ((coord[1] + coord[3]) / 2) / img_height

    width = (coord[2] - coord[0]) / img_width
    height = (coord[3] - coord[1]) / img_height

    return [0, centerx, centery, width, height]

# Given paths for full images and cropped images folders, writes a txt of the detections of each image, using the YOLO format
def write_coordinates(img_folder, crop_folder, write_path):

    if not os.path.exists(write_path):
        os.makedirs(write_path)

    # Get Images from folder
    file_list = os.listdir(img_folder)
    bmp_files = [file for file in file_list if file.lower().endswith('.bmp')]
    bmp_paths = [os.path.join(img_folder, bmp_file) for bmp_file in bmp_files]

    #For each image, load crops to coords
    for path in bmp_paths:
        #coords.append(get_crop_coordinates_image(path, crop_folder))

        file_name = os.path.basename(path)
        img_number, file_extension = os.path.splitext(file_name)
        img = cv2.imread(path)

        # Output file name is number of image.txt
        output_file_name = str(img_number) + ".txt"
        full_output_path = os.path.join(write_path, output_file_name)

        # Load crops from image
        crops = load_cropped_from_images_folder(crop_folder, img_number)

        # For each image, get crop coords
        dets = find_crops_image(img, crops)
        
        with open(full_output_path, "w") as output_file:
        # Iterate through dets
            for det in dets:
                # Convert coords into yolo format
                det_yolo = get_yolo_format(det)

                # Convert each array to a string and join its elements with a tab character
                row = " ".join(str(element) for element in det_yolo)
                # Write the row to the file
                output_file.write(row + "\n")

    return 

# Write coordinates for all of the images


# Parabasal
write_coordinates("D:\Descargas\im_Parabasal\im_Parabasal", "D:\Descargas\im_Parabasal\im_Parabasal\CROPPED", "D:\Descargas\detecciones\Parabasal")

# Koilocytotic
write_coordinates("D:\Descargas\im_Koilocytotic\im_Koilocytotic", "D:\Descargas\im_Koilocytotic\im_Koilocytotic\CROPPED", "D:\Descargas\detecciones\Koilocytotic")

# Dyskeratotic
write_coordinates("D:\Descargas\im_Dyskeratotic\im_Dyskeratotic", "D:\Descargas\im_Dyskeratotic\im_Dyskeratotic\CROPPED", "D:\Descargas\detecciones\Dyskeratotic")

# Metaplastic
write_coordinates("D:\Descargas\im_Metaplastic\im_Metaplastic", "D:\Descargas\im_Metaplastic\im_Metaplastic\CROPPED", "D:\Descargas\detecciones\Metaplastic")

# Superficial-Intermediate
write_coordinates("D:\Descargas\im_Superficial-Intermediate\im_Superficial-Intermediate", "D:\Descargas\im_Superficial-Intermediate\im_Superficial-Intermediate\CROPPED", "D:\Descargas\detecciones\Superficial-Intermediate")

