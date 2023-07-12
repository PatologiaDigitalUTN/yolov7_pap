import cv2
import detectarN
import numpy as np
from utils.plots import plot_one_box
from extract_overlapped_patches import extract_overlapped_patches
from NMS import non_max_suppression

cv_image = cv2.imread("D:\\Descargas\\0b1732c713ea3f212e602c004fedeb65.png")
dets = []
names = []
colors = []

# Check if cv_image resolution is bigger than 640 x 640
if cv_image.shape[0] > 640 or cv_image.shape[1] > 640:
    patches = extract_overlapped_patches(0.1, cv_image)
    # Iterate over dict patches
    for pcenterxy, pimage in patches.items():
        # Detect cells
        res = detectarN.detect(pimage, 'best.pt',
                                patch_center=pcenterxy, glob_img=cv_image)

        dets += res[0]
        names += res[1]
        colors += res[2]
        

# NMS
dets = non_max_suppression(dets,0.2)
#dets = nms.tolist()


# Write results
for *xyxy, conf, cls in reversed(dets):
    label = f'{names[int(cls)]} {conf:.2f}'
    plot_one_box(xyxy, cv_image, label=label, color=colors[int(cls)], line_thickness=1)


# Save image
cv2.imwrite("probando.png", cv_image)