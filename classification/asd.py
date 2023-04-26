import os
import cv2
from skimage.util import random_noise
from skimage.restoration import denoise_tv_chambolle, denoise_bilateral
import random
import shutil

img = cv2.imread("E:\\MLPathologyProject\\pap\\CRIC\\imgs_for_classification_split\\train\\Negative for intraepithelial lesion\\0c2a10baed0650fa1f7bb87e5d83a545_67_479.png")

sigma = 0.005 
noisy = random_noise(img, var=sigma**2)
nova_img = denoise_bilateral(noisy, sigma_color=0.01, sigma_spatial=5, channel_axis=-1)
nova_img = nova_img * 255

cv2.imwrite("E:\\MLPathologyProject\\pap\\CRIC\\playground\\" + 'denoise_bilateral.png', nova_img)

sigma = 0.005 
noisy = random_noise(img, var=sigma**2)
nova_img = denoise_tv_chambolle(noisy, weight=0.05)
nova_img = nova_img * 255

cv2.imwrite("E:\\MLPathologyProject\\pap\\CRIC\\playground\\" + 'denoise_tv_chambolle.png', nova_img)

sigma = 0.05 
noisy = random_noise(img, var=sigma**2)
nova_img = noisy
nova_img = nova_img * 255

cv2.imwrite("E:\\MLPathologyProject\\pap\\CRIC\\playground\\" + 'random_noise.png', nova_img)