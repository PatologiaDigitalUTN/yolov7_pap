import cv2
from process_inference import process_inference

img = cv2.imread("E:\\MLPathologyProject\\pap\\CRIC\\base\\30cef53c58d14524f9d49e24af9887fa.png")
res_img = process_inference(img)
