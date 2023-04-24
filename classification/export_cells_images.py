import json
import cv2
import os

def correctSize(xx, yy):
  corrected_x = width_in_image
  corrected_y = height_in_image
  # if xx + width_in_image > 1:
  #   corrected_x = width_in_image - ((xx + width_in_image) - 1)
  
  # if yy + height_in_image > 1:
  #   corrected_y = height_in_image - ((yy + height_in_image) - 1)

  return corrected_x, corrected_y

classes = {
    "Negative for intraepithelial lesion" : 0,
    "ASC-US" : 1,
    "ASC-H" : 2,
    "LSIL" : 3,
    "HSIL" : 4,
    "SCC" : 5,
}

# JSON file
filename = 'C:\\PatoUTN\\CRIC\\original\\classifications.json'
f = open (filename, "r")
data = json.loads(f.read())

img_width = 1376
img_height = 1020

size = 640

width = 90
height = 90

dest = "C:\\PatoUTN\\CRIC\\imgs_for_classification"

for image in data:

    # if not image['image_name'] == 'fb5e83755f682922aee859fd52013c36.png':
    #   continue
    
    image_name = image['image_name']

    image_name_only, image_extension = os.path.splitext(image_name)

    cv_img = cv2.imwrite(image_name)

    for cell in image['classifications']:
      cell_class = 0

      x = int(cell['nucleus_x'])
      y = int(cell['nucleus_y'])

     #falta agregar la clase

      cell_image = cv_img[int(y-height/2):int(y+height/2), int(x-width/2):int(x+width/2)]
      cv2.imwrite(os.path.join(dest, f"{image_name_only}_{x}_{y}.{image_extension}"))

# Closing JSON file
f.close()


