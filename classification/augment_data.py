import os
import cv2
from skimage.util import random_noise
from skimage.restoration import denoise_tv_chambolle, denoise_bilateral
import random
import shutil


def balance(src_folder, dst_folder=None):
    shutil.copytree(src_folder, dst_folder)
    
    train_folder = os.path.join(dst_folder, 'train')

    imgs_class_count = {}
    maxCaseAmount = 0

    # Get biggest class and store amounts in a dict
    for type in os.listdir(train_folder):
        imgs_class_count[type] = len(os.listdir(os.path.join(train_folder, type)))
        if imgs_class_count[type] > maxCaseAmount:
            maxCaseAmount = imgs_class_count[type]
    
    # Balance small classes choosing random samples to be randomly augmented
    for type in imgs_class_count.keys():
        type_path = os.path.join(train_folder, type)
        original_images = os.listdir(type_path)
        if imgs_class_count[type] < maxCaseAmount:
            img_diff = maxCaseAmount - imgs_class_count[type]
            # Avoid doing augmentation of the same image if it is possible
            if imgs_class_count[type] >= img_diff:
                sample = random.sample(range(0, imgs_class_count[type]), img_diff)
                for rnd in sample:
                    img_name = original_images[rnd]
                    img = cv2.imread(os.path.join(type_path, img_name))
                    ag_img = operacaoAugmentation_retornar(random.randint(1,10), img)
                    cv2.imwrite(os.path.join(type_path, 'ag_' + img_name), ag_img)

            else:
                ag_count = 0
                # Here the images must be augmented more than once
                for img_name in original_images:
                    img = cv2.imread(os.path.join(type_path, img_name))
                    ag_img = operacaoAugmentation_retornar(random.randint(1,10), img)
                    cv2.imwrite(os.path.join(type_path, f'ag_{ag_count}_' + img_name), ag_img)
                    ag_count += 1
                img_diff -= len(original_images)
                # After each image is augmented, we augment a original ohe randomly
                for i in range(img_diff):
                    img_name = random.choice(original_images)
                    img = cv2.imread(os.path.join(type_path, img_name))
                    ag_img = operacaoAugmentation_retornar(random.randint(1,10), img)
                    cv2.imwrite(os.path.join(type_path, f'ag_{ag_count}_' + img_name), ag_img)
                    ag_count += 1


def operacaoAugmentation_retornar(operacao, img):
    # rotacionar
    if(operacao == 1):
        nova_img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    elif(operacao == 2):
        nova_img = cv2.rotate(img, cv2.ROTATE_180)
    elif(operacao == 3):
        nova_img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    # espelhar
    elif(operacao == 4):
        nova_img= cv2.flip(img, 1)
    elif(operacao == 5):
        img_rotate_90 = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        nova_img = cv2.flip(img_rotate_90, 1)
    elif(operacao == 6):
        img_rotate_180 = cv2.rotate(img, cv2.ROTATE_180)
        nova_img = cv2.flip(img_rotate_180, 1)
    elif(operacao == 7):
        img_rotate_270 = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        nova_img = cv2.flip(img_rotate_270, 1)
    elif(operacao == 8):
        sigma = 0.05 
        noisy = random_noise(img, var=sigma**2)
        nova_img = noisy
        nova_img = nova_img * 255
    elif(operacao == 9):
        sigma = 0.005 
        noisy = random_noise(img, var=sigma**2)
        nova_img = denoise_tv_chambolle(noisy, weight=0.05)
        nova_img = nova_img * 255
    elif(operacao == 10):
        sigma = 0.005 
        noisy = random_noise(img, var=sigma**2)
        nova_img = denoise_bilateral(noisy, sigma_color=0.01, sigma_spatial=5, channel_axis=-1)
        nova_img = nova_img * 255
    return nova_img

            
if __name__ == '__main__':
    balance('E:\\MLPathologyProject\\pap\\CRIC\\imgs_for_classification_split',
            'E:\\MLPathologyProject\\pap\\CRIC\\imgs_for_classification_split_augmented')