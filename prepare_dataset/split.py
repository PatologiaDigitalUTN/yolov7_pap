import os
import shutil
import numpy as np

def split_dataset(dataset_dir, split_dir, train_ratio, val_ratio, test_ratio):
    # Lista todos los archivos en el directorio del conjunto de datos
    files = os.listdir(dataset_dir)

    # Filtrar archivos de imagen y anotaciones
    images = [file for file in files if file.endswith('.bmp')]
    labels = [file for file in files if file.endswith('.txt')]

    # Asegurarse de que las imágenes y las anotaciones estén ordenadas de la misma manera
    images.sort()
    labels.sort()
  
    # Mezclar los archivos
    indices = np.arange(len(images))  
    np.random.shuffle(indices)
    images = np.array(images)[indices]
    labels = np.array(labels)[indices]

    # Calcula los índices de división para entrenamiento, validación y prueba
    train_split_idx = int(len(images) * train_ratio)
    val_split_idx = int(len(images) * val_ratio) + train_split_idx

    # Divide las imágenes y las anotaciones en conjuntos de entrenamiento, validación y prueba
    train_images = images[:train_split_idx]
    val_images = images[train_split_idx:val_split_idx]
    test_images = images[val_split_idx:]
    
    train_labels = labels[:train_split_idx]
    val_labels = labels[train_split_idx:val_split_idx]
    test_labels = labels[val_split_idx:]

    # Crear directorios para entrenamiento, validación y prueba si no existen
    if not os.path.isdir(split_dir): raise Exception("Split directory doesn't exist")
    train_dir = os.path.join(split_dir, 'train')
    val_dir = os.path.join(split_dir, 'val')
    test_dir = os.path.join(split_dir, 'test')

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Mover las imágenes y las anotaciones a sus respectivos directorios
    for image, label in zip(train_images, train_labels):
        shutil.copy(os.path.join(dataset_dir, image), train_dir)
        shutil.copy(os.path.join(dataset_dir, label), train_dir)

    for image, label in zip(val_images, val_labels):
        shutil.copy(os.path.join(dataset_dir, image), val_dir)
        shutil.copy(os.path.join(dataset_dir, label), val_dir)

    for image, label in zip(test_images, test_labels):
        shutil.copy(os.path.join(dataset_dir, image), test_dir)
        shutil.copy(os.path.join(dataset_dir, label), test_dir)

# Usar la función
split_dataset('/shared/PatoUTN/PAP/Datasets/SIPaKMeD/original/2/NoSpliteado', 
'/shared/PatoUTN/PAP/Datasets/SIPaKMeD/original/2/Split' , 0.8, 0.1, 0.1)