import os
from collections import defaultdict

def count_objects_in_folder(folder):
    object_count = 0
    files_count = 0
    class_count = defaultdict(int)  # Dictionary to count the classes

    file_list = os.listdir(folder)

    for file in file_list:
        if file.endswith('.txt'):
            files_count += 1
            file_path = os.path.join(folder, file)
            with open(file_path, 'r') as file:
                lines = file.readlines()
                object_count += len(lines)
                for line in lines:
                    class_name = line.split()[0]  # First character representing the class
                    class_count[class_name] += 1

    return object_count, class_count, files_count

train_folder = '/shared/PatoUTN/PAP/Datasets/originales/1/yolo/train/'
test_folder = '/shared/PatoUTN/PAP/Datasets/originales/1/yolo/test'
validation_folder = '/shared/PatoUTN/PAP/Datasets/originales/1/yolo/val/'

objects_train, classes_train, files_train = count_objects_in_folder(train_folder)
objects_test, classes_test, files_test = count_objects_in_folder(test_folder)
objects_validation, classes_validation, files_validation = count_objects_in_folder(validation_folder)

print("Files in train folder:", files_train)
print("Objects in train folder:", objects_train)
print("Classes in train folder:", dict(classes_train))

print("Files in test folder:", files_test)
print("Objects in test folder:", objects_test)
print("Classes in test folder:", dict(classes_test))

print("Files in validation folder:", files_validation)
print("Objects in validation folder:", objects_validation)
print("Classes in validation folder:", dict(classes_validation))
