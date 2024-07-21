import os
import numpy as np
from sklearn.model_selection import train_test_split
import cv2
import argparse

from config import DATA_ROOT

dataset_root = os.path.join(DATA_ROOT, 'KolektorSDD')

dirs = os.listdir(dataset_root)
normal_images = list()
normal_labels = list()
normal_fname = list()
outlier_images = list()
outlier_labels = list()
outlier_fname = list()
for d in dirs:
    files = os.listdir(os.path.join(dataset_root, d))
    images = list()
    for f in files:
        if 'jpg' in f[-3:]:
            images.append(f)

    for image in images:
        split_images = list()
        split_labels = list()
        image_name = image.split('.')[0]
        image_data = cv2.imread(os.path.join(dataset_root, d, image))
        label_data = cv2.imread(os.path.join(dataset_root, d, image_name + '_label.bmp'))
        if image_data.shape != label_data.shape:
            raise ValueError
        image_length = image_data.shape[0]
        split_images.append(image_data[:image_length // 3, :, :])
        split_images.append(image_data[image_length // 3:image_length * 2 // 3, :, :])
        split_images.append(image_data[image_length * 2 // 3:, :, :])
        split_labels.append(label_data[:image_length // 3, :, :])
        split_labels.append(label_data[image_length // 3:image_length * 2 // 3, :, :])
        split_labels.append(label_data[image_length * 2 // 3:, :, :])
        for i, (im, la) in enumerate(zip(split_images, split_labels)):
            if np.max(la) != 0:
                outlier_images.append(im)
                outlier_labels.append(la)
                outlier_fname.append(d + '_' + image_name + '_' + str(i))
            else:
                normal_images.append(im)
                normal_labels.append(la)
                normal_fname.append(d + '_' + image_name + '_' + str(i))

normal_train, normal_test, normal_name_train, normal_name_test = train_test_split(normal_images, normal_fname, test_size=0.25, random_state=42)

target_root = '../datasets/SDD_anomaly_detection/SDD'
train_root = os.path.join(target_root, 'train/good')
if not os.path.exists(train_root):
    os.makedirs(train_root)
for image, name in zip(normal_train, normal_name_train):
    cv2.imwrite(os.path.join(train_root, name + '.png'), image)

test_root = os.path.join(target_root, 'test/good')
if not os.path.exists(test_root):
    os.makedirs(test_root)
for image, name in zip(normal_test, normal_name_test):
    cv2.imwrite(os.path.join(test_root, name + '.png'), image)

defect_root = os.path.join(target_root, 'test/defect')
label_root = os.path.join(target_root, 'ground_truth/defect')
if not os.path.exists(defect_root):
    os.makedirs(defect_root)
if not os.path.exists(label_root):
    os.makedirs(label_root)
for image, label, name in zip(outlier_images, outlier_labels, outlier_fname):
    cv2.imwrite(os.path.join(defect_root, name + '.png'), image)
    cv2.imwrite(os.path.join(label_root, name + '_mask.png'), label)

print("Done")