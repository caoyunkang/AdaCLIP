import os
import numpy as np
from sklearn.model_selection import train_test_split
import cv2
import argparse
from config import DATA_ROOT

dataset_root = os.path.join(DATA_ROOT, 'DAGM2007')

class_names = os.listdir(dataset_root)


for class_name in class_names:
    states = os.listdir(os.path.join(dataset_root, class_name))
    for state in states:
        images = list()
        mask = list()
        files = os.listdir(os.path.join(dataset_root, class_name,state))
        for f in files:
            if 'PNG' in f[-3:]:
                images.append(f)
        files = os.listdir(os.path.join(dataset_root, class_name, state,'Label'))
        for f in files:
            if 'PNG' in f[-3:]:
                mask.append(f)
        normal_image_path_train = list()
        normal_image_path_test = list()
        normal_image_path = list()
        abnormal_image_path = list()
        abnormal_image_label = list()
        for f in images:
            id = f[-8:-4]
            flag = 0
            for y in mask:
                if id in y:
                    abnormal_image_path.append(f)
                    abnormal_image_label.append(y)
                    flag = 1
                    break
            if flag == 0:
                normal_image_path.append(f)

        if len(abnormal_image_path) != len(abnormal_image_label):
            raise ValueError
        length = len(abnormal_image_path)

        normal_image_path_test = normal_image_path[:length]
        normal_image_path_train = normal_image_path[length:]

        target_root = '../datasets/DAGM_anomaly_detection'

        train_root = os.path.join(target_root, class_name, 'train','good')
        if not os.path.exists(train_root):
            os.makedirs(train_root)
        for f in normal_image_path_train:
            image_data = cv2.imread(os.path.join(dataset_root, class_name, state,f))
            cv2.imwrite(os.path.join(train_root,f), image_data)

        test_root = os.path.join(target_root, class_name, 'test','good')
        if not os.path.exists(test_root):
            os.makedirs(test_root)
        for f in normal_image_path_test:
            image_data = cv2.imread(os.path.join(dataset_root, class_name, state,f))
            cv2.imwrite(os.path.join(test_root,f), image_data)

        test_root = os.path.join(target_root, class_name, 'test','defect')
        if not os.path.exists(test_root):
            os.makedirs(test_root)
        for f in abnormal_image_path:
            image_data = cv2.imread(os.path.join(dataset_root, class_name, state,f))
            cv2.imwrite(os.path.join(test_root,f), image_data)

        test_root = os.path.join(target_root, class_name, 'ground_truth','defect')
        if not os.path.exists(test_root):
            os.makedirs(test_root)
        for f in mask:
            image_data = cv2.imread(os.path.join(dataset_root, class_name, state,'Label',f))
            cv2.imwrite(os.path.join(test_root,f), image_data)



print("Done")