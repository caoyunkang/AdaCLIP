"""
Base class for our zero-shot anomaly detection dataset
"""
import json
import os
import random
import numpy as np
import torch.utils.data as data
from PIL import Image
import cv2
from config import DATA_ROOT


class DataSolver:
    def __init__(self, root, clsnames):
        self.root = root
        self.clsnames = clsnames
        self.path = os.path.join(root, 'meta.json')

    def run(self):
        with open(self.path, 'r') as f:
            info = json.load(f)

        info_required = dict(train={}, test={})
        for cls in self.clsnames:
            for k in info.keys():
                info_required[k][cls] = info[k][cls]

        return info_required


class BaseDataset(data.Dataset):
    def __init__(self, clsnames, transform, target_transform, root, aug_rate=0., training=True):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.aug_rate = aug_rate
        self.training = training
        self.data_all = []
        self.cls_names = clsnames

        solver = DataSolver(root, clsnames)
        meta_info = solver.run()

        self.meta_info = meta_info['test']  # Only utilize the test dataset for both training and testing
        for cls_name in self.cls_names:
            self.data_all.extend(self.meta_info[cls_name])

        self.length = len(self.data_all)

    def __len__(self):
        return self.length

    def combine_img(self, cls_name):
        """
        From April-GAN: https://github.com/ByChelsea/VAND-APRIL-GAN
        Here we combine four images into a single image for data augmentation.
        """
        img_info = random.sample(self.meta_info[cls_name], 4)

        img_ls = []
        mask_ls = []

        for data in img_info:
            img_path = os.path.join(self.root, data['img_path'])
            mask_path = os.path.join(self.root, data['mask_path'])

            img = Image.open(img_path).convert('RGB')
            img_ls.append(img)

            if not data['anomaly']:
                img_mask = Image.fromarray(np.zeros((img.size[0], img.size[1])), mode='L')
            else:
                img_mask = np.array(Image.open(mask_path).convert('L')) > 0
                img_mask = Image.fromarray(img_mask.astype(np.uint8) * 255, mode='L')

            mask_ls.append(img_mask)

        # Image
        image_width, image_height = img_ls[0].size
        result_image = Image.new("RGB", (2 * image_width, 2 * image_height))
        for i, img in enumerate(img_ls):
            row = i // 2
            col = i % 2
            x = col * image_width
            y = row * image_height
            result_image.paste(img, (x, y))

        # Mask
        result_mask = Image.new("L", (2 * image_width, 2 * image_height))
        for i, img in enumerate(mask_ls):
            row = i // 2
            col = i % 2
            x = col * image_width
            y = row * image_height
            result_mask.paste(img, (x, y))

        return result_image, result_mask

    def __getitem__(self, index):
        data = self.data_all[index]
        img_path = os.path.join(self.root, data['img_path'])
        mask_path = os.path.join(self.root, data['mask_path'])
        cls_name = data['cls_name']
        anomaly = data['anomaly']
        random_number = random.random()

        if self.training and random_number < self.aug_rate:
            img, img_mask = self.combine_img(cls_name)
        else:
            if img_path.endswith('.tif'):
                img = cv2.imread(img_path)
                img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            else:
                img = Image.open(img_path).convert('RGB')
            if anomaly == 0:
                img_mask = Image.fromarray(np.zeros((img.size[0], img.size[1])), mode='L')
            else:
                if data['mask_path']:
                    img_mask = np.array(Image.open(mask_path).convert('L')) > 0
                    img_mask = Image.fromarray(img_mask.astype(np.uint8) * 255, mode='L')
                else:
                    img_mask = Image.fromarray(np.zeros((img.size[0], img.size[1])), mode='L')
        # Transforms
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None and img_mask is not None:
            img_mask = self.target_transform(img_mask)
        if img_mask is None:
            img_mask = []

        return {
            'img': img,
            'img_mask': img_mask,
            'cls_name': cls_name,
            'anomaly': anomaly,
            'img_path': img_path
        }
