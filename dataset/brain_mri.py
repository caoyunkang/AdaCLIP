import os
from .base_dataset import BaseDataset
from config import DATA_ROOT

'''dataset source: https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection'''
BrainMRI_CLS_NAMES = [
    'brain_mri',
]
BrainMRI_ROOT = os.path.join(DATA_ROOT, 'BrainMRI')

class BrainMRIDataset(BaseDataset):
    def __init__(self, transform, target_transform, clsnames=BrainMRI_CLS_NAMES, aug_rate=0.0, root=BrainMRI_ROOT, training=True):
        super(BrainMRIDataset, self).__init__(
            clsnames=clsnames, transform=transform, target_transform=target_transform,
            root=root, aug_rate=aug_rate, training=training
        )
