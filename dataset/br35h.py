import os
from .base_dataset import BaseDataset
from config import DATA_ROOT

'''dataset source: https://www.kaggle.com/datasets/ahmedhamada0/brain-tumor-detection'''

Br35h_CLS_NAMES = [
    'br35h',
]
Br35h_ROOT = os.path.join(DATA_ROOT, 'Br35h_anomaly_detection')

class Br35hDataset(BaseDataset):
    def __init__(self, transform, target_transform, clsnames=Br35h_CLS_NAMES, aug_rate=0.0, root=Br35h_ROOT, training=True):
        super(Br35hDataset, self).__init__(
            clsnames=clsnames, transform=transform, target_transform=target_transform,
            root=root, aug_rate=aug_rate, training=training
        )

