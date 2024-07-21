import os
from .base_dataset import BaseDataset
from config import DATA_ROOT

'''dataset source: https://ieeexplore.ieee.org/document/9434087/references#references'''
TN3K_CLS_NAMES = [
    'tn3k',
]
TN3K_ROOT = os.path.join(DATA_ROOT, 'TN3K')

class TN3KDataset(BaseDataset):
    def __init__(self, transform, target_transform, clsnames=TN3K_CLS_NAMES, aug_rate=0.0, root=TN3K_ROOT, training=True):
        super(TN3KDataset, self).__init__(
            clsnames=clsnames, transform=transform, target_transform=target_transform,
            root=root, aug_rate=aug_rate, training=training
        )


