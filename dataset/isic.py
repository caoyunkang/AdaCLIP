import os
from .base_dataset import BaseDataset
from config import DATA_ROOT

'''dataset source: https://challenge.isic-archive.com/data/'''
ISIC_CLS_NAMES = [
    'isic',
]
ISIC_ROOT = os.path.join(DATA_ROOT, 'ISIC')

class ISICDataset(BaseDataset):
    def __init__(self, transform, target_transform, clsnames=ISIC_CLS_NAMES, aug_rate=0.0, root=ISIC_ROOT, training=True):
        super(ISICDataset, self).__init__(
            clsnames=clsnames, transform=transform, target_transform=target_transform,
            root=root, aug_rate=aug_rate, training=training
        )


