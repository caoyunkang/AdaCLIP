import os
from .base_dataset import BaseDataset
from config import DATA_ROOT

'''dataset source: https://avires.dimi.uniud.it/papers/btad/btad.zip'''
BTAD_CLS_NAMES = [
    '01', '02', '03',
]
BTAD_ROOT = os.path.join(DATA_ROOT, 'BTech_Dataset_transformed')

class BTADDataset(BaseDataset):
    def __init__(self, transform, target_transform, clsnames=BTAD_CLS_NAMES, aug_rate=0.0, root=BTAD_ROOT, training=True):
        super(BTADDataset, self).__init__(
            clsnames=clsnames, transform=transform, target_transform=target_transform,
            root=root, aug_rate=aug_rate, training=training
        )
