import os
from .base_dataset import BaseDataset
from config import DATA_ROOT

'''dataset source: https://github.com/stepanje/MPDD'''
MPDD_CLS_NAMES = [
    'bracket_black', 'bracket_brown', 'bracket_white', 'connector', 'metal_plate','tubes',
]
MPDD_ROOT = os.path.join(DATA_ROOT, 'MPDD')

class MPDDDataset(BaseDataset):
    def __init__(self, transform, target_transform, clsnames=MPDD_CLS_NAMES, aug_rate=0.0, root=MPDD_ROOT, training=True):
        super(MPDDDataset, self).__init__(
            clsnames=clsnames, transform=transform, target_transform=target_transform,
            root=root, aug_rate=aug_rate, training=training
        )

