import os
from .base_dataset import BaseDataset
from config import DATA_ROOT

'''dataset source: https://data.vicos.si/datasets/KSDD/KolektorSDD.zip'''
SDD_CLS_NAMES = [
    'SDD',
]
SDD_ROOT = os.path.join(DATA_ROOT, 'SDD_anomaly_detection')


class SDDDataset(BaseDataset):
    def __init__(self, transform, target_transform, clsnames=SDD_CLS_NAMES, aug_rate=0.0, root=SDD_ROOT, training=True):
        super(SDDDataset, self).__init__(
            clsnames=clsnames, transform=transform, target_transform=target_transform,
            root=root, aug_rate=aug_rate, training=training
        )

