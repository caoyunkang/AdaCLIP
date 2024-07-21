import os
from .base_dataset import BaseDataset
from config import DATA_ROOT

'''dataset source: https://paperswithcode.com/dataset/cvc-clinicdb'''
ClinicDB_CLS_NAMES = [
    'ClinicDB',
]
ClinicDB_ROOT = os.path.join(DATA_ROOT, 'CVC-ClinicDB')

class ClinicDBDataset(BaseDataset):
    def __init__(self, transform, target_transform, clsnames=ClinicDB_CLS_NAMES, aug_rate=0.0, root=ClinicDB_ROOT, training=True):
        super(ClinicDBDataset, self).__init__(
            clsnames=clsnames, transform=transform, target_transform=target_transform,
            root=root, aug_rate=aug_rate, training=training
        )
