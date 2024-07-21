import os
from .base_dataset import BaseDataset
from config import DATA_ROOT

'''dataset source: https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar'''
VISA_CLS_NAMES = [
    'candle', 'capsules', 'cashew', 'chewinggum', 'fryum',
    'macaroni1', 'macaroni2', 'pcb1', 'pcb2', 'pcb3',
    'pcb4', 'pipe_fryum',
]

VISA_ROOT = os.path.join(DATA_ROOT, 'VisA_20220922')

class VisaDataset(BaseDataset):
    def __init__(self, transform, target_transform, clsnames=VISA_CLS_NAMES, aug_rate=0.0, root=VISA_ROOT, training=True):
        super(VisaDataset, self).__init__(
            clsnames=clsnames, transform=transform, target_transform=target_transform,
            root=root, aug_rate=aug_rate, training=training
        )

