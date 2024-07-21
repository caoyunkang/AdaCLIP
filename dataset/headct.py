import os
from .base_dataset import BaseDataset
from config import DATA_ROOT

'''dataset source: https://www.kaggle.com/datasets/felipekitamura/head-ct-hemorrhage'''
HEADCT_CLS_NAMES = [
    'headct',
]
HEADCT_ROOT = os.path.join(DATA_ROOT, 'HeadCT_anomaly_detection')

class HEADCTDataset(BaseDataset):
    def __init__(self, transform, target_transform, clsnames=HEADCT_CLS_NAMES, aug_rate=0.0, root=HEADCT_ROOT, training=True):
        super(HEADCTDataset, self).__init__(
            clsnames=clsnames, transform=transform, target_transform=target_transform,
            root=root, aug_rate=aug_rate, training=training
        )


