import os
from .base_dataset import BaseDataset
from config import DATA_ROOT

'''dataset source: https://paperswithcode.com/dataset/mvtecad'''

MVTEC_CLS_NAMES = [
    'bottle', 'cable', 'capsule', 'carpet', 'grid',
    'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
    'tile', 'toothbrush', 'transistor', 'wood', 'zipper',
]
MVTEC_ROOT = os.path.join(DATA_ROOT, 'mvtec_anomaly_detection')

class MVTecDataset(BaseDataset):
    def __init__(self, transform, target_transform, clsnames=MVTEC_CLS_NAMES, aug_rate=0.2, root=MVTEC_ROOT, training=True):
        super(MVTecDataset, self).__init__(
            clsnames=clsnames, transform=transform, target_transform=target_transform,
            root=root, aug_rate=aug_rate, training=training
        )
