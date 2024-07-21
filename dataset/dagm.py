import os
from .base_dataset import BaseDataset
from config import DATA_ROOT

'''dataset source: https://hci.iwr.uni-heidelberg.de/content/weakly-supervised-learning-industrial-optical-inspection'''
DAGM_CLS_NAMES = [
    'Class1', 'Class2', 'Class3', 'Class4', 'Class5','Class6','Class7','Class8','Class9','Class10',
]
DAGM_ROOT = os.path.join(DATA_ROOT, 'DAGM_anomaly_detection')

class DAGMDataset(BaseDataset):
    def __init__(self, transform, target_transform, clsnames=DAGM_CLS_NAMES, aug_rate=0.0, root=DAGM_ROOT, training=True):
        super(DAGMDataset, self).__init__(
            clsnames=clsnames, transform=transform, target_transform=target_transform,
            root=root, aug_rate=aug_rate, training=training
        )
