import os
from .base_dataset import BaseDataset
from config import DATA_ROOT

'''dataset source: http://mv.cvc.uab.es/projects/colon-qa/cvccolondb'''
ColonDB_CLS_NAMES = [
    'ColonDB',
]
ColonDB_ROOT = os.path.join(DATA_ROOT, 'CVC-ColonDB')

class ColonDBDataset(BaseDataset):
    def __init__(self, transform, target_transform, clsnames=ColonDB_CLS_NAMES, aug_rate=0.0, root=ColonDB_ROOT, training=True):
        super(ColonDBDataset, self).__init__(
            clsnames=clsnames, transform=transform, target_transform=target_transform,
            root=root, aug_rate=aug_rate, training=training
        )


