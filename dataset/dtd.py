import os
from .base_dataset import BaseDataset
from config import DATA_ROOT

'''dataset source: https://drive.google.com/drive/folders/10OyPzvI3H6llCZBxKxFlKWt1Pw1tkMK1'''
DTD_CLS_NAMES = [
    'Blotchy_099', 'Fibrous_183', 'Marbled_078', 'Matted_069', 'Mesh_114','Perforated_037','Stratified_154','Woven_001','Woven_068','Woven_104','Woven_125','Woven_127',
]
DTD_ROOT = os.path.join(DATA_ROOT, 'DTD-Synthetic')

class DTDDataset(BaseDataset):
    def __init__(self, transform, target_transform, clsnames=DTD_CLS_NAMES, aug_rate=0.0, root=DTD_ROOT, training=True):
        super(DTDDataset, self).__init__(
            clsnames=clsnames, transform=transform, target_transform=target_transform,
            root=root, aug_rate=aug_rate, training=training
        )
