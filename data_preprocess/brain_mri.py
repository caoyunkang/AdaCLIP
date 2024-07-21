import os
import json
import random
from config import DATA_ROOT

BrainMRI_ROOT = os.path.join(DATA_ROOT, 'BrainMRI')

class BrainMRISolver(object):
    CLSNAMES = [
        'brain_mri',
    ]

    def __init__(self, root=BrainMRI_ROOT, train_ratio=0.5):
        self.root = root
        self.meta_path = f'{root}/meta.json'
        self.train_ratio = train_ratio

    def run(self):
        self.generate_meta_info()

    def generate_meta_info(self):
        info = dict(train={}, test={})
        for cls_name in self.CLSNAMES:
            cls_dir = f'{self.root}/{cls_name}'
            for phase in ['train', 'test']:
                cls_info = []
                species = os.listdir(f'{cls_dir}/{phase}')
                for specie in species:
                    is_abnormal = True if specie not in ['good'] else False
                    img_names = os.listdir(f'{cls_dir}/{phase}/{specie}')
                    img_names.sort()

                    for idx, img_name in enumerate(img_names):
                        info_img = dict(
                            img_path=f'{cls_name}/{phase}/{specie}/{img_name}',
                            mask_path=f'',
                            cls_name=cls_name,
                            specie_name=specie,
                            anomaly=1 if is_abnormal else 0,
                        )
                        cls_info.append(info_img)

                info[phase][cls_name] = cls_info

        with open(self.meta_path, 'w') as f:
            f.write(json.dumps(info, indent=4) + "\n")


if __name__ == '__main__':
    runner = BrainMRISolver(root=BrainMRI_ROOT)
    runner.run()
