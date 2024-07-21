import os
import json
import random
from config import DATA_ROOT

ColonDB_ROOT = os.path.join(DATA_ROOT, 'CVC-ColonDB')

class ColonDBSolver(object):
    CLSNAMES = [
        'ColonDB',
    ]

    def __init__(self, root=ColonDB_ROOT, train_ratio=0.5):
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
                    mask_names = os.listdir(f'{cls_dir}/ground_truth/{specie}') if is_abnormal else None
                    img_names.sort()
                    mask_names.sort() if mask_names is not None else None
                    for idx, img_name in enumerate(img_names):
                        info_img = dict(
                            img_path=f'{cls_name}/{phase}/{specie}/{img_name}',
                            mask_path=f'{cls_name}/ground_truth/{specie}/{mask_names[idx]}' if is_abnormal else '',
                            cls_name=cls_name,
                            specie_name=specie,
                            anomaly=1 if is_abnormal else 0,
                        )
                        cls_info.append(info_img)

                info[phase][cls_name] = cls_info

        with open(self.meta_path, 'w') as f:
            f.write(json.dumps(info, indent=4) + "\n")


if __name__ == '__main__':
    runner = ColonDBSolver(root=ColonDB_ROOT)
    runner.run()
