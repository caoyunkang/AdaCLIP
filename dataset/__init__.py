from .mvtec import MVTEC_CLS_NAMES, MVTecDataset, MVTEC_ROOT
from .visa import VISA_CLS_NAMES, VisaDataset, VISA_ROOT
from .mpdd import MPDD_CLS_NAMES, MPDDDataset, MPDD_ROOT
from .btad import BTAD_CLS_NAMES, BTADDataset, BTAD_ROOT
from .sdd import SDD_CLS_NAMES, SDDDataset, SDD_ROOT
from .dagm import DAGM_CLS_NAMES, DAGMDataset, DAGM_ROOT
from .dtd import DTD_CLS_NAMES,DTDDataset,DTD_ROOT
from .isic import ISIC_CLS_NAMES,ISICDataset,ISIC_ROOT
from .colondb import ColonDB_CLS_NAMES, ColonDBDataset, ColonDB_ROOT
from .clinicdb import ClinicDB_CLS_NAMES, ClinicDBDataset, ClinicDB_ROOT
from .tn3k import TN3K_CLS_NAMES, TN3KDataset, TN3K_ROOT
from .headct import HEADCT_CLS_NAMES,HEADCTDataset,HEADCT_ROOT
from .brain_mri import BrainMRI_CLS_NAMES,BrainMRIDataset,BrainMRI_ROOT
from .br35h import Br35h_CLS_NAMES,Br35hDataset,Br35h_ROOT
from torch.utils.data import ConcatDataset

dataset_dict = {
    'br35h': (Br35h_CLS_NAMES, Br35hDataset, Br35h_ROOT),
    'brain_mri': (BrainMRI_CLS_NAMES, BrainMRIDataset, BrainMRI_ROOT),
    'btad': (BTAD_CLS_NAMES, BTADDataset, BTAD_ROOT),
    'clinicdb': (ClinicDB_CLS_NAMES, ClinicDBDataset, ClinicDB_ROOT),
    'colondb': (ColonDB_CLS_NAMES, ColonDBDataset, ColonDB_ROOT),
    'dagm': (DAGM_CLS_NAMES, DAGMDataset, DAGM_ROOT),
    'dtd': (DTD_CLS_NAMES, DTDDataset, DTD_ROOT),
    'headct': (HEADCT_CLS_NAMES, HEADCTDataset, HEADCT_ROOT),
    'isic': (ISIC_CLS_NAMES, ISICDataset, ISIC_ROOT),
    'mpdd': (MPDD_CLS_NAMES, MPDDDataset, MPDD_ROOT),
    'mvtec': (MVTEC_CLS_NAMES, MVTecDataset, MVTEC_ROOT),
    'sdd': (SDD_CLS_NAMES, SDDDataset, SDD_ROOT),
    'tn3k': (TN3K_CLS_NAMES, TN3KDataset, TN3K_ROOT),
    'visa': (VISA_CLS_NAMES, VisaDataset, VISA_ROOT),
}

def get_data(dataset_type_list, transform, target_transform, training):
    if not isinstance(dataset_type_list, list):
        dataset_type_list = [dataset_type_list]

    dataset_cls_names_list = []
    dataset_instance_list = []
    dataset_root_list = []
    for dataset_type in dataset_type_list:
        if dataset_dict.get(dataset_type, ''):
            dataset_cls_names, dataset_instance, dataset_root = dataset_dict[dataset_type]
            dataset_instance = dataset_instance(
                clsnames=dataset_cls_names,
                transform=transform,
                target_transform=target_transform,
                training=training
            )

            dataset_cls_names_list.append(dataset_cls_names)
            dataset_instance_list.append(dataset_instance)
            dataset_root_list.append(dataset_root)

        else:
            print(f'Only support {list(dataset_dict.keys())}, but entered {dataset_type}...')
            raise NotImplementedError

    if len(dataset_type_list) > 1:
        dataset_instance = ConcatDataset(dataset_instance_list)
        dataset_cls_names = dataset_cls_names_list
        dataset_root = dataset_root_list
    else:
        dataset_instance = dataset_instance_list[0]
        dataset_cls_names = dataset_cls_names_list[0]
        dataset_root = dataset_root_list[0]

    return dataset_cls_names, dataset_instance, dataset_root