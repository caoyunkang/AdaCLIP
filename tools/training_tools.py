import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import os
import random
import torch
import numpy as np


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_paths(args):
    save_root = args.save_path
    model_root = os.path.join(save_root, 'models')
    log_root = os.path.join(save_root, 'logs')
    csv_root = os.path.join(save_root, 'csvs')
    image_root = os.path.join(save_root, 'images')
    tensorboard_root = os.path.join(save_root, 'tensorboard')

    os.makedirs(model_root, exist_ok=True)
    os.makedirs(log_root, exist_ok=True)
    os.makedirs(csv_root, exist_ok=True)
    os.makedirs(image_root, exist_ok=True)
    os.makedirs(tensorboard_root, exist_ok=True)

    if args.use_hsf:
        # prepare model name
        model_name = f'{args.exp_indx}s-pretrained-{args.training_data}-{args.model}-' \
                     f'{args.prompting_type}-{args.prompting_branch}-' \
                     f'D{args.prompting_depth}-L{args.prompting_length}-HSF-K{args.k_clusters}'
    else:
        # prepare model name
        model_name = f'{args.exp_indx}s-pretrained-{args.training_data}-{args.model}-' \
                     f'{args.prompting_type}-{args.prompting_branch}-' \
                     f'D{args.prompting_depth}-L{args.prompting_length}-WO-HSF'


    # prepare model path
    ckp_path = os.path.join(model_root, model_name)

    # prepare tensorboard dir
    tensorboard_dir = os.path.join(tensorboard_root, f'{model_name}-{args.testing_data}')
    if os.path.exists(tensorboard_dir):
        import shutil
        shutil.rmtree(tensorboard_dir)
    tensorboard_logger = SummaryWriter(log_dir=tensorboard_dir)

    # prepare csv path
    csv_path = os.path.join(csv_root, f'{model_name}-{args.testing_data}.csv')

    # prepare image path
    image_dir = os.path.join(image_root, f'{model_name}-{args.testing_data}')
    os.makedirs(image_dir, exist_ok=True)

    # prepare log path
    log_path = os.path.join(log_root, f'{model_name}-{args.testing_data}.txt')

    return model_name, image_dir, csv_path, log_path, ckp_path, tensorboard_logger


