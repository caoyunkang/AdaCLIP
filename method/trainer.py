import cv2
import torchvision.transforms as transforms
from scipy.ndimage import gaussian_filter

from loss import FocalLoss, BinaryDiceLoss
from tools import visualization, calculate_metric, calculate_average_metric
from .adaclip import *
from .custom_clip import create_model_and_transforms
from .image_prompt_bank import ImagePromptBank


class AdaCLIP_Trainer(nn.Module):
    def __init__(
            self,
            # clip-related
            backbone, feat_list, input_dim, output_dim,

            # learning-related
            learning_rate, device, image_size,

            # model settings
            prompting_depth=3, prompting_length=2,
            prompting_branch='VL', prompting_type='SD',
            use_hsf=True, k_clusters=20,
            prompt_source='default', prompt_json_path='', prompt_fallback='default',
            train_abnormal_prompt_source='default',
            test_abnormal_prompt_source='default',
            train_abnormal_prompt_json_path='',
            test_abnormal_prompt_json_path='',
            abnormal_prompt_mode='replace',
            abnormal_prompt_fallback='default',
            json_prompt_debug_limit=5,
    ):

        super(AdaCLIP_Trainer, self).__init__()

        self.device = device
        self.feat_list = feat_list
        self.image_size = image_size
        self.prompting_branch = prompting_branch
        self.prompting_type = prompting_type
        self.train_abnormal_prompt_source = train_abnormal_prompt_source
        self.test_abnormal_prompt_source = test_abnormal_prompt_source
        self.abnormal_prompt_mode = abnormal_prompt_mode
        self.abnormal_prompt_fallback = abnormal_prompt_fallback
        self.json_prompt_debug_limit = json_prompt_debug_limit
        self.train_abnormal_prompt_bank = self._build_image_prompt_bank(
            train_abnormal_prompt_source,
            train_abnormal_prompt_json_path,
            phase='train',
        )
        self.test_abnormal_prompt_bank = self._build_image_prompt_bank(
            test_abnormal_prompt_source,
            test_abnormal_prompt_json_path,
            phase='test',
        )

        self.loss_focal = FocalLoss()
        self.loss_dice = BinaryDiceLoss()

        ########### different model choices
        freeze_clip, _, self.preprocess = create_model_and_transforms(backbone, image_size,
                                                                      pretrained='openai')
        freeze_clip  = freeze_clip.to(device)
        freeze_clip.eval()

        self.clip_model = AdaCLIP(freeze_clip=freeze_clip,
                                  text_channel=output_dim,
                                  visual_channel=input_dim,
                                  prompting_length=prompting_length,
                                  prompting_depth=prompting_depth,
                                  prompting_branch=prompting_branch,
                                  prompting_type=prompting_type,
                                  use_hsf=use_hsf,
                                  k_clusters=k_clusters,
                                  output_layers=feat_list,
                                  device=device,
                                  image_size=image_size,
                                  prompt_source=prompt_source,
                                  prompt_json_path=prompt_json_path,
                                  prompt_fallback=prompt_fallback).to(device)

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor()
        ])

        self.preprocess.transforms[0] = transforms.Resize(size=(image_size, image_size),
                                                          interpolation=transforms.InterpolationMode.BICUBIC,
                                                          max_size=None)

        self.preprocess.transforms[1] = transforms.CenterCrop(size=(image_size, image_size))

        # update parameters
        self.learnable_paramter_list = [
            'text_prompter',
            'visual_prompter',
            'patch_token_layer',
            'cls_token_layer',
            'dynamic_visual_prompt_generator',
            'dynamic_text_prompt_generator'
        ]

        self.params_to_update = []
        for name, param in self.clip_model.named_parameters():
            # print(name)
            for update_name in self.learnable_paramter_list:
                if update_name in name:
                    # print(f'updated parameters--{name}: {update_name}')
                    self.params_to_update.append(param)

        # build the optimizer
        self.optimizer = torch.optim.AdamW(self.params_to_update, lr=learning_rate, betas=(0.5, 0.999))

    def _build_image_prompt_bank(self, source, json_path, phase):
        if source not in ('default', 'json'):
            raise ValueError(f"{phase}_abnormal_prompt_source must be 'default' or 'json'")
        if source == 'default':
            return None
        return ImagePromptBank(
            json_path=json_path,
            fallback=self.abnormal_prompt_fallback,
            debug_limit=self.json_prompt_debug_limit,
            phase=phase,
        )

    def save(self, path):
        self.save_dict = {}
        for param, value in self.state_dict().items():
            for update_name in self.learnable_paramter_list:
                if update_name in param:
                    # print(f'{param}: {update_name}')
                    self.save_dict[param] = value
                    break

        torch.save(self.save_dict, path)

    def load(self, path):
        self.load_state_dict(torch.load(path, map_location=self.device), strict=False)

    def train_one_batch(self, items):
        image = items['img'].to(self.device)
        cls_name = items['cls_name']
        image_path = items.get('img_path')

        # pixel level
        anomaly_map, anomaly_score = self.clip_model(
            image,
            cls_name,
            aggregation=False,
            image_path=image_path,
            abnormal_prompt_source=self.train_abnormal_prompt_source,
            abnormal_prompt_mode=self.abnormal_prompt_mode,
            abnormal_prompt_fallback=self.abnormal_prompt_fallback,
            abnormal_prompt_bank=self.train_abnormal_prompt_bank,
        )

        if not isinstance(anomaly_map, list):
            anomaly_map = [anomaly_map]

        # losses
        gt = items['img_mask'].to(self.device)
        gt = gt.squeeze()

        gt[gt > 0.5] = 1
        gt[gt <= 0.5] = 0

        is_anomaly = items['anomaly'].to(self.device)
        is_anomaly[is_anomaly > 0.5] = 1
        is_anomaly[is_anomaly <= 0.5] = 0
        loss = 0

        # classification loss
        classification_loss = self.loss_focal(anomaly_score, is_anomaly.unsqueeze(1))
        loss += classification_loss

        # seg loss
        seg_loss = 0
        for am, in zip(anomaly_map):
            seg_loss += (self.loss_focal(am, gt) + self.loss_dice(am[:, 1, :, :], gt) +
                         self.loss_dice(am[:, 0, :, :], 1-gt))

        loss += seg_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss

    def train_epoch(self, loader):
        self.clip_model.train()
        loss_list = []
        for items in loader:
            loss = self.train_one_batch(items)
            loss_list.append(loss.item())

        return np.mean(loss_list)

    @torch.no_grad()
    def evaluation(self, dataloader, obj_list, save_fig, save_fig_dir=None):
        self.clip_model.eval()

        results = {}
        results['cls_names'] = []
        results['imgs_gts'] = []
        results['anomaly_scores'] = []
        results['imgs_masks'] = []
        results['anomaly_maps'] = []
        results['imgs'] = []
        results['names'] = []

        with torch.no_grad(), torch.cuda.amp.autocast():
            image_indx = 0
            for indx, items in enumerate(dataloader):
                if save_fig:
                    path = items['img_path']
                    for _path in path:
                        vis_image = cv2.resize(cv2.imread(_path), (self.image_size, self.image_size))
                        results['imgs'].append(vis_image)
                    cls_name = items['cls_name']
                    for _cls_name in cls_name:
                        image_indx += 1
                        results['names'].append('{:}-{:03d}'.format(_cls_name, image_indx))

                image = items['img'].to(self.device)
                cls_name = items['cls_name']
                image_path = items.get('img_path')
                results['cls_names'].extend(cls_name)
                gt_mask = items['img_mask']
                gt_mask[gt_mask > 0.5], gt_mask[gt_mask <= 0.5] = 1, 0

                for _gt_mask in gt_mask:
                    results['imgs_masks'].append(_gt_mask.squeeze(0).numpy())  # px

                # pixel level
                anomaly_map, anomaly_score = self.clip_model(
                    image,
                    cls_name,
                    aggregation=True,
                    image_path=image_path,
                    abnormal_prompt_source=self.test_abnormal_prompt_source,
                    abnormal_prompt_mode=self.abnormal_prompt_mode,
                    abnormal_prompt_fallback=self.abnormal_prompt_fallback,
                    abnormal_prompt_bank=self.test_abnormal_prompt_bank,
                )

                anomaly_map = anomaly_map.cpu().numpy()
                anomaly_score = anomaly_score.cpu().numpy()

                for _anomaly_map, _anomaly_score in zip(anomaly_map, anomaly_score):
                    _anomaly_map = gaussian_filter(_anomaly_map, sigma=4)
                    results['anomaly_maps'].append(_anomaly_map)
                    results['anomaly_scores'].append(_anomaly_score)

                is_anomaly = np.array(items['anomaly'])
                for _is_anomaly in is_anomaly:
                    results['imgs_gts'].append(_is_anomaly)

        # visualization
        if save_fig:
            print('saving fig.....')
            visualization.plot_sample_cv2(
                results['names'],
                results['imgs'],
                {'AdaCLIP': results['anomaly_maps']},
                results['imgs_masks'],
                save_fig_dir
            )

        metric_dict = dict()
        for obj in obj_list:
            metric_dict[obj] = dict()

        for obj in obj_list:
            metric = calculate_metric(results, obj)
            obj_full_name = f'{obj}'
            metric_dict[obj_full_name] = metric

        metric_dict['Average'] = calculate_average_metric(metric_dict)

        return metric_dict

