import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import exp

class FocalLoss(nn.Module):
    """
    copy from: https://github.com/Hsuxu/Loss_ToolBox-PyTorch/blob/master/FocalLoss/FocalLoss.py
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
        Focal_Loss= -1*alpha*(1-pt)*log(pt)
    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                    focus on hard misclassified example
    :param smooth: (float,double) smooth value when cross entropy
    :param balance_index: (int) balance class index, should be specific when alpha is float
    :param size_average: (bool, optional) By default, the losses are averaged over each loss element in the batch.
    """

    def __init__(self, apply_nonlin=None, alpha=None, gamma=2, balance_index=0, smooth=1e-5, size_average=True):
        super(FocalLoss, self).__init__()
        self.apply_nonlin = apply_nonlin
        self.alpha = alpha
        self.gamma = gamma
        self.balance_index = balance_index
        self.smooth = smooth
        self.size_average = size_average

        if self.smooth is not None:
            if self.smooth < 0 or self.smooth > 1.0:
                raise ValueError('smooth value should be in [0,1]')

    def forward(self, logit, target):
        if self.apply_nonlin is not None:
            logit = self.apply_nonlin(logit)
        num_class = logit.shape[1]

        if logit.dim() > 2:
            # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            logit = logit.view(logit.size(0), logit.size(1), -1)
            logit = logit.permute(0, 2, 1).contiguous()
            logit = logit.view(-1, logit.size(-1))
        target = torch.squeeze(target, 1)
        target = target.view(-1, 1)
        alpha = self.alpha

        if alpha is None:
            alpha = torch.ones(num_class, 1)
        elif isinstance(alpha, (list, np.ndarray)):
            assert len(alpha) == num_class
            alpha = torch.FloatTensor(alpha).view(num_class, 1)
            alpha = alpha / alpha.sum()
        elif isinstance(alpha, float):
            alpha = torch.ones(num_class, 1)
            alpha = alpha * (1 - self.alpha)
            alpha[self.balance_index] = self.alpha

        else:
            raise TypeError('Not support alpha type')

        if alpha.device != logit.device:
            alpha = alpha.to(logit.device)

        idx = target.cpu().long()

        one_hot_key = torch.FloatTensor(target.size(0), num_class).zero_()
        one_hot_key = one_hot_key.scatter_(1, idx, 1)
        if one_hot_key.device != logit.device:
            one_hot_key = one_hot_key.to(logit.device)

        if self.smooth:
            one_hot_key = torch.clamp(
                one_hot_key, self.smooth / (num_class - 1), 1.0 - self.smooth)
        pt = (one_hot_key * logit).sum(1) + self.smooth
        logpt = pt.log()

        gamma = self.gamma

        alpha = alpha[idx]
        alpha = torch.squeeze(alpha)
        loss = -1 * alpha * torch.pow((1 - pt), gamma) * logpt

        if self.size_average:
            loss = loss.mean()
        return loss


class BinaryDiceLoss(nn.Module):
    def __init__(self):
        super(BinaryDiceLoss, self).__init__()

    def forward(self, input, targets):
        # 获取每个批次的大小 N
        N = targets.size()[0]
        # 平滑变量
        smooth = 1
        # 将宽高 reshape 到同一纬度
        input_flat = input.view(N, -1)
        targets_flat = targets.view(N, -1)

        # 计算交集
        intersection = input_flat * targets_flat
        N_dice_eff = (2 * intersection.sum(1) + smooth) / (input_flat.sum(1) + targets_flat.sum(1) + smooth)
        # 计算一个批次中平均每张图的损失
        loss = 1 - N_dice_eff.sum() / N
        return loss




class ConADLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, contrast_mode='all',random_anchors=10):
        super(ConADLoss, self).__init__()
        assert contrast_mode in ['all', 'mean', 'random']
        self.contrast_mode = contrast_mode
        self.random_anchors = random_anchors
    def forward(self, features, labels):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, C, ...].
            labels: ground truth of shape [bsz, 1, ...]., where 1 denotes to abnormal, and 0 denotes to normal
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))
        if len(features.shape) != len(labels.shape):
            raise ValueError('`features` needs to have the same dimensions with labels')

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, C, ...],'
                             'at least 3 dimensions are required')

        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)
            labels = labels.view(labels.shape[0], labels.shape[1], -1)

        labels = labels.squeeze()
        batch_size = features.shape[0]

        C = features.shape[1]
        normal_feats = features[:, :, labels == 0]
        abnormal_feats = features[:, :, labels == 1]

        normal_feats = normal_feats.permute((1, 0, 2)).contiguous().view(C, -1)
        abnormal_feats = abnormal_feats.permute((1, 0, 2)).contiguous().view(C, -1)

        contrast_count = normal_feats.shape[1]
        contrast_feature = normal_feats

        if self.contrast_mode == 'mean':
            anchor_feature = torch.mean(normal_feats, dim=1)
            anchor_feature = F.normalize(anchor_feature, dim=0, p=2)
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        elif self.contrast_mode == 'random':
            dim_to_sample = 1
            num_samples = min(self.random_anchors, contrast_count)
            permuted_indices = torch.randperm(normal_feats.size(dim_to_sample)).to(normal_feats.device)
            selected_indices = permuted_indices[:num_samples]
            anchor_feature = normal_feats.index_select(dim_to_sample, selected_indices)
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        # maximize similarity
        anchor_dot_normal = torch.matmul(anchor_feature.T, normal_feats).mean()

        # minimize similarity
        anchor_dot_abnormal = torch.matmul(anchor_feature.T, abnormal_feats).mean()

        loss = 0
        if normal_feats.shape[1] > 0:
            loss -= anchor_dot_normal
        if abnormal_feats.shape[1] > 0:
            loss += anchor_dot_abnormal

        loss = torch.exp(loss)

        return loss
