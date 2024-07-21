import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns

##
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

##
import matplotlib.ticker as mtick


def plot_sample_cv2(names, imgs, scores_: dict, gts, save_folder=None):
    os.makedirs(save_folder, exist_ok=True)

    # get subplot number
    total_number = len(imgs)

    scores = scores_.copy()
    # normarlisze anomalies
    for k, v in scores.items():
        max_value = np.max(v)
        min_value = np.min(v)

        scores[k] = (scores[k] - min_value) / max_value * 255
        scores[k] = scores[k].astype(np.uint8)
    # draw gts
    mask_imgs = []
    for idx in range(total_number):
        gts_ = gts[idx]
        mask_imgs_ = imgs[idx].copy()
        mask_imgs_[gts_ > 0.5] = (0, 0, 255)
        mask_imgs.append(mask_imgs_)

    # save imgs
    for idx in range(total_number):

        cv2.imwrite(os.path.join(save_folder, f'{names[idx]}_ori.jpg'), imgs[idx])
        cv2.imwrite(os.path.join(save_folder, f'{names[idx]}_gt.jpg'), mask_imgs[idx])

        for key in scores:
            heat_map = cv2.applyColorMap(scores[key][idx], cv2.COLORMAP_JET)
            visz_map = cv2.addWeighted(heat_map, 0.5, imgs[idx], 0.5, 0)
            cv2.imwrite(os.path.join(save_folder, f'{names[idx]}_{key}.jpg'),
                        visz_map)




def plot_feat_cv2(names, feat, save_folder=None):
    # get subplot number
    total_number = len(feat)

    # save imgs
    for idx in range(total_number):
        feat[idx] = cv2.resize(feat[idx], (256, 256), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(os.path.join(save_folder, f'{names[idx]}_feat.jpg'), feat[idx])



valid_feature_visualization_methods = ['TSNE', 'PCA']

def visualize_feature(features, labels, legends, n_components=3, method='TSNE'):
    assert method in valid_feature_visualization_methods
    assert n_components in [2, 3]

    if method == 'TSNE':
        model = TSNE(n_components=n_components)
    elif method == 'PCA':
        model = PCA(n_components=n_components)

    else:
        raise NotImplementedError

    feat_proj = model.fit_transform(features)

    if n_components == 2:
        ax = scatter_2d(feat_proj, labels)
    elif n_components == 3:
        ax = scatter_3d(feat_proj, labels)
    else:
        raise NotImplementedError

    plt.legend(legends)
    plt.axis('off')


def scatter_3d(feat_proj, label):
    plt.clf()
    ax1 = plt.axes(projection='3d')

    label_unique = np.unique(label)

    for l in label_unique:
        ax1.scatter3D(feat_proj[label == l, 0],
                      feat_proj[label == l, 1],
                      feat_proj[label == l, 2], s=5)

    return ax1


def scatter_2d(feat_proj, label):
    plt.clf()
    ax1 = plt.axes()

    label_unique = np.unique(label)

    for l in label_unique:
        ax1.scatter(feat_proj[label == l, 0],
                    feat_proj[label == l, 1], s=5)

    return ax1
