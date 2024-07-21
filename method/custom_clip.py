# This file is largely borrowed from open clip
import hashlib
import json
import logging
import os
import re
import urllib
import warnings
from copy import deepcopy
from dataclasses import dataclass, asdict
from functools import partial
from pathlib import Path
from typing import Any, Optional, Tuple
from typing import Dict, Union
from typing import List
import torch
import torch.nn as nn
import torchvision.transforms.functional as F
from torchvision.transforms import Normalize, Compose, RandomResizedCrop, InterpolationMode, ToTensor, Resize, \
    CenterCrop
from tqdm import tqdm
from .clip_model import CLIP, convert_to_custom_text_state_dict, \
    resize_pos_embed
from .clip_model import build_model_from_openai_state_dict, convert_weights_to_lp, get_cast_dtype
from .tokenizer import HFTokenizer, tokenize

__version__ = '2.16.0'

try:
    from huggingface_hub import hf_hub_download

    hf_hub_download = partial(hf_hub_download, library_name="open_clip", library_version=__version__)
    _has_hf_hub = True
except ImportError:
    hf_hub_download = None
    _has_hf_hub = False


def _pcfg(url='', hf_hub='', mean=None, std=None):
    return dict(
        url=url,
        hf_hub=hf_hub,
        mean=mean,
        std=std,
    )


_VITB32 = dict(
    openai=_pcfg(
        "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt"),
    laion400m_e31=_pcfg(
        "https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/vit_b_32-quickgelu-laion400m_e31-d867053b.pt"),
    laion400m_e32=_pcfg(
        "https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/vit_b_32-quickgelu-laion400m_e32-46683a32.pt"),
    laion2b_e16=_pcfg(
        "https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/vit_b_32-laion2b_e16-af8dbd0c.pth"),
    laion2b_s34b_b79k=_pcfg(hf_hub='laion/CLIP-ViT-B-32-laion2B-s34B-b79K/')
)


_VITB16 = dict(
    openai=_pcfg(
        "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt"),
    laion400m_e31=_pcfg(
        "https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/vit_b_16-laion400m_e31-00efa78f.pt"),
    laion400m_e32=_pcfg(
        "https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/vit_b_16-laion400m_e32-55e67d44.pt"),
    laion2b_s34b_b88k=_pcfg(hf_hub='laion/CLIP-ViT-B-16-laion2B-s34B-b88K/'),
)

_VITL14 = dict(
    openai=_pcfg(
        "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt"),
    laion400m_e31=_pcfg(
        "https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/vit_l_14-laion400m_e31-69988bb6.pt"),
    laion400m_e32=_pcfg(
        "https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/vit_l_14-laion400m_e32-3d133497.pt"),
    laion2b_s32b_b82k=_pcfg(
        hf_hub='laion/CLIP-ViT-L-14-laion2B-s32B-b82K/',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
)

_VITL14_336 = dict(
    openai=_pcfg(
        "https://openaipublic.azureedge.net/clip/models/3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02/ViT-L-14-336px.pt"),
)

_VITH14 = dict(
    laion2b_s32b_b79k=_pcfg(hf_hub='laion/CLIP-ViT-H-14-laion2B-s32B-b79K/'),
)

_VITg14 = dict(
    laion2b_s12b_b42k=_pcfg(hf_hub='laion/CLIP-ViT-g-14-laion2B-s12B-b42K/'),
    laion2b_s34b_b88k=_pcfg(hf_hub='laion/CLIP-ViT-g-14-laion2B-s34B-b88K/'),
)

_VITbigG14 = dict(
    laion2b_s39b_b160k=_pcfg(hf_hub='laion/CLIP-ViT-bigG-14-laion2B-39B-b160k/'),
)



_PRETRAINED = {
    "ViT-B-32": _VITB32,
    "ViT-B-16": _VITB16,
    "ViT-L-14": _VITL14,
    "ViT-L-14-336": _VITL14_336,
    "ViT-H-14": _VITH14,
    "ViT-g-14": _VITg14,
    "ViT-bigG-14": _VITbigG14,
}


def _clean_tag(tag: str):
    # normalize pretrained tags
    return tag.lower().replace('-', '_')


def list_pretrained(as_str: bool = False):
    """ returns list of pretrained models
    Returns a tuple (model_name, pretrain_tag) by default or 'name:tag' if as_str == True
    """
    return [':'.join([k, t]) if as_str else (k, t) for k in _PRETRAINED.keys() for t in _PRETRAINED[k].keys()]


def list_pretrained_models_by_tag(tag: str):
    """ return all models having the specified pretrain tag """
    models = []
    tag = _clean_tag(tag)
    for k in _PRETRAINED.keys():
        if tag in _PRETRAINED[k]:
            models.append(k)
    return models


def list_pretrained_tags_by_model(model: str):
    """ return all pretrain tags for the specified model architecture """
    tags = []
    if model in _PRETRAINED:
        tags.extend(_PRETRAINED[model].keys())
    return tags


def is_pretrained_cfg(model: str, tag: str):
    if model not in _PRETRAINED:
        return False
    return _clean_tag(tag) in _PRETRAINED[model]


def get_pretrained_cfg(model: str, tag: str):
    if model not in _PRETRAINED:
        return {}
    model_pretrained = _PRETRAINED[model]
    if 'openai' in model_pretrained.keys():
        tag = 'openai'
    else:
        tag = list(model_pretrained.keys())[0]
    print('*' * 50)
    print(f'Use pretrained model from {tag}...')
    print('*' * 50)
    return model_pretrained.get(_clean_tag(tag), {})


def get_pretrained_url(model: str, tag: str):
    cfg = get_pretrained_cfg(model, _clean_tag(tag))
    return cfg.get('url', '')


def download_pretrained_from_url(
        url: str,
        cache_dir: Union[str, None] = None,
):
    if not cache_dir:
        cache_dir = os.path.expanduser("~/.cache/clip")
    os.makedirs(cache_dir, exist_ok=True)
    filename = os.path.basename(url)

    if 'openaipublic' in url:
        expected_sha256 = url.split("/")[-2]
    elif 'mlfoundations' in url:
        expected_sha256 = os.path.splitext(filename)[0].split("-")[-1]
    else:
        expected_sha256 = ''

    download_target = os.path.join(cache_dir, filename)

    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f"{download_target} exists and is not a regular file")

    if os.path.isfile(download_target):
        if expected_sha256:
            if hashlib.sha256(open(download_target, "rb").read()).hexdigest().startswith(expected_sha256):
                return download_target
            else:
                warnings.warn(
                    f"{download_target} exists, but the SHA256 checksum does not match; re-downloading the file")
        else:
            return download_target

    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        with tqdm(total=int(source.headers.get("Content-Length")), ncols=80, unit='iB', unit_scale=True) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break

                output.write(buffer)
                loop.update(len(buffer))

    if expected_sha256 and not hashlib.sha256(open(download_target, "rb").read()).hexdigest().startswith(
            expected_sha256):
        raise RuntimeError(f"Model has been downloaded but the SHA256 checksum does not not match")

    return download_target


def has_hf_hub(necessary=False):
    if not _has_hf_hub and necessary:
        # if no HF Hub module installed, and it is necessary to continue, raise error
        raise RuntimeError(
            'Hugging Face hub model specified but package not installed. Run `pip install huggingface_hub`.')
    return _has_hf_hub


def download_pretrained_from_hf(
        model_id: str,
        filename: str = 'open_clip_pytorch_model.bin',
        revision=None,
        cache_dir: Union[str, None] = None,
):
    has_hf_hub(True)
    cached_file = hf_hub_download(model_id, filename, revision=revision, cache_dir=cache_dir)
    return cached_file


def download_pretrained(
        cfg: Dict,
        force_hf_hub: bool = False,
        cache_dir: Union[str, None] = None,
):
    target = ''
    if not cfg:
        return target

    download_url = cfg.get('url', '')
    download_hf_hub = cfg.get('hf_hub', '')
    if download_hf_hub and force_hf_hub:
        # use HF hub even if url exists
        download_url = ''

    if download_url:
        target = download_pretrained_from_url(download_url, cache_dir=cache_dir)
    elif download_hf_hub:
        has_hf_hub(True)
        # we assume the hf_hub entries in pretrained config combine model_id + filename in
        # 'org/model_name/filename.pt' form. To specify just the model id w/o filename and
        # use 'open_clip_pytorch_model.bin' default, there must be a trailing slash 'org/model_name/'.
        model_id, filename = os.path.split(download_hf_hub)
        if filename:
            target = download_pretrained_from_hf(model_id, filename=filename, cache_dir=cache_dir)
        else:
            target = download_pretrained_from_hf(model_id, cache_dir=cache_dir)

    return target


OPENAI_DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_DATASET_STD = (0.26862954, 0.26130258, 0.27577711)


@dataclass
class AugmentationCfg:
    scale: Tuple[float, float] = (0.9, 1.0)
    ratio: Optional[Tuple[float, float]] = None
    color_jitter: Optional[Union[float, Tuple[float, float, float]]] = None
    interpolation: Optional[str] = None
    re_prob: Optional[float] = None
    re_count: Optional[int] = None
    use_timm: bool = False


class ResizeMaxSize(nn.Module):

    def __init__(self, max_size, interpolation=InterpolationMode.BICUBIC, fn='max', fill=0):
        super().__init__()
        if not isinstance(max_size, int):
            raise TypeError(f"Size should be int. Got {type(max_size)}")
        self.max_size = max_size
        self.interpolation = interpolation
        self.fn = min if fn == 'min' else min
        self.fill = fill

    def forward(self, img):
        if isinstance(img, torch.Tensor):
            height, width = img.shape[:2]
        else:
            width, height = img.size
        scale = self.max_size / float(max(height, width))
        if scale != 1.0:
            new_size = tuple(round(dim * scale) for dim in (height, width))
            img = F.resize(img, new_size, self.interpolation)
            pad_h = self.max_size - new_size[0]
            pad_w = self.max_size - new_size[1]
            img = F.pad(img, padding=[pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2], fill=self.fill)
        return img


def _convert_to_rgb(image):
    return image.convert('RGB')


def image_transform(
        image_size: int,
        is_train: bool,
        mean: Optional[Tuple[float, ...]] = None,
        std: Optional[Tuple[float, ...]] = None,
        resize_longest_max: bool = False,
        fill_color: int = 0,
        aug_cfg: Optional[Union[Dict[str, Any], AugmentationCfg]] = None,
):
    mean = mean or OPENAI_DATASET_MEAN
    if not isinstance(mean, (list, tuple)):
        mean = (mean,) * 3

    std = std or OPENAI_DATASET_STD
    if not isinstance(std, (list, tuple)):
        std = (std,) * 3

    if isinstance(image_size, (list, tuple)) and image_size[0] == image_size[1]:
        # for square size, pass size as int so that Resize() uses aspect preserving shortest edge
        image_size = image_size[0]

    if isinstance(aug_cfg, dict):
        aug_cfg = AugmentationCfg(**aug_cfg)
    else:
        aug_cfg = aug_cfg or AugmentationCfg()
    normalize = Normalize(mean=mean, std=std)
    if is_train:
        aug_cfg_dict = {k: v for k, v in asdict(aug_cfg).items() if v is not None}
        use_timm = aug_cfg_dict.pop('use_timm', False)
        if use_timm:
            from timm.data import create_transform  # timm can still be optional
            if isinstance(image_size, (tuple, list)):
                assert len(image_size) >= 2
                input_size = (3,) + image_size[-2:]
            else:
                input_size = (3, image_size, image_size)
            # by default, timm aug randomly alternates bicubic & bilinear for better robustness at inference time
            aug_cfg_dict.setdefault('interpolation', 'random')
            aug_cfg_dict.setdefault('color_jitter', None)  # disable by default
            train_transform = create_transform(
                input_size=input_size,
                is_training=True,
                hflip=0.,
                mean=mean,
                std=std,
                re_mode='pixel',
                **aug_cfg_dict,
            )
        else:
            train_transform = Compose([
                RandomResizedCrop(
                    image_size,
                    scale=aug_cfg_dict.pop('scale'),
                    interpolation=InterpolationMode.BICUBIC,
                ),
                _convert_to_rgb,
                ToTensor(),
                normalize,
            ])
            if aug_cfg_dict:
                warnings.warn(
                    f'Unused augmentation cfg items, specify `use_timm` to use ({list(aug_cfg_dict.keys())}).')
        return train_transform
    else:
        if resize_longest_max:
            transforms = [
                ResizeMaxSize(image_size, fill=fill_color)
            ]
        else:
            transforms = [
                Resize(image_size, interpolation=InterpolationMode.BICUBIC),
                CenterCrop(image_size),
            ]
        transforms.extend([
            _convert_to_rgb,
            ToTensor(),
            normalize,
        ])
        return Compose(transforms)


def list_openai_models() -> List[str]:
    """Returns the names of available CLIP models"""
    return list_pretrained_models_by_tag('openai')


def load_openai_model(
        name: str,
        precision: Optional[str] = None,
        device: Optional[Union[str, torch.device]] = None,
        jit: bool = True,
        cache_dir: Optional[str] = None,
):
    """Load a CLIP model

    Parameters
    ----------
    name : str
        A model name listed by `clip.available_models()`, or the path to a model checkpoint containing the state_dict
    precision: str
        Model precision, if None defaults to 'fp32' if device == 'cpu' else 'fp16'.
    device : Union[str, torch.device]
        The device to put the loaded model
    jit : bool
        Whether to load the optimized JIT model (default) or more hackable non-JIT model.
    cache_dir : Optional[str]
        The directory to cache the downloaded model weights

    Returns
    -------
    model : torch.nn.Module
        The CLIP model
    preprocess : Callable[[PIL.Image], torch.Tensor]
        A torchvision transform that converts a PIL image into a tensor that the returned model can take as its input
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if precision is None:
        precision = 'fp32' if device == 'cpu' else 'fp16'

    cfg = get_pretrained_cfg(name, 'openai')
    if cfg:
        model_path = download_pretrained(cfg, cache_dir=cache_dir)
    elif os.path.isfile(name):
        model_path = name
    else:
        raise RuntimeError(f"Model {name} not found; available models = {list_pretrained()}")

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location=device if jit else "cpu").eval()
        state_dict = None
    except RuntimeError:
        # loading saved state dict
        if jit:
            warnings.warn(f"File {model_path} is not a JIT archive. Loading as a state dict instead")
            jit = False
        state_dict = torch.load(model_path, map_location="cpu")

    # JIT -> Just In Time
    if not jit:
        # Build a non-jit model from the OpenAI jitted model state dict
        cast_dtype = get_cast_dtype(precision)
        try:
            model = build_model_from_openai_state_dict(state_dict or model.state_dict(), cast_dtype=cast_dtype)
        except KeyError:
            sd = {k[7:]: v for k, v in state_dict["state_dict"].items()}
            model = build_model_from_openai_state_dict(sd, cast_dtype=cast_dtype)

        # model from OpenAI state dict is in manually cast fp16 mode, must be converted for AMP/fp32/bf16 use
        model = model.to(device)
        if precision.startswith('amp') or precision == 'fp32':
            model.float()
        elif precision == 'bf16':
            convert_weights_to_lp(model, dtype=torch.bfloat16)

        return model

    # patch the device names
    device_holder = torch.jit.trace(lambda: torch.ones([]).to(torch.device(device)), example_inputs=[])
    device_node = [n for n in device_holder.graph.findAllNodes("prim::Constant") if "Device" in repr(n)][-1]

    def patch_device(module):
        try:
            graphs = [module.graph] if hasattr(module, "graph") else []
        except RuntimeError:
            graphs = []

        if hasattr(module, "forward1"):
            graphs.append(module.forward1.graph)

        for graph in graphs:
            for node in graph.findAllNodes("prim::Constant"):
                if "value" in node.attributeNames() and str(node["value"]).startswith("cuda"):
                    node.copyAttributes(device_node)

    model.apply(patch_device)
    patch_device(model.encode_image)
    patch_device(model.encode_text)

    # patch dtype to float32 (typically for CPU)
    if precision == 'fp32':
        float_holder = torch.jit.trace(lambda: torch.ones([]).float(), example_inputs=[])
        float_input = list(float_holder.graph.findNode("aten::to").inputs())[1]
        float_node = float_input.node()

        def patch_float(module):
            try:
                graphs = [module.graph] if hasattr(module, "graph") else []
            except RuntimeError:
                graphs = []

            if hasattr(module, "forward1"):
                graphs.append(module.forward1.graph)

            for graph in graphs:
                for node in graph.findAllNodes("aten::to"):
                    inputs = list(node.inputs())
                    for i in [1, 2]:  # dtype can be the second or third argument to aten::to()
                        if inputs[i].node()["value"] == 5:
                            inputs[i].node().copyAttributes(float_node)

        model.apply(patch_float)
        patch_float(model.encode_image)
        patch_float(model.encode_text)
        model.float()

    # ensure image_size attr available at consistent location for both jit and non-jit
    model.visual.image_size = model.input_resolution.item()
    return model


HF_HUB_PREFIX = 'hf-hub:'
_MODEL_CONFIG_PATHS = [Path(__file__).parent.parent / f"./model_configs/"]
_MODEL_CONFIGS = {}  # directory (model_name: config) of model architecture configs


def _natural_key(string_):
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_.lower())]


def _rescan_model_configs():
    global _MODEL_CONFIGS

    config_ext = ('.json',)
    config_files = []
    for config_path in _MODEL_CONFIG_PATHS:
        if config_path.is_file() and config_path.suffix in config_ext:
            config_files.append(config_path)
        elif config_path.is_dir():
            for ext in config_ext:
                config_files.extend(config_path.glob(f'*{ext}'))

    for cf in config_files:
        with open(cf, 'r') as f:
            model_cfg = json.load(f)
            if all(a in model_cfg for a in ('embed_dim', 'vision_cfg', 'text_cfg')):
                _MODEL_CONFIGS[cf.stem] = model_cfg

    _MODEL_CONFIGS = {k: v for k, v in sorted(_MODEL_CONFIGS.items(), key=lambda x: _natural_key(x[0]))}


_rescan_model_configs()  # initial populate of model config registry


def list_models():
    """ enumerate available model architectures based on config files """
    return list(_MODEL_CONFIGS.keys())


def add_model_config(path):
    """ add model config path or file and update registry """
    if not isinstance(path, Path):
        path = Path(path)
    _MODEL_CONFIG_PATHS.append(path)
    _rescan_model_configs()


def get_model_config(model_name):
    if model_name in _MODEL_CONFIGS:
        return deepcopy(_MODEL_CONFIGS[model_name])
    else:
        return None


def get_tokenizer(model_name):
    if model_name.startswith(HF_HUB_PREFIX):
        tokenizer = HFTokenizer(model_name[len(HF_HUB_PREFIX):])
    else:
        config = get_model_config(model_name)
        tokenizer = HFTokenizer(
            config['text_cfg']['hf_tokenizer_name']) if 'hf_tokenizer_name' in config['text_cfg'] else tokenize
    return tokenizer


def load_state_dict(checkpoint_path: str, map_location='cpu'):
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    if next(iter(state_dict.items()))[0].startswith('module'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    return state_dict


def load_checkpoint(model, checkpoint_path, strict=True):
    state_dict = load_state_dict(checkpoint_path)
    # detect old format and make compatible with new format
    if 'positional_embedding' in state_dict and not hasattr(model, 'positional_embedding'):
        state_dict = convert_to_custom_text_state_dict(state_dict)
    resize_pos_embed(state_dict, model)
    incompatible_keys = model.load_state_dict(state_dict, strict=strict)
    return incompatible_keys


def create_model(
        model_name: str,
        img_size: int,
        pretrained: Optional[str] = None,
        precision: str = 'fp32',
        device: Union[str, torch.device] = 'cpu',
        jit: bool = False,
        cache_dir: Optional[str] = None,
        output_dict: Optional[bool] = None,
):
    if model_name.count('ViT') < 1:
        print('only support ViT model..')
        raise NotImplementedError

    # in which means, we can also use old naming rules.
    model_name = model_name.replace('/', '-')  # for callers using old naming with / in ViT names
    checkpoint_path = None
    pretrained_cfg = {}
    model_cfg = None

    if isinstance(device, str):
        device = torch.device(device)

    # our default version are borrowed from openai
    assert pretrained and pretrained.lower() == 'openai', 'only support openai module.'
    logging.info(f'Loading pretrained {model_name} from OpenAI.')
    model_cfg = model_cfg or get_model_config(model_name)

    model_cfg['vision_cfg']['image_size'] = img_size
    cast_dtype = get_cast_dtype(precision)

    model_pre = load_openai_model(
        model_name,
        precision=precision,
        device=device,
        jit=jit,
        cache_dir=cache_dir,
    )
    state_dict = model_pre.state_dict()

    # to always output dict even if it is clip
    if output_dict and hasattr(model_pre, "output_dict"):
        model_pre.output_dict = True

    model = CLIP(**model_cfg, cast_dtype=cast_dtype)

    # mainly need to resize the position embeddings
    resize_pos_embed(state_dict, model)
    incompatible_keys = model.load_state_dict(state_dict, strict=True)
    model.to(device=device)
    if precision in ("fp16", "bf16"):
        convert_weights_to_lp(model, dtype=torch.bfloat16 if precision == 'bf16' else torch.float16)

    # set image / mean metadata from pretrained_cfg if available, or use default
    model.visual.image_mean = pretrained_cfg.get('mean', None) or OPENAI_DATASET_MEAN
    model.visual.image_std = pretrained_cfg.get('std', None) or OPENAI_DATASET_STD

    # to always output dict even if it is clip
    if output_dict and hasattr(model, "output_dict"):
        model.output_dict = True

    if jit:
        model = torch.jit.script(model)

    return model


def create_model_and_transforms(
        model_name: str,
        img_size: int,
        pretrained: Optional[str] = None,
        precision: str = 'fp32',
        device: Union[str, torch.device] = 'cpu',
        jit: bool = False,
        image_mean: Optional[Tuple[float, ...]] = None,
        image_std: Optional[Tuple[float, ...]] = None,
        aug_cfg: Optional[Union[Dict[str, Any], AugmentationCfg]] = None,
        cache_dir: Optional[str] = None,
        output_dict: Optional[bool] = None,
):
    ######### create the clip model
    model = create_model(
        model_name,
        img_size,
        pretrained,
        precision=precision,
        device=device,
        jit=jit,
        cache_dir=cache_dir,
        output_dict=output_dict,
    )

    image_mean = image_mean or getattr(model.visual, 'image_mean', None)
    image_std = image_std or getattr(model.visual, 'image_std', None)
    preprocess_train = image_transform(
        model.visual.image_size,
        is_train=True,
        mean=image_mean,
        std=image_std,
        aug_cfg=aug_cfg,
    )
    preprocess_val = image_transform(
        model.visual.image_size,
        is_train=False,
        mean=image_mean,
        std=image_std,
    )

    return model, preprocess_train, preprocess_val
