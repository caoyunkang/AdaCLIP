from typing import Union, List, Optional
import json
import numpy as np
import torch
from pkg_resources import packaging
from torch import nn
from torch.nn import functional as F
from .clip_model import CLIP
from .simple_tokenizer import SimpleTokenizer as _Tokenizer
from .image_prompt_bank import merge_abnormal_prompt_lists
from sklearn.cluster import KMeans

class ProjectLayer(nn.Module):
    def __init__(self, input_dim, output_dim, num_replicas, stack=False, is_array=True):
        super(ProjectLayer, self).__init__()

        self.head = nn.ModuleList([nn.Linear(input_dim, output_dim) for _ in range(num_replicas)])
        self.num_replicas = num_replicas
        self.stack = stack
        self.is_array = is_array

    def forward(self, tokens):
        out_tokens = []
        for i in range(self.num_replicas):
            if self.is_array:
                temp = self.head[i](tokens[i][:, 1:, :]) # for ViT, we exclude the class token and only extract patch tokens here.
            else:
                temp = self.head[i](tokens)

            out_tokens.append(temp)

        if self.stack:
            out_tokens = torch.stack(out_tokens, dim=1)

        return out_tokens

class PromptLayer(nn.Module):
    def __init__(self, channel, length, depth, is_text, prompting_type, enabled=True):
        super(PromptLayer, self).__init__()

        self.channel = channel
        self.length = length
        self.depth = depth
        self.is_text = is_text
        self.enabled = enabled

        self.prompting_type = prompting_type

        if self.enabled: # only when enabled, the parameters should be constructed
            if 'S' in prompting_type: # static prompts
                # learnable
                self.static_prompts = nn.ParameterList(
                    [nn.Parameter(torch.empty(self.length, self.channel))
                     for _ in range(self.depth)])

                for single_para in self.static_prompts:
                    nn.init.normal_(single_para, std=0.02)

            if 'D' in prompting_type: # dynamic prompts
                self.dynamic_prompts = [0.] # place holder

    def set_dynamic_prompts(self, dynamic_prompts):
        self.dynamic_prompts = dynamic_prompts

    def forward_text(self, resblock, indx, x, k_x=None, v_x=None, attn_mask: Optional[torch.Tensor] = None):
        if self.enabled:
            length = self.length

            # only prompt the first J layers
            if indx < self.depth:
                if 'S' in self.prompting_type and 'D' in self.prompting_type: # both
                    static_prompts = self.static_prompts[indx].unsqueeze(0).expand(x.shape[1], -1, -1)
                    textual_context = self.dynamic_prompts + static_prompts
                elif 'S' in self.prompting_type:  # static
                    static_prompts = self.static_prompts[indx].unsqueeze(0).expand(x.shape[1], -1, -1)
                    textual_context = static_prompts
                elif 'D' in self.prompting_type:  # dynamic
                    textual_context = self.dynamic_prompts
                else:
                    print('You should at least choose one type of prompts when the prompting branches are not none.')
                    raise NotImplementedError

            if indx == 0:  # for the first layer
                x = x
            else:
                if indx < self.depth:  # replace with learnalbe tokens
                    prefix = x[:1, :, :]
                    suffix = x[1 + length:, :, :]
                    textual_context = textual_context.permute(1, 0, 2).half()
                    x = torch.cat([prefix, textual_context, suffix], dim=0)
                else:  # keep the same
                    x = x
        else:
            x = x

        x, attn_tmp = resblock(q_x=x, k_x=k_x, v_x= v_x, attn_mask=attn_mask)

        return x, attn_tmp

    def forward_visual(self, resblock, indx, x, k_x=None, v_x=None, attn_mask: Optional[torch.Tensor] = None):
        if self.enabled:
            length = self.length

            # only prompt the first J layers
            if indx < self.depth:
                if 'S' in self.prompting_type and 'D' in self.prompting_type: # both
                    static_prompts = self.static_prompts[indx].unsqueeze(0).expand(x.shape[1], -1, -1)
                    visual_context = self.dynamic_prompts + static_prompts
                elif 'S' in self.prompting_type:  # static
                    static_prompts = self.static_prompts[indx].unsqueeze(0).expand(x.shape[1], -1, -1)
                    visual_context = static_prompts
                elif 'D' in self.prompting_type:  # dynamic
                    visual_context = self.dynamic_prompts
                else:
                    print('You should at least choose one type of prompts when the prompting branches are not none.')
                    raise NotImplementedError


            if indx == 0:  # for the first layer
                visual_context = visual_context.permute(1, 0, 2).half()
                x = torch.cat([x, visual_context], dim=0)
            else:
                if indx < self.depth:  # replace with learnalbe tokens
                    prefix = x[0:x.shape[0] - length, :, :]
                    visual_context = visual_context.permute(1, 0, 2).half()
                    x = torch.cat([prefix, visual_context], dim=0)
                else:  # keep the same
                    x = x
        else:
            x = x

        x, attn_tmp = resblock(q_x=x, k_x=k_x, v_x= v_x, attn_mask=attn_mask)

        if self.enabled:
            tokens = x[:x.shape[0] - length, :, :]
        else:
            tokens = x

        return x, tokens, attn_tmp

    def forward(self, resblock, indx, x, k_x=None, v_x=None, attn_mask: Optional[torch.Tensor] = None):
        if self.is_text:
            return self.forward_text(resblock, indx, x, k_x, v_x, attn_mask)
        else:
            return self.forward_visual(resblock, indx, x, k_x, v_x, attn_mask)


class TextEmbebddingLayer(nn.Module):
    def __init__(self, fixed):
        super(TextEmbebddingLayer, self).__init__()
        self.tokenizer = _Tokenizer()
        self.ensemble_text_features = {}
        self.prompt_normal = ['{}', 'flawless {}', 'perfect {}', 'unblemished {}', '{} without flaw',
                              '{} without defect',
                              '{} without damage']
        self.prompt_abnormal = ['damaged {}', 'broken {}', '{} with flaw', '{} with defect', '{} with damage']
        self.prompt_state = [self.prompt_normal, self.prompt_abnormal]
        self.prompt_templates = ['a bad photo of a {}.',
                                 'a low resolution photo of the {}.',
                                 'a bad photo of the {}.',
                                 'a cropped photo of the {}.',
                                 ]
        self.fixed = fixed

    def tokenize(self, texts: Union[str, List[str]], context_length: int = 77, truncate: bool = False) -> Union[
        torch.IntTensor, torch.LongTensor]:
        if isinstance(texts, str):
            texts = [texts]

        sot_token = self.tokenizer.encoder["<|startoftext|>"]
        eot_token = self.tokenizer.encoder["<|endoftext|>"]
        all_tokens = [[sot_token] + self.tokenizer.encode(text) + [eot_token] for text in texts]
        if packaging.version.parse(torch.__version__) < packaging.version.parse("1.8.0"):
            result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)
        else:
            result = torch.zeros(len(all_tokens), context_length, dtype=torch.int)

        for i, tokens in enumerate(all_tokens):
            if len(tokens) > context_length:
                if truncate:
                    tokens = tokens[:context_length]
                    tokens[-1] = eot_token
                else:
                    raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
            result[i, :len(tokens)] = torch.tensor(tokens)

        return result

    @staticmethod
    def _dedupe_prompt_list(prompts):
        cleaned = []
        seen = set()
        for prompt in prompts:
            prompt = str(prompt).strip()
            if not prompt or prompt in seen:
                continue
            seen.add(prompt)
            cleaned.append(prompt)
        return cleaned

    def _build_default_prompts(self, text, prompt_states):
        text = text.replace('-', ' ')
        prompted_state = [state.format(text) for state in prompt_states]
        prompted_sentence = []
        for state in prompted_state:
            for template in self.prompt_templates:
                prompted_sentence.append(template.format(state))
        return prompted_sentence

    def build_default_normal_prompts(self, text):
        return self._build_default_prompts(text, self.prompt_normal)

    def build_default_abnormal_prompts(self, text):
        return self._build_default_prompts(text, self.prompt_abnormal)

    @staticmethod
    def _single_image_path(image_paths):
        if isinstance(image_paths, (list, tuple)):
            if len(image_paths) != 1:
                raise ValueError(
                    "Per-image JSON abnormal prompts currently require batch_size=1. "
                    f"Received batch_size={len(image_paths)}"
                )
            return image_paths[0]
        return image_paths

    def _resolve_abnormal_prompts(
            self,
            text,
            image_path,
            abnormal_prompt_source,
            abnormal_prompt_mode,
            abnormal_prompt_fallback,
            abnormal_prompt_bank,
    ):
        if abnormal_prompt_source == "default":
            return None, None
        if abnormal_prompt_source != "json":
            raise ValueError(f"Unsupported abnormal_prompt_source: {abnormal_prompt_source}")
        if abnormal_prompt_mode not in ("replace", "append"):
            raise ValueError("abnormal_prompt_mode must be 'replace' or 'append'")
        if abnormal_prompt_fallback not in ("default", "error"):
            raise ValueError("abnormal_prompt_fallback must be 'default' or 'error'")
        if abnormal_prompt_bank is None:
            raise ValueError("abnormal_prompt_bank is required when abnormal_prompt_source='json'")
        if image_path is None:
            raise ValueError("image_path is required when abnormal_prompt_source='json'")

        prompt_info = abnormal_prompt_bank.get_difference_prompts(text, image_path)
        if prompt_info["found"]:
            return prompt_info["prompts"], prompt_info
        if abnormal_prompt_fallback == "default":
            return None, prompt_info
        raise KeyError(
            "Could not resolve JSON difference_prompts for "
            f"class_name={text}, image_path={image_path}, relative_key={prompt_info['relative_key']}"
        )

    def _log_image_prompt_resolution(self, prompt_bank, mode, fallback_used, final_prompt_count):
        if prompt_bank is None:
            return
        count = getattr(prompt_bank, "_resolved_debug_count", 0)
        if count >= getattr(prompt_bank, "debug_limit", 0):
            return
        setattr(prompt_bank, "_resolved_debug_count", count + 1)
        print(f"mode: {mode}")
        print(f"fallback_used: {fallback_used}")
        print(f"final_abnormal_prompt_count: {final_prompt_count}")

    ## TODO: text layeer is not compitable with multiple batches...
    def forward(
            self,
            model,
            texts,
            device,
            image_paths=None,
            abnormal_prompt_source="default",
            abnormal_prompt_mode="replace",
            abnormal_prompt_fallback="default",
            abnormal_prompt_bank=None,
    ):
        text_feature_list = []

        for indx, text in enumerate(texts):
            image_path = None
            abnormal_prompts = None
            prompt_info = None

            if abnormal_prompt_source == "json":
                if len(texts) != 1:
                    raise ValueError(
                        "Per-image JSON abnormal prompts currently require batch_size=1. "
                        f"Received batch_size={len(texts)}"
                    )
                image_path = self._single_image_path(image_paths)
                abnormal_prompts, prompt_info = self._resolve_abnormal_prompts(
                    text,
                    image_path,
                    abnormal_prompt_source,
                    abnormal_prompt_mode,
                    abnormal_prompt_fallback,
                    abnormal_prompt_bank,
                )
                text_features = self.encode_text(
                    model,
                    text,
                    device,
                    abnormal_prompts=abnormal_prompts,
                    abnormal_prompt_mode=abnormal_prompt_mode,
                )
                final_prompt_count = len(
                    self.resolve_final_abnormal_prompt_list(text, abnormal_prompts, abnormal_prompt_mode)
                )
                self._log_image_prompt_resolution(
                    abnormal_prompt_bank,
                    abnormal_prompt_mode,
                    prompt_info["fallback_used"] if prompt_info else False,
                    final_prompt_count,
                )
            elif self.fixed:
                if self.ensemble_text_features.get(text) is None:
                    text_features = self.encode_text(model, text, device)
                    self.ensemble_text_features[text] = text_features
                else:
                    text_features = self.ensemble_text_features[text]
            else:
                text_features = self.encode_text(model, text, device)
                self.ensemble_text_features[text] = text_features

            text_feature_list.append(text_features)

        text_features = torch.stack(text_feature_list, dim=0)
        text_features = F.normalize(text_features, dim=1)

        return text_features

    def _encode_prompt_list(self, model, prompt_list, device):
        prompted_sentence = self.tokenize(prompt_list, context_length=77).to(device)
        class_embeddings = model.encode_text(prompted_sentence)
        class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
        class_embedding = class_embeddings.mean(dim=0)
        class_embedding = class_embedding / class_embedding.norm()
        return class_embedding

    def resolve_final_abnormal_prompt_list(self, text, abnormal_prompts=None, abnormal_prompt_mode="replace"):
        default_prompts = self.build_default_abnormal_prompts(text)
        if abnormal_prompts is None:
            return default_prompts

        abnormal_prompts = self._dedupe_prompt_list(abnormal_prompts)
        return merge_abnormal_prompt_lists(default_prompts, abnormal_prompts, abnormal_prompt_mode)

    def encode_text(self, model, text, device, abnormal_prompts=None, abnormal_prompt_mode="replace"):
        text_features = []
        normal_prompt_list = self.build_default_normal_prompts(text)
        abnormal_prompt_list = self.resolve_final_abnormal_prompt_list(
            text,
            abnormal_prompts=abnormal_prompts,
            abnormal_prompt_mode=abnormal_prompt_mode,
        )
        for prompt_list in (normal_prompt_list, abnormal_prompt_list):
            text_features.append(self._encode_prompt_list(model, prompt_list, device))

        text_features = torch.stack(text_features, dim=1)

        return text_features


class JsonPromptTextEmbeddingLayer(nn.Module):
    def __init__(self, fixed, prompt_json_path, prompt_fallback="default"):
        super(JsonPromptTextEmbeddingLayer, self).__init__()
        if prompt_fallback not in ("default", "error"):
            raise ValueError("--prompt_fallback must be 'default' or 'error'")
        if not prompt_json_path:
            raise ValueError("--prompt_json_path is required when --prompt_source json")

        self.fixed = fixed
        self.prompt_json_path = prompt_json_path
        self.prompt_fallback = prompt_fallback
        self.default_layer = TextEmbebddingLayer(fixed=fixed)
        self.ensemble_text_features = {}
        self._printed_classes = set()

        self.prompt_bank = self._load_prompt_bank(prompt_json_path)

        self._normalized_prompt_bank = {
            self._normalize_class_name(class_name): prompts
            for class_name, prompts in self.prompt_bank.items()
        }

    @staticmethod
    def _normalize_class_name(class_name):
        return str(class_name).replace("-", " ").replace("_", " ").strip().lower()

    def _load_prompt_bank(self, prompt_json_path):
        with open(prompt_json_path, "r", encoding="utf-8") as handle:
            content = handle.read()

        try:
            prompt_data = json.loads(content)
        except json.JSONDecodeError:
            prompt_data = self._load_multiple_json_objects(content, prompt_json_path)

        prompt_bank = {}
        self._merge_prompt_records(prompt_data, prompt_bank)
        if not prompt_bank:
            raise ValueError(
                "Prompt JSON must contain class-keyed prompts, records with class_name/name/class "
                f"and normal_prompts, or LLaVA rows with class_name and response: {prompt_json_path}"
            )
        return prompt_bank

    @staticmethod
    def _load_multiple_json_objects(content, prompt_json_path):
        decoder = json.JSONDecoder()
        records = []
        idx = 0
        length = len(content)
        while idx < length:
            while idx < length and content[idx].isspace():
                idx += 1
            if idx >= length:
                break
            try:
                record, idx = decoder.raw_decode(content, idx)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Failed to parse prompt JSON/JSONL near char {idx}: {prompt_json_path}"
                ) from exc
            records.append(record)
        return records

    @staticmethod
    def _as_prompt_list(value):
        if value is None:
            return []
        if isinstance(value, str):
            value = value.strip()
            return [value] if value else []
        if isinstance(value, list):
            prompts = []
            for item in value:
                prompts.extend(JsonPromptTextEmbeddingLayer._as_prompt_list(item))
            return prompts
        return []

    @staticmethod
    def _strip_json_code_fence(text):
        text = text.strip()
        if not text.startswith("```"):
            return text

        lines = text.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        return "\n".join(lines).strip()

    @classmethod
    def _normal_prompts_from_response(cls, response):
        prompts = []
        for item in cls._as_prompt_list(response):
            text = cls._strip_json_code_fence(item)
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                prompts.extend(cls._as_prompt_list(item))
                continue

            if isinstance(parsed, dict):
                nested_prompts = cls._as_prompt_list(parsed.get("normal_prompts"))
                if nested_prompts:
                    prompts.extend(nested_prompts)
                else:
                    prompts.extend(cls._as_prompt_list(item))
            elif isinstance(parsed, list):
                for parsed_item in parsed:
                    if isinstance(parsed_item, dict):
                        prompts.extend(cls._as_prompt_list(parsed_item.get("normal_prompts")))
                    else:
                        prompts.extend(cls._as_prompt_list(parsed_item))
        return prompts

    @staticmethod
    def _add_prompt_record(prompt_bank, class_name, normal_prompts, abnormal_prompts=None):
        normal_prompts = JsonPromptTextEmbeddingLayer._as_prompt_list(normal_prompts)
        abnormal_prompts = JsonPromptTextEmbeddingLayer._as_prompt_list(abnormal_prompts)
        if not normal_prompts and not abnormal_prompts:
            return

        if class_name not in prompt_bank:
            prompt_bank[class_name] = {"normal_prompts": [], "abnormal_prompts": []}

        prompt_bank[class_name]["normal_prompts"].extend(normal_prompts)
        prompt_bank[class_name]["abnormal_prompts"].extend(abnormal_prompts)

    def _merge_prompt_records(self, record, prompt_bank):
        if isinstance(record, list):
            for item in record:
                self._merge_prompt_records(item, prompt_bank)
            return

        if not isinstance(record, dict):
            return

        if "normal_prompts" in record:
            class_name = record.get("class_name") or record.get("name") or record.get("class")
            if class_name is None:
                raise ValueError("Prompt records with normal_prompts must include class_name, name, or class")
            self._add_prompt_record(
                prompt_bank,
                class_name,
                record.get("normal_prompts"),
                record.get("abnormal_prompts"),
            )
            return

        class_name = record.get("class_name") or record.get("name") or record.get("class")
        if class_name is not None:
            mode = str(record.get("mode", "")).lower()
            if "abnormal" not in mode:
                normal_prompts = (
                    record.get("normal_prompt")
                    or record.get("normal")
                    or record.get("normal_prompts")
                    or record.get("responses")
                )
                if normal_prompts is None and record.get("response") is not None:
                    normal_prompts = self._normal_prompts_from_response(record.get("response"))
                self._add_prompt_record(
                    prompt_bank,
                    class_name,
                    normal_prompts,
                    record.get("abnormal_prompt") or record.get("abnormal"),
                )
            return

        for class_name, prompts in record.items():
            if isinstance(prompts, dict):
                prompts = dict(prompts)
                prompts.setdefault("class_name", class_name)
                self._merge_prompt_records(prompts, prompt_bank)
            elif isinstance(prompts, (str, list)):
                self._add_prompt_record(prompt_bank, class_name, prompts)

    def _get_prompts(self, class_name):
        normalized = self._normalize_class_name(class_name)
        prompts = self._normalized_prompt_bank.get(normalized)
        if prompts is None:
            if self.prompt_fallback == "default":
                return None
            raise KeyError(f"Class '{class_name}' is missing in prompt JSON: {self.prompt_json_path}")

        normal_prompts = prompts.get("normal_prompts", [])
        abnormal_prompts = prompts.get("abnormal_prompts", None)
        if not normal_prompts:
            if self.prompt_fallback == "default":
                return None
            raise ValueError(
                f"Class '{class_name}' requires non-empty normal_prompts "
                f"in prompt JSON: {self.prompt_json_path}"
            )
        return normal_prompts, abnormal_prompts

    def _build_default_abnormal_prompts(self, text):
        text = text.replace("-", " ").replace("_", " ")
        prompted_state = [state.format(text) for state in self.default_layer.prompt_abnormal]
        prompted_sentence = []
        for state in prompted_state:
            for template in self.default_layer.prompt_templates:
                prompted_sentence.append(template.format(state))
        return prompted_sentence

    def _encode_prompt_list(self, model, prompt_list, device):
        tokenized = self.default_layer.tokenize(prompt_list, context_length=77).to(device)
        class_embeddings = model.encode_text(tokenized)
        class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
        class_embedding = class_embeddings.mean(dim=0)
        class_embedding = class_embedding / class_embedding.norm()
        return class_embedding

    def _should_use_cache(self, model):
        # Dynamic text prompts are image-dependent, so class-only cached features would be stale.
        dynamic_text_prompt = (
            getattr(model, "enable_text_prompt", False)
            and "D" in getattr(model, "prompting_type", "")
        )
        return self.fixed or not dynamic_text_prompt

    def forward(self, model, texts, device):
        text_feature_list = []
        use_cache = self._should_use_cache(model)

        for text in texts:
            if use_cache and self.ensemble_text_features.get(text) is not None:
                text_features = self.ensemble_text_features[text]
            else:
                text_features = self.encode_text(model, text, device)
                self.ensemble_text_features[text] = text_features
            text_feature_list.append(text_features)

        text_features = torch.stack(text_feature_list, dim=0)
        text_features = F.normalize(text_features, dim=1)
        return text_features

    def encode_text(self, model, text, device):
        prompt_pair = self._get_prompts(text)
        if prompt_pair is None:
            if text not in self._printed_classes:
                print(f"[JsonPrompt] {text}: missing prompts, fallback to default TextEmbebddingLayer")
                self._printed_classes.add(text)
            return self.default_layer.encode_text(model, text, device)

        normal_prompts, abnormal_prompts = prompt_pair
        if abnormal_prompts is None or len(abnormal_prompts) == 0:
            abnormal_prompts = self._build_default_abnormal_prompts(text)
            abnormal_source = "default abnormal prompts"
        else:
            abnormal_source = "json abnormal prompts"

        if text not in self._printed_classes:
            print(
                f"[JsonPrompt] {text}: {len(normal_prompts)} normal prompts, "
                f"{len(abnormal_prompts)} abnormal prompts ({abnormal_source})"
            )
            self._printed_classes.add(text)

        text_features = [
            self._encode_prompt_list(model, normal_prompts, device),
            self._encode_prompt_list(model, abnormal_prompts, device),
        ]

        text_features = torch.stack(text_features, dim=1)
        return text_features


# Note: the implementation of HSF is slightly different to the reported one, since we found that the upgraded one is more stable.
class HybridSemanticFusion(nn.Module):
    def __init__(self, k_clusters):
        super(HybridSemanticFusion, self).__init__()
        self.k_clusters = k_clusters
        self.n_aggregate_patch_tokens = k_clusters * 5
        self.cluster_performer = KMeans(n_clusters=self.k_clusters, n_init="auto")

    # @torch.no_grad()
    def forward(self, patch_tokens: list, anomaly_maps: list):
        anomaly_map = torch.mean(torch.stack(anomaly_maps, dim=1), dim=1)
        anomaly_map = torch.softmax(anomaly_map, dim=2)[:, :, 1] # B, L

        # extract most abnormal feats
        selected_abnormal_tokens = []
        k = min(anomaly_map.shape[1], self.n_aggregate_patch_tokens)
        top_k_indices = torch.topk(anomaly_map, k=k, dim=1).indices
        for layer in range(len(patch_tokens)):
            selected_tokens = patch_tokens[layer]. \
                gather(dim=1, index=top_k_indices.unsqueeze(-1).
                       expand(-1, -1, patch_tokens[layer].shape[-1]))
            selected_abnormal_tokens.append(selected_tokens)

        # use kmeans to extract these centriods
        # Stack the data_preprocess
        stacked_data = torch.cat(selected_abnormal_tokens, dim=2)

        batch_cluster_centers = []
        # Perform K-Means clustering
        for b in range(stacked_data.shape[0]):
            cluster_labels = self.cluster_performer.fit_predict(stacked_data[b, :, :].detach().cpu().numpy())

            # Initialize a list to store the cluster centers
            cluster_centers = []

            # Extract cluster centers for each cluster
            for cluster_id in range(self.k_clusters):
                collected_cluster_data = []
                for abnormal_tokens in selected_abnormal_tokens:
                    cluster_data = abnormal_tokens[b, :, :][cluster_labels == cluster_id]
                    collected_cluster_data.append(cluster_data)
                collected_cluster_data = torch.cat(collected_cluster_data, dim=0)
                cluster_center = torch.mean(collected_cluster_data, dim=0, keepdim=True)
                cluster_centers.append(cluster_center)

            # Normalize the cluster centers
            cluster_centers = torch.cat(cluster_centers, dim=0)
            cluster_centers = torch.mean(cluster_centers, dim=0)
            batch_cluster_centers.append(cluster_centers)

        batch_cluster_centers = torch.stack(batch_cluster_centers, dim=0)
        batch_cluster_centers = F.normalize(batch_cluster_centers, dim=1)

        return batch_cluster_centers

        # # preprocess
        # # compute the anomaly map
        # anomaly_map = torch.mean(torch.stack(anomaly_maps, dim=1), dim=1)
        # anomaly_map = torch.softmax(anomaly_map, dim=2)[:, :, 1] # B, L
        #
        # # compute the average multi-hierarchy patch embeddings
        # avg_patch_tokens = torch.mean(torch.stack(patch_tokens, dim=0), dim=0) # B, L, C
        #
        # # Initialize a list to store the centroids of clusters with the largest anomaly scores
        # cluster_centroids = []
        #
        # # loop across the batch size
        # for b in range(avg_patch_tokens.shape[0]):
        #     # step1: group features into clusters
        #     cluster_labels = self.cluster_performer.fit_predict(avg_patch_tokens[b, :, :].detach().cpu().numpy())
        #
        #     # step2: compute the anomaly scores for individual clusters via the anomaly map
        #     # Convert cluster labels back to tensor
        #     cluster_labels = torch.tensor(cluster_labels).to(avg_patch_tokens.device)
        #     cluster_anomaly_scores = {}
        #     for label in torch.unique(cluster_labels):
        #         cluster_indices = torch.where(cluster_labels == label)[0]
        #         cluster_anomaly_scores[label.item()] = anomaly_map[b, cluster_indices].mean().item()
        #
        #     # step3: select the cluster with the largest anomaly score and then compute its centroid by averaging the
        #     # corresponding avg_patch_tokens
        #     # Find the cluster with the largest anomaly score
        #     largest_anomaly_cluster = max(cluster_anomaly_scores, key=cluster_anomaly_scores.get)
        #
        #     # Get the indices of the tokens belonging to the largest anomaly cluster
        #     largest_anomaly_cluster_indices = torch.where(cluster_labels == largest_anomaly_cluster)[0]
        #
        #     # Compute the centroid of the largest anomaly cluster by averaging the corresponding avg_patch_tokens
        #     centroid = avg_patch_tokens[b, largest_anomaly_cluster_indices, :].mean(dim=0)
        #
        #     # Append the centroid to the list of cluster centroids
        #     cluster_centroids.append(centroid)
        #
        # # Convert the list of centroids to a tensor
        # cluster_centroids = torch.stack(cluster_centroids, dim=0)
        # cluster_centroids = F.normalize(cluster_centroids, dim=1)

        # return cluster_centroids

class AdaCLIP(nn.Module):
    def __init__(self, freeze_clip: CLIP, text_channel: int, visual_channel: int,
                 prompting_length: int, prompting_depth: int, prompting_branch: str, prompting_type: str,
                 use_hsf: bool, k_clusters: int,
                 output_layers: list, device: str, image_size: int,
                 prompt_source: str = "default", prompt_json_path: str = "", prompt_fallback: str = "default"):
        super(AdaCLIP, self).__init__()
        self.freeze_clip = freeze_clip

        self.visual = self.freeze_clip.visual
        self.transformer = self.freeze_clip.transformer
        self.token_embedding = self.freeze_clip.token_embedding
        self.positional_embedding = self.freeze_clip.positional_embedding
        self.ln_final = self.freeze_clip.ln_final
        self.text_projection = self.freeze_clip.text_projection
        self.attn_mask = self.freeze_clip.attn_mask

        self.output_layers = output_layers

        self.prompting_branch = prompting_branch
        self.prompting_type = prompting_type
        self.prompting_depth = prompting_depth
        self.prompting_length = prompting_length
        self.use_hsf = use_hsf
        self.k_clusters = k_clusters

        if 'L' in self.prompting_branch:
            self.enable_text_prompt = True
        else:
            self.enable_text_prompt = False

        if 'V' in self.prompting_branch:
            self.enable_visual_prompt = True
        else:
            self.enable_visual_prompt = False

        self.prompt_source = prompt_source
        self.prompt_json_path = prompt_json_path
        self.prompt_fallback = prompt_fallback
        fixed_text_layer = (not self.enable_text_prompt)
        self.default_text_embedding_layer = TextEmbebddingLayer(fixed=fixed_text_layer)
        if prompt_source == "default":
            self.text_embedding_layer = self.default_text_embedding_layer
        elif prompt_source == "json":
            self.text_embedding_layer = JsonPromptTextEmbeddingLayer(
                fixed=fixed_text_layer,
                prompt_json_path=prompt_json_path,
                prompt_fallback=prompt_fallback,
            )
        else:
            raise ValueError(f"Unsupported prompt_source: {prompt_source}")
        self.text_prompter = PromptLayer(text_channel, prompting_length, prompting_depth, is_text=True,
                                         prompting_type=prompting_type,
                                         enabled=self.enable_text_prompt)
        self.visual_prompter = PromptLayer(visual_channel, prompting_length, prompting_depth, is_text=False,
                                           prompting_type=prompting_type,
                                           enabled=self.enable_visual_prompt)

        self.patch_token_layer = ProjectLayer(
            visual_channel,
            text_channel,
            len(output_layers), stack=False, is_array=True
        )

        self.cls_token_layer = ProjectLayer(
            text_channel,
            text_channel,
            1, stack=False, is_array=False
        )

        if 'D' in self.prompting_type: # dynamic prompts
            self.dynamic_visual_prompt_generator = ProjectLayer(text_channel,
                                                                visual_channel,
                                                                prompting_length,
                                                                stack=True,
                                                                is_array=False)
            self.dynamic_text_prompt_generator = ProjectLayer(text_channel,
                                                              text_channel,
                                                              prompting_length,
                                                              stack=True,
                                                              is_array=False)

        if self.use_hsf:
            self.HSF = HybridSemanticFusion(k_clusters)

        self.image_size = image_size
        self.device = device

    def generate_and_set_dynamic_promtps(self, image):
        with torch.no_grad():
            # extract image features
            image_features, _ = self.visual.forward(image, self.output_layers)

        dynamic_visual_prompts = self.dynamic_visual_prompt_generator(image_features)
        dynamic_text_prompts = self.dynamic_text_prompt_generator(image_features)

        self.visual_prompter.set_dynamic_prompts(dynamic_visual_prompts)
        self.text_prompter.set_dynamic_prompts(dynamic_text_prompts)


    def encode_image(self, image):

        x = image
        # to patches - whether to use dual patchnorm - https://arxiv.org/abs/2302.01327v1
        if self.visual.input_patchnorm:
            # einops - rearrange(x, 'b c (h p1) (w p2) -> b (h w) (c p1 p2)')
            x = x.reshape(x.shape[0], x.shape[1],
                          self.visual.grid_size[0],
                          self.visual.patch_size[0],
                          self.visual.grid_size[1],
                          self.visual.patch_size[1])
            x = x.permute(0, 2, 4, 1, 3, 5)
            x = x.reshape(x.shape[0], self.visual.grid_size[0] * self.visual.grid_size[1], -1)
            x = self.visual.patchnorm_pre_ln(x)
            x = self.visual.conv1(x)
        else:
            x = self.visual.conv1(x)  # shape = [*, width, grid, grid]
            x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
            x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]

        # class embeddings and positional embeddings
        x = torch.cat(
            [self.visual.class_embedding.to(x.dtype) +
             torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
             x], dim=1)  # shape = [*, grid ** 2 + 1, width]

        x = x + self.visual.positional_embedding.to(x.dtype)

        # a patch_dropout of 0. would mean it is disabled and this function would do nothing but return what was passed in
        x = self.visual.patch_dropout(x)
        x = self.visual.ln_pre(x)

        patch_embedding = x

        x = x.permute(1, 0, 2)  # NLD -> LND

        patch_tokens = []

        for indx, r in enumerate(self.visual.transformer.resblocks):
            x, tokens, attn_tmp = self.visual_prompter(r, indx, x, k_x=None, v_x=None, attn_mask=None)

            if (indx + 1) in self.output_layers:
                patch_tokens.append(tokens)

        x = x.permute(1, 0, 2)  # LND -> NLD
        patch_tokens = [patch_tokens[t].permute(1, 0, 2) for t in range(len(patch_tokens))]  # LND -> NLD

        if self.visual.attn_pool is not None:
            x = self.visual.attn_pool(x)
            x = self.visual.ln_post(x)
            pooled, tokens = self.visual._global_pool(x)
        else:
            pooled, tokens = self.visual._global_pool(x)
            pooled = self.visual.ln_post(pooled)

        if self.visual.proj is not None:
            pooled = pooled @ self.visual.proj

        return pooled, patch_tokens, patch_embedding

    def proj_visual_tokens(self, image_features, patch_tokens):

        # for patch tokens
        proj_patch_tokens = self.patch_token_layer(patch_tokens)
        for layer in range(len(proj_patch_tokens)):
            #proj_patch_tokens[layer] /= proj_patch_tokens[layer].norm(dim=-1, keepdim=True)
            proj_patch_tokens[layer] = proj_patch_tokens[layer] / proj_patch_tokens[layer].norm(dim=-1, keepdim=True)

        # for cls tokens
        proj_cls_tokens = self.cls_token_layer(image_features)[0]
        proj_cls_tokens = proj_cls_tokens / proj_cls_tokens.norm(dim=-1, keepdim=True)

        return proj_cls_tokens, proj_patch_tokens

    def encode_text(self, text):
        cast_dtype = self.transformer.get_cast_dtype()

        x = self.token_embedding(text).to(cast_dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.to(cast_dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND

        for indx, r in enumerate(self.transformer.resblocks):
            # add prompt here
            x, attn_tmp = self.text_prompter(r, indx, x, k_x=None, v_x=None, attn_mask=self.attn_mask)

        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)  # [batch_size, n_ctx, transformer.width]

        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return x

    def visual_text_similarity(self, image_feature, patch_token, text_feature, aggregation):
        anomaly_maps = []

        for layer in range(len(patch_token)):
            anomaly_map = (100.0 * patch_token[layer] @ text_feature)
            anomaly_maps.append(anomaly_map)

        if self.use_hsf:
            alpha = 0.2
            clustered_feature = self.HSF.forward(patch_token, anomaly_maps)
            # aggregate the class token and the clustered features for more comprehensive information
            cur_image_feature = alpha * clustered_feature + (1 - alpha) * image_feature
            cur_image_feature = F.normalize(cur_image_feature, dim=1)
        else:
            cur_image_feature = image_feature

        anomaly_score = (100.0 * cur_image_feature.unsqueeze(1) @ text_feature)
        anomaly_score = anomaly_score.squeeze(1)
        anomaly_score = torch.softmax(anomaly_score, dim=1)

        # NOTE: this bilinear interpolation is not unreproducible and may occasionally lead to unstable ZSAD performance.
        for i in range(len(anomaly_maps)):
            B, L, C = anomaly_maps[i].shape
            H = int(np.sqrt(L))
            anomaly_maps[i] = anomaly_maps[i].permute(0, 2, 1).view(B, 2, H, H)
            anomaly_maps[i] = F.interpolate(anomaly_maps[i], size=self.image_size, mode='bilinear', align_corners=True)

        if aggregation: # in the test stage, we firstly aggregate logits from all hierarchies and then do the softmax normalization
            anomaly_map = torch.mean(torch.stack(anomaly_maps, dim=1), dim=1)
            anomaly_map = torch.softmax(anomaly_map, dim=1)
            anomaly_map = (anomaly_map[:, 1:, :, :] + 1 - anomaly_map[:, 0:1, :, :]) / 2.0
            anomaly_score = anomaly_score[:, 1]
            return anomaly_map, anomaly_score
        else: # otherwise, we do the softmax normalization for individual hierarchies
            for i in range(len(anomaly_maps)):
                anomaly_maps[i] = torch.softmax(anomaly_maps[i], dim=1)
            return anomaly_maps, anomaly_score

    def extract_feat(
            self,
            image,
            cls_name,
            image_path=None,
            abnormal_prompt_source="default",
            abnormal_prompt_mode="replace",
            abnormal_prompt_fallback="default",
            abnormal_prompt_bank=None,
    ):
        if 'D' in self.prompting_type:
            self.generate_and_set_dynamic_promtps(image) # generate and set dynamic prompts for corresponding prompters

        if self.enable_visual_prompt:
            image_features, patch_tokens, _ = self.encode_image(image)
        else:
            with torch.no_grad():
                image_features, patch_tokens, _ = self.encode_image(image)

        if abnormal_prompt_source == "json":
            text_layer = self.default_text_embedding_layer
            text_kwargs = dict(
                image_paths=image_path,
                abnormal_prompt_source=abnormal_prompt_source,
                abnormal_prompt_mode=abnormal_prompt_mode,
                abnormal_prompt_fallback=abnormal_prompt_fallback,
                abnormal_prompt_bank=abnormal_prompt_bank,
            )
        else:
            text_layer = self.text_embedding_layer
            text_kwargs = {}

        if self.enable_text_prompt:
            text_features = text_layer(self, cls_name, self.device, **text_kwargs)
        else:
            with torch.no_grad():
                text_features = text_layer(self, cls_name, self.device, **text_kwargs)

        proj_cls_tokens, proj_patch_tokens = self.proj_visual_tokens(image_features, patch_tokens)

        return proj_cls_tokens, proj_patch_tokens, text_features

    @torch.cuda.amp.autocast()
    def forward(
            self,
            image,
            cls_name,
            aggregation=True,
            image_path=None,
            abnormal_prompt_source="default",
            abnormal_prompt_mode="replace",
            abnormal_prompt_fallback="default",
            abnormal_prompt_bank=None,
    ):
        # extract features for images and texts
        image_features, patch_tokens, text_features = self.extract_feat(
            image,
            cls_name,
            image_path=image_path,
            abnormal_prompt_source=abnormal_prompt_source,
            abnormal_prompt_mode=abnormal_prompt_mode,
            abnormal_prompt_fallback=abnormal_prompt_fallback,
            abnormal_prompt_bank=abnormal_prompt_bank,
        )
        anomaly_map, anomaly_score = self.visual_text_similarity(image_features, patch_tokens, text_features, aggregation)

        if aggregation:
            anomaly_map = anomaly_map # tensor
            anomaly_score = anomaly_score
            anomaly_map = anomaly_map.squeeze(1)

            return anomaly_map, anomaly_score
        else:
            anomaly_maps = anomaly_map # list
            anomaly_score = anomaly_score

            return anomaly_maps, anomaly_score

