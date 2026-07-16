import math

import torch


PROMPT_MODES = ("hybrid", "static_only", "dynamic_only")


def _tokens_to_feature_map(tokens):
    if tokens.dim() != 3:
        raise ValueError(f"Expected prompted patch tokens [B,N,C], got {tuple(tokens.shape)}")

    num_tokens = tokens.shape[1]
    grid = int(math.sqrt(num_tokens))
    if grid * grid != num_tokens:
        patch_tokens = num_tokens - 1
        patch_grid = int(math.sqrt(patch_tokens))
        if patch_grid * patch_grid != patch_tokens:
            raise ValueError(
                "Prompted patch token count must be square after class-token removal, "
                f"got N={num_tokens}"
            )
        tokens = tokens[:, 1:, :]
        grid = patch_grid

    batch, num_patches, channels = tokens.shape
    if grid * grid != num_patches:
        raise ValueError(f"Prompted patch token count must be square, got N={num_patches}")
    return tokens.transpose(1, 2).reshape(batch, channels, grid, grid).contiguous()


def _prompt_type_for_mode(original_prompt_type, prompt_mode):
    if prompt_mode not in PROMPT_MODES:
        raise ValueError(f"Unsupported prompt_mode: {prompt_mode}. Expected one of {PROMPT_MODES}.")
    if prompt_mode == "hybrid":
        return original_prompt_type
    if prompt_mode == "static_only":
        if "S" not in original_prompt_type:
            raise ValueError("prompt_mode='static_only' requires static prompts in prompting_type.")
        return "S"
    if "D" not in original_prompt_type:
        raise ValueError("prompt_mode='dynamic_only' requires dynamic prompts in prompting_type.")
    return "D"


def _capture_prompt_types(self):
    return (
        self.prompting_type,
        self.visual_prompter.prompting_type,
        self.text_prompter.prompting_type,
    )


def _set_prompt_types(self, prompt_type):
    self.prompting_type = prompt_type
    self.visual_prompter.prompting_type = prompt_type
    self.text_prompter.prompting_type = prompt_type


def _restore_prompt_types(self, prompt_types):
    self.prompting_type, self.visual_prompter.prompting_type, self.text_prompter.prompting_type = prompt_types


def _extract_prompted_features(self, images, layers=None, return_projected=False, prompt_mode="hybrid"):
    """
    Return prompted visual patch features for selected ViT blocks.

    layers are 1-indexed transformer block numbers. return_projected=False returns
    projection-pre patch tokens from encode_image. return_projected=True returns the
    patch_token_layer outputs and is kept for later experiments.

    prompt_mode controls only this feature-extraction call:
      hybrid: original AdaCLIP static + dynamic prompts
      static_only: static prompts only, without generating dynamic prompts
      dynamic_only: generated dynamic prompts only
    """
    layers = list(layers or self.output_layers)
    if not layers:
        raise ValueError("layers must contain at least one 1-indexed transformer block number.")

    previous_output_layers = self.output_layers
    previous_prompt_types = _capture_prompt_types(self)
    prompt_type = _prompt_type_for_mode(self.prompting_type, prompt_mode)
    self.output_layers = layers
    _set_prompt_types(self, prompt_type)
    try:
        if "D" in prompt_type:
            self.generate_and_set_dynamic_promtps(images)

        if self.enable_visual_prompt:
            image_features, patch_tokens, _ = self.encode_image(images)
        else:
            with torch.no_grad():
                image_features, patch_tokens, _ = self.encode_image(images)

        if len(patch_tokens) != len(layers):
            raise RuntimeError(
                f"Expected {len(layers)} prompted feature levels, got {len(patch_tokens)}. "
                f"layers={layers}"
            )

        if return_projected:
            _, patch_tokens = self.proj_visual_tokens(image_features, patch_tokens)

        return {
            f"layer{layer}": _tokens_to_feature_map(tokens)
            for layer, tokens in zip(layers, patch_tokens)
        }
    finally:
        self.output_layers = previous_output_layers
        _restore_prompt_types(self, previous_prompt_types)


def install_prompted_feature_api(adaclip_cls):
    if not hasattr(adaclip_cls, "extract_prompted_features"):
        adaclip_cls.extract_prompted_features = _extract_prompted_features
    return adaclip_cls
