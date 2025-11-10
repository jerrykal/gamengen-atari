import os

import accelerate
import torch
from accelerate.state import AcceleratorState
from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    UNet2DConditionModel,
)
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import FromOriginalModelMixin
from diffusers.models.modeling_utils import ModelMixin
from torch import nn
from transformers.utils import ContextManagers


class ActionEmbeddingModel(ModelMixin, ConfigMixin, FromOriginalModelMixin):
    @register_to_config
    def __init__(self, num_embeddings: int, embedding_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)

    def forward(self, action_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(action_ids)


def get_models(
    pretrained_model_name_or_path: str,
    context_length: int,
    num_noise_buckets: int,
    action_dim: int,
    finetuned_vae_name_or_path: str | None = None,
    revision: str | None = None,
    variant: str | None = None,
) -> tuple[UNet2DConditionModel, AutoencoderKL, DDIMScheduler, ActionEmbeddingModel]:
    # Use DDIM scheduler and v-prediction following the GameNGen paper
    noise_scheduler = DDIMScheduler.from_pretrained(pretrained_model_name_or_path, subfolder="scheduler")
    noise_scheduler.register_to_config(prediction_type="v_prediction")

    def deepspeed_zero_init_disabled_context_manager():
        """
        returns either a context list that includes one that will disable zero.Init or an empty context list
        """
        deepspeed_plugin = AcceleratorState().deepspeed_plugin if accelerate.state.is_initialized() else None
        if deepspeed_plugin is None:
            return []

        return [deepspeed_plugin.zero3_init_context_manager(enable=False)]

    # Currently Accelerate doesn't know how to handle multiple models under Deepspeed ZeRO stage 3.
    # For this to work properly all models must be run through `accelerate.prepare`. But accelerate
    # will try to assign the same optimizer with the same weights to all models during
    # `deepspeed.initialize`, which of course doesn't work.
    #
    # For now the following workaround will partially support Deepspeed ZeRO-3, by excluding the 2
    # frozen models from being partitioned during `zero.Init` which gets called during
    # `from_pretrained` So CLIPTextModel and AutoencoderKL will not enjoy the parameter sharding
    # across multiple gpus and only UNet2DConditionModel will get ZeRO sharded.
    with ContextManagers(deepspeed_zero_init_disabled_context_manager()):
        if finetuned_vae_name_or_path is None:
            vae = AutoencoderKL.from_pretrained(
                pretrained_model_name_or_path,
                subfolder="vae",
                revision=revision,
                variant=variant,
            )
        else:
            vae = AutoencoderKL.from_pretrained(finetuned_vae_name_or_path)

    unet = UNet2DConditionModel.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="unet",
        revision=revision,
    )

    # Replace the text encoder with an action embedding model, replace the cross-attention
    # from text to action sequences.
    try:
        action_embedding = ActionEmbeddingModel.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="action_embedding",
            revision=revision,
        )
    except Exception:
        # NOTE: If an action embedding model is not found, it is assumed to be a standard
        #       stable diffusion model. Consequently, the unet model is modified to incorporate
        #       the architectural changes following the GameNGen paper.
        action_embedding = ActionEmbeddingModel(
            num_embeddings=action_dim, embedding_dim=unet.config.cross_attention_dim
        )

        # Modify the input channels to accommodate past observations
        new_in_channels = unet.config.in_channels * (context_length + 1)
        unet.register_to_config(in_channels=new_in_channels)
        unet.conv_in = nn.Conv2d(
            new_in_channels,
            unet.conv_in.out_channels,
            kernel_size=unet.conv_in.kernel_size,
            padding=unet.conv_in.padding,
        )

        # Add class_embedding for noise augmentation
        if num_noise_buckets > 0:
            unet.register_to_config(num_class_embeds=num_noise_buckets)
            unet.class_embedding = nn.Embedding(num_noise_buckets, unet.time_embedding.linear_2.out_features)

    return unet, vae, noise_scheduler, action_embedding


def save_models(
    unet: UNet2DConditionModel,
    vae: AutoencoderKL,
    noise_scheduler: DDIMScheduler,
    action_embedding: ActionEmbeddingModel,
    output_dir: str,
) -> None:
    unet.save_pretrained(os.path.join(output_dir, "unet"))
    vae.save_pretrained(os.path.join(output_dir, "vae"))
    noise_scheduler.save_pretrained(os.path.join(output_dir, "scheduler"))
    action_embedding.save_pretrained(os.path.join(output_dir, "action_embedding"))
