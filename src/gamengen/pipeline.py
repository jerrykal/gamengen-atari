from dataclasses import dataclass

import numpy as np
import PIL
import torch
from diffusers.image_processor import VaeImageProcessor
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, ImagePipelineOutput
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils.torch_utils import randn_tensor
from einops import rearrange
from gamengen.models import ActionEmbeddingModel


@dataclass
class GameNGenPipelineOutput(ImagePipelineOutput):
    """
    Output class for GameNGen pipeline.

    Args:
        images: list[PIL.image.image] | np.ndarray
            List of denoised PIL new frame images of length `batch_size` or NumPy array of shape `(batch_size, height, width,
            num_channels)`.
        latents: torch.Tensor
            Latent space representation of the generated new frame images.
    """

    images: list[PIL.Image.Image] | np.ndarray
    latents: torch.Tensor


class GameNGenPipeline(DiffusionPipeline):
    def __init__(
        self,
        vae: AutoencoderKL,
        unet: UNet2DConditionModel,
        scheduler: KarrasDiffusionSchedulers,
        action_embedding: ActionEmbeddingModel,
    ) -> None:
        super().__init__()

        self.register_modules(
            vae=vae,
            unet=unet,
            scheduler=scheduler,
            action_embedding=action_embedding,
        )

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.expected_context_length = self.unet.config.in_channels // self.unet.config.out_channels - 1

    @property
    def guidance_scale(self) -> float:
        return self._guidance_scale

    @property
    def do_classifier_free_guidance(self) -> bool:
        return self._guidance_scale > 1 and self.unet.config.time_cond_proj_dim is None

    def encode_context_images(self, images: torch.Tensor) -> torch.Tensor:
        assert images.shape[1] == self.expected_context_length, (
            f"Context frame images must have {self.expected_context_length} context frames, but got {images.shape[1]}"
        )
        images = rearrange(images, "b l c h w -> (b l) c h w")
        latents = self.vae.encode(images).latent_dist.sample()
        latents = rearrange(latents, "(b l) c h w -> b l c h w", l=self.expected_context_length)
        latents = latents * self.vae.config.scaling_factor
        return latents

    def prepare_latents(
        self,
        batch_size: int,
        num_channels_latents: int,
        height: int,
        width: int,
        dtype: torch.dtype,
        device: torch.device,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        shape = (
            batch_size,
            1,
            num_channels_latents,
            height,
            width,
        )
        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    @torch.no_grad()
    def __call__(
        self,
        context_actions: torch.Tensor,
        context_latents: torch.Tensor | None = None,
        context_images: torch.Tensor | None = None,
        num_inference_steps: int = 4,
        guidance_scale: float = 1.5,
        generator: torch.Generator | None = None,
        output_type: str = "pil",
        return_dict: bool = True,
    ) -> ImagePipelineOutput | tuple:
        self._guidance_scale = guidance_scale

        # 1. Encode past frames
        # context_latents shape -> (batch_size, context_length, num_channels_latents, height, width)
        if (context_latents is None) ^ (context_images is not None):
            raise ValueError("Either context_latents or context_images must be provided, but not both")
        if context_images is not None:
            context_latents = self.encode_context_images(context_images)

        batch_size = context_latents.shape[0]
        context_length = context_latents.shape[1]
        height, width = context_latents.shape[-2:]
        device = self._execution_device

        assert context_length == self.expected_context_length, (
            f"Context length must be equal to {self.expected_context_length}, but got {context_length}"
        )

        # 2. Encode past actions
        action_embeds = self.action_embedding(context_actions)
        if self.do_classifier_free_guidance:
            action_embeds = torch.cat([action_embeds, action_embeds])

        # 3. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 4. Prepare latents
        num_channels_latents = context_latents.shape[2]
        # latents shape -> (batch_size, num_past_frames + 1, num_channels_latents, height, width)
        latents = self.prepare_latents(
            batch_size,
            num_channels_latents,
            height,
            width,
            context_latents.dtype,
            device,
            generator,
        )
        latents = torch.cat([context_latents, latents], dim=1)

        # 5. Denoising loop
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.do_classifier_free_guidance:
                    uncond_latents = latents.clone()
                    uncond_latents[:, :context_length].zero_()

                    latent_model_input = torch.cat([uncond_latents, latents])
                else:
                    latent_model_input = latents

                # Reshape so that context frames are concatenated at the latent channels
                latent_model_input = rearrange(
                    latent_model_input,
                    "b l c h w -> b (l c) h w",
                )
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # Predict the noise residual
                # TODO: Implement noise augmentation
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=action_embeds,
                    class_labels=torch.zeros(batch_size, dtype=torch.long).to(device),
                    return_dict=False,
                )[0]

                # Peform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

                # Denoise the last frame which is the next frame to be generated
                next_frame_latent = latents[:, -1]
                denoised_next_frame_latent = self.scheduler.step(noise_pred, t, next_frame_latent, return_dict=False)[0]
                latents[:, -1] = denoised_next_frame_latent

                progress_bar.update()

        # 6. Decode and postprocess the latents
        new_latents = latents[:, -1:]
        images = self.vae.decode(new_latents.squeeze(1) / self.vae.config.scaling_factor, return_dict=False)[0]
        images = self.image_processor.postprocess(
            images, output_type=output_type, do_denormalize=[True] * images.shape[0]
        )

        if not return_dict:
            return images, new_latents

        return GameNGenPipelineOutput(images=images, latents=new_latents)
