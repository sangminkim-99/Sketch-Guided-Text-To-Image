import contextlib
from typing import Callable, List, Optional, Union

from diffusers import StableDiffusionPipeline
from diffusers.utils import logging
from einops import rearrange
import torch
import torch.nn.functional as F

from internals.diffusion_utils import hook_unet
from internals.latent_edge_predictor import LatentEdgePredictor


class AntiGradientPipeline(StableDiffusionPipeline):
        
    def setup_LEP(self, LEP):
        self.LEP: LatentEdgePredictor = LEP
        self.feature_blocks = hook_unet(self.unet)

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 8.0,
        beta: float = 1.6,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        sketch_image=None,
    ):
        self.LEP.to(self.unet.device)
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(prompt, height, width, callback_steps)

        # 2. Define call parameters
        batch_size = 1 if isinstance(prompt, str) else len(prompt)
        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        text_embeddings = self._encode_prompt(
            prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt
        )

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.unet.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            text_embeddings.dtype,
            device,
            generator,
            latents,
        )
        noise = latents.detach().clone()

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                latent_model_input = latent_model_input.requires_grad_(True)
                
                ctx = torch.enable_grad()
                step_stop = 0.5 * len(timesteps)
                if i > step_stop:
                    ctx = contextlib.nullcontext()

                # predict the noise residual
                with ctx:
                    noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample
                
                # compute predictor gradient
                with ctx:
                    if i <= step_stop:
                        latents = self.apply_anti_gradient(latent_model_input, latents, noise, t, sketch_image, beta)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        # 8. Post-processing
        image = self.decode_latents(latents)

        # 9. Run safety checker
        image, has_nsfw_concept = self.run_safety_checker(image, device, text_embeddings.dtype)

        # 10. Convert to PIL
        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image, has_nsfw_concept)

        return image
    
    def get_noise_level(self, noise, timesteps):
        sqrt_one_minus_alpha_prod = (1 - self.scheduler.alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(noise.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)
            
        noise_level = sqrt_one_minus_alpha_prod.to(noise.device) * noise
        return noise_level
    
    def apply_anti_gradient(self, latents_prev, latents, noise, timestep, target, beta):
        if target is None:
            return latents
        
        intermediate_result = []
        for block in self.feature_blocks:
            resized = F.interpolate(block.output, size=latents.shape[2], mode="bilinear") 
            intermediate_result.append(resized)
            del block.output
                    
        intermediate_result = torch.cat(intermediate_result, dim=1)
        estimate_noise = self.get_noise_level(noise, timestep)
        outputs = self.LEP(intermediate_result, torch.cat([estimate_noise] * 2))
        
        b, _, h, w = latents_prev.shape
        _, outputs = rearrange(outputs, "(b w h) c -> b c h w", b=b, h=h, w=w).chunk(2)
        loss = F.mse_loss(target.float(), outputs.float(), reduction="mean")
            
        _, cond_grad = (-torch.autograd.grad(loss, latents_prev)[0]).chunk(2)
        alpha = torch.linalg.norm(latents_prev - latents) / torch.linalg.norm(cond_grad) * beta
        return latents + alpha * cond_grad