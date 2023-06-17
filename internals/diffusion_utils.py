from diffusers.models import AutoencoderKL
import torch
from transformers.models.clip import CLIPTextModel, CLIPTokenizer


def encode_img(vae: AutoencoderKL, image: torch.Tensor):
    generator = torch.Generator(vae.device).manual_seed(0)
    latents = vae.encode(image.to(device=vae.device, dtype=vae.dtype)).latent_dist.sample(generator=generator)
    latents = latents * 0.18215
    return latents


def encode_text(text_encoder: CLIPTextModel, tokenizer: CLIPTokenizer, text):
    text_input = tokenizer([text], padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
    with torch.no_grad():
        text_embeddings = text_encoder(text_input.input_ids.to(text_encoder.device))[0]
    max_length = text_input.input_ids.shape[-1]
    uncond_input = tokenizer([""], padding="max_length", max_length=max_length, return_tensors="pt")
    with torch.no_grad():
        uncond_embeddings = text_encoder(uncond_input.input_ids.to(text_encoder.device))[0]   
    # return torch.cat([uncond_embeddings, text_embeddings]).unsqueeze(0)
    return torch.cat([uncond_embeddings, text_embeddings])

def noisy_latent(image, noise_scheduler, num_train_timestep):
    # TODO: check the range of timesteps
    timesteps = torch.randint(0, num_train_timestep, (1,), device=image.device).long()
    noise = torch.randn(image.shape).to(image.device)
    noisy_image = noise_scheduler.add_noise(image, noise, timesteps)
    sqrt_alpha_prod = noise_scheduler.alphas_cumprod[timesteps.cpu()].to(image.device) ** 0.5
    sqrt_alpha_prod = sqrt_alpha_prod.flatten()
    while len(sqrt_alpha_prod.shape) < len(image.shape):
        sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
    noise_level = noisy_image - (sqrt_alpha_prod * image)
    return noisy_image, noise_level, timesteps