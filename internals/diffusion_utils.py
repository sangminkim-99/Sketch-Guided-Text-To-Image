from diffusers import AutoencoderKL, UNet2DConditionModel
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


def noisy_latent(image, noise_scheduler, batch_size, num_train_timestep):
    # TODO: check the range of timesteps
    timesteps = torch.randint(0, num_train_timestep, (batch_size,), dtype=torch.int64, device=image.device).long()
    # timesteps = torch.randint(250, 900, (batch_size,), dtype=torch.int64, device=image.device).long()
    noise = torch.randn(image.shape).to(image.device)
    noisy_image = noise_scheduler.add_noise(image, noise, timesteps)
    sqrt_alpha_prod = noise_scheduler.alphas_cumprod[timesteps.cpu()].to(image.device) ** 0.5
    sqrt_alpha_prod = sqrt_alpha_prod.flatten()
    while len(sqrt_alpha_prod.shape) < len(image.shape):
        sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
    noise_level = noisy_image - (sqrt_alpha_prod * image)
    return noisy_image, noise_level, timesteps

    
def hook_unet(unet: UNet2DConditionModel):
    blocks_idx = [0, 1, 2]
    feature_blocks = []
    def hook(module, input, output):
        if isinstance(output, tuple):
            output = output[0]
        
        if isinstance(output, torch.TensorType):
            feature = output.float()
            setattr(module, "output", feature)
        elif isinstance(output, dict): 
            feature = output.sample.float()
            setattr(module, "output", feature)
        else: 
            feature = output.float()
            setattr(module, "output", feature)
    
    # TODO: Check below lines are correct

    # 0, 1, 2 -> (ldm-down) 2, 4, 8
    for idx, block in enumerate(unet.down_blocks):
        if idx in blocks_idx:
            block.register_forward_hook(hook)
            feature_blocks.append(block) 
            
    # ldm-mid 0, 1, 2
    for block in unet.mid_block.attentions + unet.mid_block.resnets:
        block.register_forward_hook(hook)
        feature_blocks.append(block) 
    
    # 0, 1, 2 -> (ldm-up) 2, 4, 8
    for idx, block in enumerate(unet.up_blocks):
        if idx in blocks_idx:
            block.register_forward_hook(hook)
            feature_blocks.append(block)  
            
    return feature_blocks