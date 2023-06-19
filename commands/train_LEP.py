import os

from diffusers import StableDiffusionPipeline
from einops import rearrange
import numpy as np
import torch
from tqdm import tqdm
from transformers import CLIPTokenizer
import typer
from typing_extensions import Annotated

from internals.diffusion_utils import encode_img, encode_text, hook_unet, noisy_latent
from internals.latent_edge_predictor import LatentEdgePredictor
from internals.LEP_dataset import LEPDataset


def train_LEP(
    model_id: Annotated[str, typer.Option()] = "CompVis/stable-diffusion-v1-4",
    device: Annotated[str, typer.Option()] = "cuda",
    dataset_dir: Annotated[str, typer.Option(help="path to the parent directory of image data")] = "./data/imagenet/imagenet_images",
    edge_map_dir: Annotated[str, typer.Option(help="path to the parent directory of edge map data")] = "./data/imagenet/edge_maps",
    save_path: Annotated[str, typer.Option(help="path to save LEP model")] = "./output/LEP.pt",
    batch_size: Annotated[int, typer.Option(help="batch size for training LEP. Decrease this if OOM occurs.")] = 16,
    training_step: Annotated[int, typer.Option()] = 3000,
    lr: Annotated[float, typer.Option()] = 1e-4, # not specified in the paper
    num_train_timestep: Annotated[int, typer.Option(help="maximum diffusion timestep")] = 250, # not specified in the paper
):
    '''
    Train the Latent Edge Predictor.
    '''
    # create output folder
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # create dataset & loader
    dataset = LEPDataset(dataset_dir, edge_map_dir)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # initialize stable diffusion pipeline.
    # the paper use stable-diffusion-v1.4
    pipe = StableDiffusionPipeline.from_pretrained(model_id, safety_checker=None).to(device)

    unet = pipe.unet
    unet.enable_xformers_memory_efficient_attention()

    # hook the feature_blocks of unet
    feature_blocks = hook_unet(pipe.unet)

    # initialize LEP
    LEP = LatentEdgePredictor(input_dim=9324, output_dim=4, num_layers=10).to(device)

    pipe.unet.eval()
    pipe.vae.eval()
    pipe.text_encoder.eval()

    # need this lines?
    pipe.unet.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    LEP.requires_grad_(True)

    # load clip tokenizer
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

    optimizer = torch.optim.Adam(LEP.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    step = 0
    while True:
        tbar = tqdm(dataloader)
        for _, (image, edge_map, caption) in enumerate(tbar):

            optimizer.zero_grad()

            # image to latent
            latent_image = encode_img(pipe.vae, image)
            latent_edge = encode_img(pipe.vae, edge_map)
            
            caption_embedding = torch.cat([encode_text(pipe.text_encoder, tokenizer, c) for c in caption])
            noisy_image, noise_level, timesteps = noisy_latent(latent_image, pipe.scheduler, batch_size * 2, num_train_timestep)

            # one reverse step to get the feature blocks
            pipe.unet(torch.cat([latent_image] * 2), timesteps, encoder_hidden_states=caption_embedding)

            # Edge prediction
            intermediate_result = []
            for block in feature_blocks:
                resized = torch.nn.functional.interpolate(block.output, size=latent_image.shape[2], mode="bilinear") 
                intermediate_result.append(resized)
                # free vram
                del block.output
                
            intermediate_result = torch.cat(intermediate_result, dim=1)
            pred_edge_map = LEP(intermediate_result, noise_level)
            pred_edge_map = rearrange(pred_edge_map, "(b w h) c -> b c h w", b=batch_size * 2, h=latent_edge.shape[2], w=latent_edge.shape[3])

            # calculate MSE loss
            loss = criterion(pred_edge_map, latent_edge)
            loss.backward()

            optimizer.step()

            if step % 10 == 0:
                tbar.set_description(f"Loss: {loss.item():.3f}")

            if step >= training_step:
                break

            step += 1

        if step >= training_step:
            print(f'Finish to optimize. Save file to {save_path}')
            torch.save(LEP.state_dict(), save_path)
            break
