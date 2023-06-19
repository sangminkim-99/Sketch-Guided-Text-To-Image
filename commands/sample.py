import gradio as gr
from gradio import inputs
from PIL import Image
import torch
from torchvision import transforms
import typer
from typing_extensions import Annotated

from internals.latent_edge_predictor import LatentEdgePredictor
from internals.anti_gradient_pipeline import AntiGradientPipeline

def sample(
    model_id: Annotated[str, typer.Option()] = "CompVis/stable-diffusion-v1-4",
    LEP_path: Annotated[str, typer.Option()] = "output/LEP.pt",
    sketch_file_path: Annotated[str, typer.Option()] = "data/sketchs/sample_sketch.png",
    output_file_path: Annotated[str, typer.Option()] = "output/sample.png",
    prompt: Annotated[str, typer.Option()] = "",
    neg_prompt: Annotated[str, typer.Option()] = "",
    guidance_scale: Annotated[float, typer.Option()] = 8.0,
    beta: Annotated[float, typer.Option()] = 1.6,
    device: Annotated[str, typer.Option()] = "cuda",
    inference_step: Annotated[int, typer.Option()] = 250,
    seed: Annotated[int, typer.Option()] = 0,
):

    pipe = AntiGradientPipeline.from_pretrained(
        model_id,
        # torch_dtype=torch.float16,
    )

    # inject
    unet = pipe.unet
    unet.enable_xformers_memory_efficient_attention()

    pipe.to(device)

    transform = transforms.Compose(
        [
            transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: t * 2 - 1)
        ]
    )

    LEP =  LatentEdgePredictor(9324, 4, 10)
    LEP.load_state_dict(torch.load(LEP_path))
    LEP.to(device)
    pipe.setup_LEP(LEP)

    generator = torch.Generator("cuda").manual_seed(seed) if seed != 0 else None

    sketch_img = Image.open(sketch_file_path)
    tensor_img = torch.tile(transform(sketch_img), (3, 1, 1)).unsqueeze(0)
    sketchs = pipe.vae.encode(tensor_img.to(pipe.device, dtype=pipe.vae.dtype)).latent_dist.sample() * 0.18215

    result = pipe(
        prompt,
        negative_prompt=neg_prompt,
        num_inference_steps=inference_step,
        guidance_scale=guidance_scale,
        beta=beta,
        width=512,
        height=512,
        generator=generator,
        sketch_image=sketchs,
    )

    result[0].save(output_file_path)