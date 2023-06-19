import time

import gradio as gr
from gradio import inputs
from PIL import Image
import torch
from torchvision import transforms
import typer
from typing_extensions import Annotated

from internals.latent_edge_predictor import LatentEdgePredictor
from internals.anti_gradient_pipeline import AntiGradientPipeline


def demo(
    model_id: Annotated[str, typer.Option()] = "CompVis/stable-diffusion-v1-4",
    LEP_path: Annotated[str, typer.Option()] = "output/LEP.pt",
    device: Annotated[str, typer.Option()] = "cuda",
    port: Annotated[int, typer.Option()] = 7777,
):
    start_time = time.time()

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

    def inference(
        prompt,
        guidance,
        steps,
        width=512,
        height=512,
        seed=0,
        neg_prompt="",
        spimg=None,
        beta=1.6,
    ):
        # global pipe

        generator = torch.Generator("cuda").manual_seed(seed) if seed != 0 else None

        sketchs=None
        if spimg is not None:
            gsimg = Image.fromarray(spimg)
            tensor_img = torch.tile(transform(gsimg), (3, 1, 1)).unsqueeze(0)
            sketchs = pipe.vae.encode(tensor_img.to(pipe.device, dtype=pipe.vae.dtype)).latent_dist.sample() * 0.18215

        result = pipe(
            prompt,
            negative_prompt=neg_prompt,
            num_inference_steps=int(steps),
            guidance_scale=guidance,
            width=width,
            height=height,
            generator=generator,
            sketch_image=sketchs,
            beta=beta,
        )
        return result[0], None

    IMG_SIZE = 512

    css = """.finetuned-diffusion-div div{display:inline-flex;align-items:center;gap:.8rem;font-size:1.75rem}.finetuned-diffusion-div div h1{font-weight:900;margin-bottom:7px}.finetuned-diffusion-div p{margin-bottom:10px;font-size:94%}a{text-decoration:underline}.tabs{margin-top:0;margin-bottom:0}#gallery{min-height:20rem}
    """
    with gr.Blocks(css=css) as demo:
        gr.HTML(
            f"""
                <div class="finetuned-diffusion-div">
                <div>
                    <h1>Demo for orangemix</h1>
                </div>
                <p>Duplicating this space: <a style="display:inline-block" href="https://huggingface.co/spaces/akhaliq/anything-v3.0?duplicate=true"><img src="https://img.shields.io/badge/-Duplicate%20Space-blue?labelColor=white&style=flat&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAAAXNSR0IArs4c6QAAAP5JREFUOE+lk7FqAkEURY+ltunEgFXS2sZGIbXfEPdLlnxJyDdYB62sbbUKpLbVNhyYFzbrrA74YJlh9r079973psed0cvUD4A+4HoCjsA85X0Dfn/RBLBgBDxnQPfAEJgBY+A9gALA4tcbamSzS4xq4FOQAJgCDwV2CPKV8tZAJcAjMMkUe1vX+U+SMhfAJEHasQIWmXNN3abzDwHUrgcRGmYcgKe0bxrblHEB4E/pndMazNpSZGcsZdBlYJcEL9Afo75molJyM2FxmPgmgPqlWNLGfwZGG6UiyEvLzHYDmoPkDDiNm9JR9uboiONcBXrpY1qmgs21x1QwyZcpvxt9NS09PlsPAAAAAElFTkSuQmCC&logoWidth=14" alt="Duplicate Space"></a>       </p>
                </p>
                </div>
            """
        )
        with gr.Row():

            with gr.Column(scale=55):
                with gr.Group():
                    with gr.Row():
                        prompt = gr.Textbox(
                            label="Prompt",
                            show_label=True,
                            max_lines=2,
                            placeholder="Enter prompt.",
                        )
                        neg_prompt = gr.Textbox(
                            label="Negative Prompt",
                            show_label=True,
                            max_lines=2,
                            placeholder="Enter negative prompt.",
                        )

                    with gr.Row():
                        generate = gr.Button(value="Generate")

                    image_out = gr.Image(height=IMG_SIZE)

                error_output = gr.Markdown()

            with gr.Column(scale=45):
                with gr.Tab("Options"):
                    with gr.Group():
                        model = gr.Textbox(
                            interactive=False,
                            label="Model",
                            placeholder=model_id,
                        )

                        with gr.Row():
                            guidance = gr.Slider(
                                label="Guidance scale", value=8.0, maximum=15
                            )
                            beta = gr.Slider(
                                label="Beta", value=1.6, maximum=10.0, step=0.1
                            )
                            steps = gr.Slider(
                                label="Steps", value=250, minimum=50, maximum=500, step=10
                            )

                        with gr.Row():
                            width = gr.Slider(
                                label="Width", value=IMG_SIZE, minimum=64, maximum=1024, step=8
                            )
                            height = gr.Slider(
                                label="Height", value=IMG_SIZE, minimum=64, maximum=1024, step=8
                            )

                        seed = gr.Slider(
                            0, 2147483647, label="Seed (0 = random)", value=0, step=1
                        )

                with gr.Tab("SketchPad"):
                    with gr.Group():
                        # sp = gr.Sketchpad(shape=(IMG_SIZE, IMG_SIZE), tool="sketch", brush_radius=3)
                        sp = gr.Sketchpad(shape=(128, 128), tool="sketch", brush_radius=3)

        inputs = [
            prompt,
            guidance,
            steps,
            width,
            height,
            seed,
            neg_prompt,
            sp,
            beta,
        ]
        outputs = [image_out, error_output]
        prompt.submit(inference, inputs=inputs, outputs=outputs)
        generate.click(inference, inputs=inputs, outputs=outputs)

    print(f"Space built in {time.time() - start_time:.2f} seconds")
    demo.launch(debug=True, share=True, server_port=port)