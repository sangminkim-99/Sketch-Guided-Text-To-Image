# Sketch-Guided Text-to-Image Diffusion Models

This repository contains an unofficial implementation of Google's paper **Sketch-Guided Text-to-Image Diffusion Models**. 
The goal of this project is to generate high-quality images from textual descriptions and corresponding sketches.


## References

This implementation was inspired by and references the following repositories:

- [ogkalu2/Sketch-Guided-Stable-Diffusion](https://github.com/ogkalu2/Sketch-Guided-Stable-Diffusion)
- [Mikubill/sketch2img](https://github.com/Mikubill/sketch2img)


## Overview

The Sketch-Guided Text-to-Image Diffusion Models project focuses on generating realistic images from textual descriptions and corresponding sketches.


## Installation

1. Clone the repository:

```shell
git clone --recurse-submodules https://github.com/sangminkim-99/Sketch-Guided-Text-To-Image.git
cd Sketch-Guided-Text-To-Image
```

2. Create and activate a new Conda environment:

```shell
conda create -n sketch-guided-env python=3.9
conda activate sketch-guided-env
```

3. Install the necessary dependencies. You may use `pip` to install the required packages:

```shell
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia # change to your own version of torch
pip install -r requirements.txt
```

4. Download necessary datasets.
    * Download some indoor images from ImageNet Dataset with [ImageNet-Dataset-Downloader](https://github.com/mf1024/ImageNet-Datasets-Downloader)
    ```shell
    chmod u+x scripts/download_imagenet_room_dataset.sh
    ./scripts/download_imagenet_room_dataset.sh
    ```
    
    * Edge Map Generation with [pidinet](https://github.com/ogkalu2/pidinet-for-imagenet)
    ```shell
    chmod u+x scripts/generate_edge_map.sh
    ./scripts/generate_edge_map.sh
    ```
    

## Usage

- Show helper messages for all possible commands

```shell
python app.py --help
```

- Train Latent Edge Predictor

Currently supports `--batch-size 1` only.

```shell
python app.py train-lep
```

- Sample image with Latent Edge Predictor

```shell
python app.py sample --sketch-file-path {PATH} --prompt {PROMPT}
```

- Gradio web demo (_debugging_)

```shell
python app.py demo
```


## TODOs

- [ ] Reproduce the bicycle example

- [ ] Upload pretrained LEP

&#9745; aa


## Acknowledgments

We would like to express our gratitude to the authors of the original paper and the developers of the referenced repositories for their valuable contributions, which served as the foundation for this implementation.


## Disclaimer

This is an unofficial implementation and is not affiliated with Google or the authors of the original paper.
