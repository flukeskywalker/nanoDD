<p align="center">
  <img src="./_img/dd.gif" width=800 style="max-width: 100%;" alt="[nano] discrete diffusion">
</p>
<p align="center" style="font-family: sans-serif; font-size: 24px;">
  <b>nanoDD</b>
</p>

I'm writing simple & scalable Discrete Diffusion implementations in PyTorch for education, research and fun!

# What is Discrete Diffusion?
Discrete Diffusion is a set of techniques for modeling discrete data by learning a series of conditional *noisy* distributions over *all* token variables. 
It is the application of ideas similar to those used by _continuous diffusion_ (used for image generation models like Stable Diffusion, Midjourney, Flux etc) to discrete data, like text. 
This is in contrast to auto-regressive LLMs (GPTs etc) that learn *non-noisy* conditional distributions over *one* token variable at a time.
In simple terms, auto-regressive LMs generate text from left-to-right, while DDLMs generate a chunk of text in parallel.

Example:

![img](./_img/sample.gif)

This GIF shows what sampling from a pre-trained masking-based discrete diffusion model looks like using the [sampling script](./sample.py) in this repo.
We start out with the whole sequence composed of mask tokens (maximum "noise") and iteratively unmask --- and hence reduce the noise --- in the sequence.

# Goals for this repo
I want more people to work on discrete diffusion, so the primary goals are to be correct, simple and instructive for newcomers to these algorithms.
Readers should be able to use the implementations to help understand the original papers.

I also want the code to be efficient and scalable so that the repo can be used as a starting point for hacking on ideas. 
Similar to the philosophies in [nanoGPT](https://github.com/karpathy/nanoGPT) and nanoDO, nanoDD tries to avoid dependencies in favor of transparency and does not create or depend on libraries/frameworks other than pure PyTorch.
The training script itself is directly adapted from nanoGPT.

To start, this repo provides the code to [train](./train.py) an [Absorbing (or "Masking") D3PM](https://arxiv.org/abs/2107.03006) using a [Diffusion Transformer](./dit.py) on the [text8](https://paperswithcode.com/dataset/text8) dataset.

Upcoming:
- more models in addition to D3PM Absorbing
- more datasets; text8 is a standard academic benchmark allowing us to compare results to papers, but samples are ugly :(

# Usage

Install dependencies:
```bash
pip install -r requirements.txt
```

Download pre-trained model and generate samples:
```bash
# download pre-trained weights from HF (~700 MB)
git clone git@hf.co:rupspace/nanoDD-D3PM-text8
# sample 1 sequence (default):
python sample.py ./nanoDD-D3PM-text8/ckpt.pt
# sample 4 sequence in a batch:
python sample.py ./nanoDD-D3PM-text8/ckpt.pt --batch_size 4
```

Training:
```bash
# download and prepare text8
python data/prepare_text8.py
# train D3PM on text8 with default hyperparameters
python train.py d3pm_text8
```
Note that the training will attempt to compile the model by default, which takes extra time to begin training.
Set `compile=False` in the config or training script to disable this.

The config system is rather simple:
`d3pm_text8` is the name of the function that defines the configuration in [configs.py](./configs.py).
Add new configs by defining functions here, and specify their name on the cmd line to over-ride any global training configuration in `train.py`.

Multi-GPU training:
```bash
# d3pm_text8_8gpu modifies the original config (see `configs.py`)
# note that the batch_size config is per GPU, not the global batch size
torchrun --standalone --nproc_per_node=8 train.py d3pm_text8_8gpu
```

Evaluating loss:
```bash
# evaluate on val set (default)
python evaluate.py ./nanoDD-D3PM-text8/ckpt.pt
# evaluate on test set
python evaluate.py ./nanoDD-D3PM-text8/ckpt.pt --split test
# check options for eval script
python evaluate.py --help
```
