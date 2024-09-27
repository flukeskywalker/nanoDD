<p align="center">
  <img src="./_img/dd.gif" width=800 style="max-width: 100%;" alt="[nano] discrete diffusion">
</p>
<p align="center" style="font-family: sans-serif; font-size: 24px;">
  <b>nanoDD</b>
</p>

I'm writing simple & scalable Discrete Diffusion implementations in PyTorch for education, research and fun!

## What is Discrete Diffusion?
In simple terms, typical LLMs (such as GPTs) generate text from left-to-right, while Discrete Diffusion LMs generate a chunk of text in parallel.

More formally, Diffusion is a set of techniques for modeling data by learning a series of conditional *noisy* distributions over *all* token variables. 
This is in contrast to autoregressive models (GPTs etc.) that learn *non-noisy* conditional distributions over *one* token variable at a time.
*Discrete* Diffusion is the application of ideas similar to those used by *continuous* diffusion (used for image generation models like Stable Diffusion, Midjourney, Flux etc.) to discrete data, like text. 

Example:

![img](./_img/sample.gif)

This GIF shows what sampling from a pre-trained masking-based discrete diffusion model looks like using the [sampling script](./sample.py) in this repo.
We start out with the whole sequence composed of mask tokens (maximum "noise") and iteratively unmask --- and hence reduce the noise --- in the sequence.

## Goals for this repo
I want more people to work on discrete diffusion, so the primary goals are to be correct, simple and instructive for newcomers to these algorithms.
Readers should be able to use the implementations to help understand the original papers.

I also want the code to be efficient and scalable so that the repo can be used as a starting point for hacking on ideas.
Similar to the philosophy in [nanoGPT](https://github.com/karpathy/nanoGPT), nanoDD relies on pure PyTorch and avoids abstractions (as well as dependencies that contain abstractions such as training frameworks).
The training script itself is directly adapted from nanoGPT with several modifications.

## Models

To start off, you can train, evaluate, and sample from an [Absorbing (or "Masking") D3PM](https://arxiv.org/abs/2107.03006) using a [Diffusion Transformer](./dit.py) on the [text8](https://paperswithcode.com/dataset/text8) and [openwebtext](https://skylion007.github.io/OpenWebTextCorpus/) datasets.
Additional models are planned.

## Usage

Install dependencies:
```bash
pip install -r requirements.txt
```

Download pre-trained model and generate samples for text8 dataset:
```bash
# download pre-trained weights for D3PMAbsorbing from HF (~700 MB for text8 ,~1GB for openwebtext)
git clone git@hf.co:rupspace/nanoDD-D3PM-text8
git clone git@hf.co:rupspace/nanoDD-D3PM-openwebtext

# sample 1 text8 sequence (default):
python sample.py ./nanoDD-D3PM-text8/ckpt.pt
python sample.py ./nanoDD-D3PM-openwebtext/ckpt.pt --dataset openwebtext

# sample 4 sequences in a batch:
python sample.py ./nanoDD-D3PM-text8/ckpt.pt --batch_size 4
# check options for sample.py
python sample.py --help
```

Or train from scratch:
```bash
# download and prepare text8
python data/prepare_text8.py
# train Absorbing D3PM on single GPU (for prototyping etc)
python train.py d3pm_text8
```

### Multi-GPU training
Multi-GPU training is likely necessary if you want to train 12-layer models on openwebtext (or even text8).
For openwebtext in particular you should ideally train on 16 or 32 A100 GPUs.
Note that Discrete Diffusion models take much longer to train than autoregressive models.

```bash
# d3pm_text8_4gpu modifies the single GPU config (see `configs.py`)
# note that the batch_size config is per GPU, while global_batch_size == batch_size * gradient_accumulation_steps * num_gpus
# validation uses all GPUs, so eval_iters should be modified when changing number of GPUs
# following uses ~35 GB GPU memory per GPU in my experiments
torchrun --standalone --nproc_per_node=8 train.py d3pm_text8_4gpu

# to train on openwebtext, first prepare using script borrowed from nanoGPT
python data/prepare_openwebtext.py
# train Absorbing D3PM
torchrun --standalone --nproc_per_node=8 train.py d3pm_openwebtext_8gpu
# see train.py for commands to launch on 32 GPUs across 4 nodes
```

### Training Notes

Note that the training will attempt to compile the model by default, which takes extra time to begin training.
Append `--no-compile` to the training commands when launching to disable this for debugging etc.

Currently, sampling and evaluation scripts do not compile the model, so they get going immediately.

### Configuration

This is not a full-on research library so the config system is rather simple to avoid using a tool that readers might not know.
However, there is basic support for using different model/configs.

For example, `d3pm_text8()` is a function that defines a configuration in [configs.py](./configs.py) for training D3PMAbsorbing on text8.
One can add new configs by defining functions in this file, and specify their name on the cmd line when running `train.py`.
Any training args defined by a config function over-ride any global training args defined in `train.py`.


### Evaluating loss
```bash
# evaluate on val set (default)
python evaluate.py ./nanoDD-D3PM-text8/ckpt.pt
# evaluate on test set
python evaluate.py ./nanoDD-D3PM-text8/ckpt.pt --split test
# check options for eval script
python evaluate.py --help
```

## Results (text8)

| Model               | Test Bits/Token (text8) |
|---------------------|:-----------------------:|
| D3PM Absorbing      |          1.37           |

Training the D3PM Absorbing model will produce loss values similar to those in the plot below, finally reaching a validation loss of 1.30, which results in a test loss of 1.37.
Note that this is substantially better than in the original paper (1.45).
The training loss will start at around 5.0 (approximated and converted to bits-per-token) and you will observe noisy loss values throughout training due to noise in the diffusion process.

You can sample diffusion time steps more uniformly per batch ("low-discrepancy sampler") and this reduces the variance in the loss but in my experience does not make the training faster or reach a lower mean loss.

![img](_img/d3pm_absorbing_loss.png)