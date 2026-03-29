<div align="center">
<h1>Official PyTorch Implementation of "Self-Corrected Flow Distillation for Consistent One-Step and Few-Step Text-to-Image Generation" <a href="https://arxiv.org/abs/2412.16906">(AAAI 2025)</a></h1>
</div>

<div align="center">
  <a href="https://quandao10.github.io/" target="_blank">Quan&nbsp;Dao</a><sup>*12†</sup> &emsp; <b>&middot;</b> &emsp;
  <a href="https://hao-pt.github.io/" target="_blank">Hao&nbsp;Phung</a><sup>*13†</sup> &emsp; <b>&middot;</b> &emsp;
  <a href="https://trung-dt.com/" target="_blank">Trung&nbsp;Dao</a><sup>1</sup> &emsp; <b>&middot;</b> &emsp;
  <a href="https://people.cs.rutgers.edu/~dnm/" target="_blank">Dimitris&nbsp;N. Metaxas</a><sup>2</sup> &emsp; <b>&middot;</b> &emsp;
  <a href="https://sites.google.com/site/anhttranusc/" target="_blank">Anh&nbsp;Tran</a><sup>1</sup>
  <br> <br>
  <sup>1</sup>VinAI Research &emsp;
  <sup>2</sup>Rutgers University &emsp;
  <sup>3</sup>Cornell University
  <br> <br>
  <a href="https://arxiv.org/abs/2412.16906">[Paper]</a>
  <a href="https://huggingface.co/haopt/scflow_t2i">[HuggingFace <img src="https://huggingface.co/front/assets/huggingface_logo.svg" width=20>]</a> &emsp;&emsp;
  <br> <br>
  <emp><sup>*</sup>Equal contribution</emp> &emsp;
  <emp><sup>†</sup>Work done while at VinAI Research</emp>
</div>

<br>

<img src="assets/teaser.png" width="100%">

**TLDR:** We introduce a self-corrected flow distillation method that integrates consistency models and adversarial training within the flow-matching framework, enabling consistent generation quality in both one-step and few-step sampling.

---

<details>
<summary><b>Table of Contents</b></summary>

- [Installation](#installation)
- [Data](#data)
- [Checkpoints](#checkpoints)
- [Inference](#inference)
- [Evaluation](#evaluation)
- [Training](#training)
- [Acknowledgment](#acknowledgment)
- [Contacts](#contacts)

</details>


> Abstract: Flow matching has emerged as a promising framework for training generative models, demonstrating impressive empirical performance while offering relative ease of training compared to diffusion-based models. However, this method still requires numerous function evaluations in the sampling process. To address these limitations, we introduce a self-corrected flow distillation method that effectively integrates consistency models and adversarial training within the flow-matching framework. This work is a pioneer in achieving consistent generation quality in both few-step and one-step sampling. Our extensive experiments validate the effectiveness of our method, yielding superior results both quantitatively and qualitatively on CelebA-HQ and zero-shot benchmarks on the COCO dataset.

Details of the model architecture and experimental results can be found in [our paper](https://arxiv.org/abs/2412.16906).

```bibtex
@inproceedings{dao2025scflow,
  title     = {Self-Corrected Flow Distillation for Consistent One-Step and Few-Step Text-to-Image Generation},
  author    = {Quan Dao and Hao Phung and Trung Dao and Dimitris Metaxas and Anh Tran},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  year      = {2025}
}
```

Please **CITE** our paper and give us a :star: whenever this repository is used to help produce published results or incorporated into other software.


TODO:
- [] add our pretrained checkpoints

---

## Installation

Tested on Linux with Python 3.10 and PyTorch 2.1.0.

```shell
conda create -n scflow python=3.10
conda activate scflow
pip install -r requirements.txt
```


## Data

### Unconditional generation

**CelebA-HQ 256:** Download our preprocessed lmdb as follow:

```bash
mkdir data/
cd data/
gdown --fuzzy https://drive.google.com/file/d/12NJYQv9lOZoVFcQgUeBaA0noYdewQ3lE/view?usp=sharing -O data/
unzip celeba-lmdb.zip
cd ../
```

### Text-to-Image generation

We use a 2M-sample subset of LAION with aesthetic score > 6.25. Equivalently, [laion/aesthetics_v2_6_5plus](https://dagshub.com/datasets/laion-aesthetics-v2-6-5/) is a subset of ~3M images with at aesthetic score >=6.5 that is publicly available. 

To precompute latents for a dataset for faster training:
```bash
pip install dagshub
python sd_distill/precomputed_data.py \
    --datadir laion/aesthetics_v2_6_5plus \
    --cache_dir ./data/laion_aesthetics_v2_6_5plus \
    --save_path ./data/laion_aesthetics_v2_6_5plus/latent_laion_aes/ \
    --num_samples 2_000_000 \
    --batch_size 64
```

### Pre-computed FID Statistics

Download pre-computed dataset stats for CelebA-HQ dataset [here](https://drive.google.com/file/d/1xuqt8KU_GiuiaTmHUhgMErC0r4jwwKvv/view?usp=drive_link) and place them in `pytorch_fid/`.

## Checkpoints

To download pretrained teacher checkpoints:
```bash
bash tools/download_teacher.sh
```

To download our distilled checkpoints:
```bash
bash tools/download_distilled.sh
```

All checkpoints are saved under the `pretrained/` folder.

## Inference

### Unconditional Generation

```shell
bash tools/test.sh
```

### Text-to-Image Generation

```shell
bash tools/infer_instaflow.sh
```

## Evaluation

To evaluate on CelebA-HQ, add the following flags to [tools/test_adm.sh](tools/test_adm.sh) to enable FID computation:

```shell
--compute_fid --output_log ${EXP}_${EPOCH_ID}_${METHOD}${STEPS}.log
```

Multi-GPU sampling with 8 GPUs is supported by default for faster evaluation.


## Training

### Unconditional Generation

```shell
bash tools/train.sh
```

### Text-to-Image Generation

```shell
bash tools/train_instaflow.sh
```

## Acknowledgment

This codebase builds upon [Flow Matching in Latent Space (LFM)](https://github.com/VinAIResearch/LFM.git). We also thank the authors of [LCM](https://github.com/luosiallen/latent-consistency-model), [Rectified Flow](https://github.com/gnobitab/RectifiedFlow), and [InstaFlow](https://github.com/gnobitab/InstaFlow) for their great work and publicly available codebases that facilitated this research.

## Contacts

If you have any problems, please open an issue in this repository or send an email to **tienhaophung@gmail.com**.
