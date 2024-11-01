# Latent Flow Matching

## Installation

Python `3.10` and Pytorch `1.13.1`/`2.0.0` are used in this implementation.
Please install required libraries:

```
pip install -r requirements.txt
```

## Dataset preparation

For CelebA HQ 256 and FFHQ 256 please check [NVAE's instructions](https://github.com/NVlabs/NVAE#set-up-file-paths-and-data) out.

For higher resolution datasets (CelebA HQ 512 & 1024), please refer to [WaveDiff's documents](https://github.com/VinAIResearch/WaveDiff#dataset-preparation).

For ImageNet dataset, please download it directly from [the official website](https://www.image-net.org/download.php).

## Pretrained VAE encoder
Please download it [here](https://drive.google.com/file/d/16No4pSk5qcDXRPmCmCCnD0jjoA4FIL8S/view?usp=drive_link).

## Training

All training scripts are wrapped in [run.sh](bash_scripts/run.sh). Simply comment/uncomment the relevant commands and run `bash run.sh`.

## Testing

Run [run_test.sh](bash_scripts/run_test.sh) / [run_test_cls.sh](bash_scripts/run_test_cls.sh) with corresponding argument's file.

```
bash run_test.sh <path_to_arg_file>
```

> Only 1 gpu is required.

<details>

<summary>These arguments are specified as follows:</summary>

```bash
MODEL_TYPE=DiT-L/2
EPOCH_ID=475
DATASET=celeba_256
EXP=celeb_f8_dit
METHOD=dopri5
STEPS=0
USE_ORIGIN_ADM=True
IMG_SIZE=256
```

</details>

Detailed arguments and checkpoints are provided below:

<table>
  <tr>
    <th>Exp</th>
    <th>Args</th>
    <th>FID</th>
    <th>Checkpoints</th>
  </tr>

  <tr>
    <td> celeb_f8_dit </td>
    <td><a href="test_args/celeb256_dit.txt"> test_args/celeb256_dit.txt</a></td>
    <td>5.26</td>
    <td><a href="https://drive.google.com/drive/folders/1tbd1t0Yt3ix1v_OCGWJ7xyeubhCi99ql?usp=share_link">model_475.pth</a></td>
  </tr>

  <tr>
    <td> ffhq_f8_dit </td>
    <td><a href="test_args/ffhq_dit.txt"> test_args/ffhq_dit.txt</a></td>
    <td>4.55</td>
    <td><a href="https://drive.google.com/drive/folders/1jn6xHlaQ72hKk9RtJKo5lvr7SvYMCobU?usp=share_link">model_475.pth</a></td>
  </tr>
  <tr>
    <td> imnet_f8_ditb2 </td>
    <td><a href="test_args/imnet_dit.txt"> test_args/imnet_dit.txt</a></td>
    <td>4.46</td>
    <td><a href="https://drive.google.com/file/d/1SUc9M85WOfi4CZfRJkpf2tDu3g9dC3YI/view?usp=sharing">model_875.pth</a></td>
  </tr>
  <tr>
    <td> celeb512_f8_adm </td>
    <td><a href="test_args/celeb256_adm.txt"> test_args/celeb512_adm.txt</a></td>
    <td>6.35</td>
    <td><a href="https://drive.google.com/drive/folders/1lWE9hCqzZ2Q1mS2BmTsA3nYWB_T25wqV?usp=share_link">model_575.pth</a></td>
  </tr>
  <tr>
    <td> celeba_f8_adm </td>
    <td><a href="test_args/celeb256_adm.txt"> test_args/celeb256_adm.txt</a></td>
    <td>5.82</td>
    <td>---</td>
  </tr>
  <tr>
    <td> ffhq_f8_adm </td>
    <td><a href="test_args/ffhq_adm.txt"> test_args/ffhq_adm.txt</a></td>
    <td>6.12</td>
    <td>---</td>
  </tr>
  <tr>
    <td> imnet_f8_adm </td>
    <td><a href="test_args/imnet_adm.txt"> test_args/imnet_adm.txt</a></td>
    <td>8.58</td>
    <td>---</td>
  </tr>
  <tr>
    <td> celeba_f8_adm_distill </td>
    <td><a href="test_args/celeb256_adm.txt"> test_args/celeb256_adm.txt</a></td>
    <td>10.69</td>
    <td>---</td>
  </tr>
  

</table>

Remaining models are put at [here](https://drive.google.com/drive/folders/1sJhl9gR0H4cIQJ3rVjJAqcOHZ8jpNVte?usp=sharing).
> All attached links are made by an anonymous account.
>
All models: https://drive.google.com/file/d/1_6HlS_tTKBrvlHDQiVYsMyREBlIL6Nvg/view?usp=drive_link

Please put downloaded pre-trained models in `saved_info/latent_flow/<DATASET>/<EXP>` directory where `<DATASET>` is defined as in [bash_scripts/run.sh](./bash_scripts/run.sh).

<details>
<summary>Utilities</summary>

To measure time, please add `--measure_time` in the script.

To compute the number of function evaluations of adaptive solver (default: `dopri5`), please add `--compute_nfe` in the script.

To use fixed-steps solver (e.g. `euler` and `heun`), please add `--use_karras_samplers` and change two arguments as follow:

```
METHOD=heun
STEPS=50
```

</details>

### Evaluation

To evaluate FID scores, please download pre-computed stats from [here](https://drive.google.com/drive/folders/1BXCqPUD36HSdrOHj2Gu_vFKA3M3hJspI?usp=share_link) and put it to `pytorch_fid`.

Please add `--compute_fid --output_log ${EXP}_${EPOCH_ID}_${METHOD}${STEPS}.log \` in `run_test.sh` for unconditional generation and 
Then run `bash run_test_ddp.sh` for unconditional generation or `run_test_cls` for conditional generation. By default, multi-gpu sampling with 8 GPUs is supported for faster compute.

<details>
<summary>Computing stats for new dataset</summary>

`pytorch_fid/compute_dataset_stat.py` is provided for this purpose.

```bash
python pytorch_fid/compute_dataset_stat.py \
  --dataset <dataset> --datadir <path_to_data> \
  --image_size <image_size> --save_path <path_to_save>
```

</details>

## Distillation for few-step generation
Training:
```
bash run_reflow.sh
```
Test:
```
bash run_test_consistent.sh
```

