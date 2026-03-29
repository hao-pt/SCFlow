#!/bin/bash
# Download pretrained teacher model checkpoints to pretrained/

set -e

mkdir -p pretrained/celeb256_f8_adm
mkdir -p pretrained/InstaFlow

echo "==> Downloading celeba_f8_adm teacher checkpoint..."
gdown --fuzzy "https://drive.google.com/file/d/1AIuMr5Ewti6_wQAJdM9elsrERwrxI9Sb/view?usp=drive_link" \
    -O pretrained/celeb256_f8_adm/model_480.pth

echo "==> Downloading InstaFlow teacher checkpoint from HuggingFace..."
huggingface-cli download XCLiu/instaflow_0_9B_from_sd_1_5 \
    --local-dir pretrained/InstaFlow

echo "Done. Teacher checkpoints saved under pretrained/."
