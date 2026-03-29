#!/bin/bash
# Download distilled model checkpoints to pretrained/

set -e

mkdir -p pretrained/scflow_t2i

# celeba_f8_adm distilled model is currently TBU
# echo "==> Downloading celeba_f8_adm distilled checkpoint..."

echo "==> Downloading scflow_t2i distilled checkpoint from HuggingFace..."
huggingface-cli download haopt/scflow_t2i \
    --local-dir pretrained/scflow_t2i

echo "Done. Distilled checkpoints saved under pretrained/."
