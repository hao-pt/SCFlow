import os
import time
import subprocess

import pandas as pd

slurm_template = """#!/bin/bash -e
#SBATCH --job-name={job_name}
#SBATCH --output={slurm_output}/slurm_%A.out
#SBATCH --error={slurm_output}/slurm_%A.err
#SBATCH --gpus={num_gpus}
#SBATCH --nodes=1
#SBATCH --mem-per-gpu=36G
#SBATCH --cpus-per-gpu=32
#SBATCH --partition=research
#SBATCH --mail-type=all
#SBATCH --mail-user=v.haopt12@vinai.io
#SBATCH --ntasks=1

module purge
module load python/miniconda3/miniconda3
eval "$(conda shell.bash hook)"
conda activate {env_path}

export MASTER_PORT={master_port}
export WORLD_SIZE=1

export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

export PYTHONPATH=$(pwd):$PYTHONPATH

export MODEL_TYPE={model_type}
export EPOCH_ID={epoch}
export DATASET={dataset}
export EXP={exp}
export OUTPUT_LOG={output_log}

echo "----------------------------"
echo $MODEL_TYPE $EPOCH_ID $DATASET $EXP {method} {num_steps}
echo "----------------------------"

# CUDA_VISIBLE_DEVICES={device} python test_flow_latent.py --exp $EXP \
#     --dataset $DATASET --batch_size 128 --epoch_id $EPOCH_ID \
#     --image_size 256 --f 8 --num_in_channels 4 --num_out_channels 4 \
#     --nf 256 --ch_mult 1 2 2 2 --attn_resolution 16 8 --num_res_blocks 2 \
#     --model_type $MODEL_TYPE \
#     --method {method} --num_steps {num_steps} \
#     --master_port $MASTER_PORT  --num_process_per_node {num_gpus} \
#     --n_sample 50_000 \
#     --cfg_scale {cfg_scale} \
#     --compute_fid --output_log $OUTPUT_LOG \
#     --faster_test \
#     --num_classes 1 \
#     # --load_ema \
#     # --num_classes 1000 --label_dim 1000 --label_dropout 0.15 \
#     # --measure_time \
#     # --num_head_channels 64 \
#     # --use_karras_samplers \

CUDA_VISIBLE_DEVICES={device} python test_flow_latent.py --exp $EXP \
    --dataset $DATASET --batch_size 128 --epoch_id $EPOCH_ID \
    --image_size 256 --f 8 --num_in_channels 4 --num_out_channels 4 \
    --nf 256 --ch_mult 1 2 2 2 --attn_resolution 16 8 --num_res_blocks 2 \
    --model_type $MODEL_TYPE \
    --method {method} --num_steps {num_steps} \
    --master_port $MASTER_PORT  --num_process_per_node {num_gpus} \
    --n_sample 50_000 \
    --cfg_scale {cfg_scale} \
    --compute_fid --output_log $OUTPUT_LOG \
    --use_origin_adm \
    # --faster_test \
    # --num_classes 1 \
    # --load_ema \
    # --num_classes 1000 --label_dim 1000 --label_dropout 0.15 \
    # --measure_time \
    # --num_head_channels 64 \
    # --use_karras_samplers \



"""

###### ARGS
env_path = ["../envs/flow1.8.1/", "../envs/flow_pytorch2.2/"][0]
model_type = ["DiT-L/2", "adm", "DiT-B/2"][1]
dataset = ["celeba_256", "ffhq_256", "imagenet_256"][0]
exp = "celeba_f8_adm_lr2e-5_huber1e-2" 
BASE_PORT = 8025
num_gpus = 1
device = "0," #,2,3,4,5,6,7"

config = pd.DataFrame({
    "epochs": [450],
    "num_steps": [0],
    "methods": ['dopri5'],
    "cfg_scale": [1.],
})
print(config)

###################################
slurm_file_path = f"/lustre/scratch/client/vinai/users/haopt12/cnf_flow/slurm_scripts/{exp}/run.sh"
slurm_output = f"/lustre/scratch/client/vinai/users/haopt12/cnf_flow/slurm_scripts/{exp}/"
output_log = f"{slurm_output}/log"
os.makedirs(slurm_output, exist_ok=True)
job_name = "test"

for idx, row in config.iterrows():
    # device = str(idx % 2)
    # slurm_file_path = f"/lustre/scratch/client/vinai/users/haopt12/cnf_flow/slurm_scripts/{exp}/run{device}.sh"
    slurm_command = slurm_template.format(
        job_name=job_name,
        model_type=model_type,
        dataset=dataset,
        exp=exp,
        epoch=row.epochs,
        master_port=str(BASE_PORT+idx),
        slurm_output=slurm_output,
        num_gpus=num_gpus,
        output_log=output_log,
        method=row.methods,
        num_steps=row.num_steps,
        device=device,
        cfg_scale=row.cfg_scale,
        env_path=env_path,
    )
    mode = "w" if idx == 0 else "a"
    with open(slurm_file_path, mode) as f:
        f.write(slurm_command)
print("Slurm script is saved at", slurm_file_path)

# print(f"Summited {slurm_file_path}")
# subprocess.run(['sbatch', slurm_file_path])
