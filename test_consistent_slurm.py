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
#SBATCH --cpus-per-gpu=8
#SBATCH --partition=research
#SBATCH --mail-type=all
#SBATCH --mail-user=v.haopt12@vinai.io
#SBATCH --ntasks=1

# module purge
# module load python/miniconda3/miniconda3
# eval "$(conda shell.bash hook)"
# conda activate /lustre/scratch/client/vinai/users/ngocbh8/quan/envs/flow
# cd /lustre/scratch/client/vinai/users/ngocbh8/quan/cnf_flow

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
export METHOD={method}
export STEPS={num_steps}

echo "----------------------------"
echo $MODEL_TYPE $EPOCH_ID $DATASET $EXP $METHOD $STEPS
echo "----------------------------"

CUDA_VISIBLE_DEVICES={device} python test_consistent_flow.py --exp $EXP \
        --dataset $DATASET --batch_size 100 --epoch_id $EPOCH_ID \
        --image_size 256 --f 8 --num_in_channels 4 --num_out_channels 4 \
        --nf 256 --ch_mult 1 2 2 2 --attn_resolution 16 8 --num_res_blocks 2 \
        --master_port $MASTER_PORT --num_process_per_node {num_gpus} \
        --use_karras_samplers \
        --method $METHOD --num_steps $STEPS \
        --use_origin_adm \
        --compute_fid --output_log $OUTPUT_LOG \
        --exp_root_dir {exp_root_dir} \
        --stochastic --beta {beta} \
        # --num_head_channels 32 \
        # --model_type DiT-L/2 --num_classes 1 --label_dropout 0. --faster_test \
        # --measure_time \
        # --compute_nfe \

"""

###### ARGS
model_type = ["DiT-L/2", "adm"][1]
dataset = ["celeba_256", "ffhq_256"][0]
exp = "celeba_f8_adm_lr2e-5_100steps_ema0.95_fmloss_skip20_notrunct_gan_orihuber0.01"# "celeba_f8_adm_lr2e-5_100steps_ema0.95_fmloss_skip20_notrunct_gan_huber0.1" # "celeba_f8_adm_lr2e-5_100steps_ema0.95_fmloss_skip20_notrunct_gan_huber0.1" # "celeba_f8_adm_lr2e-5_100steps_ema0._fmloss_skip20" # "celeba_f8_adm_lr2e-5_100steps_ema0.95_fmloss_skip20_gan_notrunct_huber0.01" # ["celeba_f8_adm_lr2e-5_100steps_ema0.95_fmloss_skip20_gan_notrunct_huber0.01", "celeba_f8_adm_lr2e-5_100steps_ema0._fmloss_skip20_gan_notrunct", "celeba_f8_adm_lr2e-5_100steps_ema0.95_fmloss_skip20_notrunct", "celeba_f8_adm_lr2e-5_100steps_ema0.95_fmloss_skip20_gan_skipteacher"][-1]
# "con_fm" #  "celeba_f8_adm_lr2e-5_100steps_ema0.95_fmloss_skip1_gan" # ""celeba_f8_adm_lr2e-5_100steps_ema0._fmloss_skip20_gan" # "celeba_f8_dit_lr1e-4_100steps_ema0.95_fmloss_skip20_gan_skipteacher_warmup15k" # "celeba_f8_adm_lr2e-5_100steps_ema0.95_fmloss_skip20_gan_warmup15k" # "celeba_f8_adm_lr2e-5_100steps_ema0.95_fmloss_skip20_gan_songbound0.2_warmup15k" # "# "celeba_f8_dit_lr1e-4_100steps_ema0.95_fmloss_skip20_gan_skipteacher"  "celeba_f8_adm_lr2e-5_100steps_ema0.9_fmloss_skip30_gan_skipteacher"
exp_root_dir = ["ct_flow", "cd_flow"][1]
BASE_PORT = 8026
num_gpus = 1
device = "0" #,2,3,4,5,6,7"

config = pd.DataFrame({
    "epochs": [200]*6,
    "num_steps": [1, 2, 4, 8, 12, 16],
    "methods": ['euler']*6,
    "cfg_scale": [1]*6,
    "beta": [0.]*6
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
    os.makedirs(slurm_output, exist_ok=True)
    slurm_command = slurm_template.format(
        job_name=job_name,
        model_type=model_type,
        dataset=dataset,
        exp=exp,
        exp_root_dir=exp_root_dir,
        epoch=row.epochs,
        master_port=str(BASE_PORT+idx),
        slurm_output=slurm_output,
        num_gpus=num_gpus,
        output_log=output_log,
        method=row.methods,
        num_steps=row.num_steps,
        device=device,
        cfg_scale=row.cfg_scale,
        beta=row.beta,
    )
    mode = "w" if idx == 0 else "a"
    with open(slurm_file_path, mode) as f:
        f.write(slurm_command)
print("Slurm script is saved at", slurm_file_path)

# print(f"Summited {slurm_file_path}")
# subprocess.run(['sbatch', slurm_file_path])
