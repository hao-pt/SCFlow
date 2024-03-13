#!/bin/sh
#SBATCH --job-name=rf01 # create a short name for your job
#SBATCH --output=/lustre/scratch/client/vinai/users/haopt12/flow_distill/slurms/slurm_%A.out # create a output file
#SBATCH --error=/lustre/scratch/client/vinai/users/haopt12/flow_distill/slurms/slurm_%A.err # create a error file
#SBATCH --partition=research # choose partition
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-gpu=16 # 80
#SBATCH --mem-per-gpu=32GB
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=10-00:00          # total run time limit (DD-HH:MM)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-type=fail          # send email when job fails
#SBATCH --mail-user=v.haopt12@vinai.io

set -x
set -e
export MASTER_PORT=6033
NUM_GPUS=2

# module purge
# module load python/miniconda3/miniconda3
# eval "$(conda shell.bash hook)"
# conda activate ../envs/flow1.8.1/
# # conda activate ../envs/flow_pytorch2.2/

## celeb adm
python train_consistent_flow_distill.py \
	--exp celeb_adm_threshold1e-1_trunc4e-1_scale1e-1x2_rf1 \
	--init_threshold 0.1 \
	--trunc_threshold 0.4 \
	--dataset celeba_256 \
	--datadir ../data/celeba_256/celeba-lmdb/ \
	--batch_size 32 \
	--ch_mult 1 2 2 2 \
	--attn_resolution 16 8 \
	--target_ema_decay 0.9 \
	--master_port $MASTER_PORT \
	--num_process_per_node $NUM_GPUS \
	--num_sample_timesteps 4 \
	--warm_up_reflow 0 \
	--warm_up_gan 1_000 \
	--warm_up_inverse 1_000 \
	--num_epoch 200 \
	--model_ckpt ../flow_public_models/celeb256_f8_adm/model_480.pth \
	--save_content \
	--save_content_every 10 \
	--lr 2e-5 \
	--lrD 1e-4 --d_base_channels 16384 --d_temb_channels 256 --r1_gamma 1. \
	--use_origin_adm \
	--use_gan \
        --scale_inverse 0.1 \
        --scale_gan 0.1 \
        --scale_reflow 1 \

## celeb dit
# python train_consistent_flow_distill.py \
# 	--exp celeb_dit_threshold5e-2_trunc4e-1 \
# 	--init_threshold 0.05 \
# 	--trunc_threshold 0.4 \
# 	--dataset celeba_256 \
# 	--datadir ../data/celeba_256/celeba-lmdb/ \
# 	--batch_size 8 \
# 	--ch_mult 1 2 2 2 \
# 	--attn_resolution 16 8 \
# 	--target_ema_decay 0.9 \
# 	--master_port $MASTER_PORT \
# 	--num_process_per_node $NUM_GPUS \
# 	--num_sample_timesteps 4 \
# 	--warm_up_con 0 \
# 	--warm_up_gan 1_000 \
# 	--warm_up_inverse 1_000 \
# 	--num_epoch 200 \
# 	--model_ckpt ../flow_public_models/celeb_f8_dit/model_475.pth \
# 	--save_content \
# 	--save_content_every 10 \
# 	--lr 1e-4 \
# 	--lrD 5e-4 --d_base_channels 16384 --d_temb_channels 256 --r1_gamma 1. \
# 	--use_gan \
#    	--model_type DiT-L/2 --num_classes 1 --label_dropout 0. \
#    	--faster_training \
#    	# --compile \

## ffhq
# python train_consistent_flow_distill.py \
# 	--exp ffhq_adm_threshold1e-1_trunc4e-1 \
# 	--init_threshold 0.1 \
# 	--trunc_threshold 0.4 \
# 	--dataset ffhq_256 \
# 	--datadir ../data/ffhq-lmdb/ \
# 	--batch_size 32 \
# 	--ch_mult 1 2 3 4 \
# 	--attn_resolution 16 8 4 \
# 	--target_ema_decay 0.9 \
# 	--master_port $MASTER_PORT \
# 	--num_process_per_node $NUM_GPUS \
# 	--num_sample_timesteps 4 \
# 	--warm_up_con 0 \
# 	--warm_up_gan 1_000 \
# 	--warm_up_inverse 1_000 \
# 	--num_epoch 200 \
# 	--model_ckpt ../flow_public_models/ffhq_f8_adm/model_325.pth \
# 	--save_content \
# 	--save_content_every 10 \
# 	--lr 2e-5 \
# 	--lrD 1e-4 --d_base_channels 16384 --d_temb_channels 256 --r1_gamma 1. \
# 	--use_origin_adm \
# 	--use_gan \

