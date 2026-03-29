export MASTER_PORT=6034
NUM_GPUS=1

python train.py \
	--exp celeb_adm_scflow \
	--init_threshold 0.1 \
	--trunc_threshold 0.4 \
	--dataset celeba_256 \
	--datadir /share/elor/htp26/data/celeba-tfr/celeba-lmdb \
	--batch_size 32 \
	--ch_mult 1 2 2 2 \
	--attn_resolution 16 8 \
	--target_ema_decay 0.9 \
	--master_port $MASTER_PORT \
	--num_process_per_node $NUM_GPUS \
	--num_sample_timesteps 4 \
	--warm_up_reflow 1_000 \
	--warm_up_gan 0 \
	--warm_up_inverse 1_000 \
	--num_epoch 200 \
	--model_ckpt pretrained/celeb256_f8_adm/model_480.pth \
	--save_content \
	--save_content_every 10 \
	--lr 2e-5 \
	--lrD 1e-4 --d_base_channels 16384 --d_temb_channels 256 --r1_gamma 1. \
	--use_origin_adm \
	--use_gan \
    --scale_inverse 0.1 \
    --scale_gan 0.1 \
    --scale_reflow 0.1