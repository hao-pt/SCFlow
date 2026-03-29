export MASTER_PORT=13004
NUM_GPUS=1
echo MASTER_PORT=${MASTER_PORT}
export PYTHONPATH=$(pwd):$PYTHONPATH

MODEL_TYPE=adm
EPOCH_ID=200
DATASET=celeba_256
METHOD=euler
STEPS=4
EXP=celeba_f8_adm_distill
EXP_DIR=pretrained  # dir of model checkpoints

# use --stochastic arg to use a modified ODE function with a control of stochasticity through --beta arg to generate better output images
# beta=1: resample a random noise at each step like ddpm, beta=0.: no stochasticity.

python test.py \
        --exp ${EXP} \
        --exp_root_dir pretrained \
        --dataset ${DATASET} \
        --batch_size 128 \
        --epoch_id ${EPOCH_ID} \
        --image_size 256 \
        --f 8 \
        --num_in_channels 4 \
        --num_out_channels 4 \
        --nf 256 \
        --ch_mult 1 2 2 2 \
        --attn_resolution 16 8 \
        --num_res_blocks 2 \
        --use_origin_adm \
        --num_head_channels 32 \
        --master_port $MASTER_PORT \
        --num_process_per_node $NUM_GPUS \
        --use_karras_samplers \
        --method ${METHOD} --num_steps ${STEPS} \
        --compute_fid --output_log ${EXP}_${EPOCH_ID}_${METHOD}${STEPS}.log \
        # --stochastic --beta 0.9 \
        # --plot_traj
        # --measure_time \