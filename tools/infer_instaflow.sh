# can do inference from coco2014/coco2017 prompts
# by setting --prompts=coco2014-30k or --prompts=coco2017-5k

CUDA_VISIBLE_DEVICES=0 torchrun --nnodes=1 --rdzv_endpoint 0.0.0.0:10985 --nproc_per_node=1 sd_distill/infer.py \
    --guidance 1 \
    --batch-size 32 \
    --step 1 \
    --pretrained-unet-ckpt pretrained/scflow_t2i/checkpoint-18000/ \
    --prompts='A hyper-realistic photo of a cute cat.'