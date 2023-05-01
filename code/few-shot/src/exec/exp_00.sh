EXP_NAME="exp_00"
GPU="3"

export CUDA_VISIBLE_DEVICES=$GPU

python train.py --exp_name $EXP_NAME \
    --cuda

