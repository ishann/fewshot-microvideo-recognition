EXP_NAME="ss_video_00"
GPU="1"
DEVICE="cuda"
EPOCHS=10
BATCH_SIZE=2
NUM_WORKERS=4

python train.py  --exp_name $EXP_NAME \
    --batch_size $BATCH_SIZE \
    --device $DEVICE \
    --epochs $EPOCHS \
    --gpu $GPU \
    --num_workers $NUM_WORKERS
