EXP_NAME="exp_00c"
SPLIT="val"
GPU="4"
SIDE_SIZE="256"
CROP_SIZE="256"
NUM_FRAMES="32"
SAMPLING_RATE="2"
FPS="30"
SLOWFAST_ALPHA="4"
NUM_CLIPS="10"
NUM_CROPS="3"
MODEL_ZOO="facebookresearch/pytorchvideo"
MODEL_ARCH="slowfast_r50"
DEVICE="cuda"

python video_inference_v0.py --exp_name $EXP_NAME \
    --split $SPLIT \
    --gpu $GPU \
    --side_size $SIDE_SIZE \
    --crop_size $CROP_SIZE \
    --num_frames $NUM_FRAMES \
    --sampling_rate $SAMPLING_RATE \
    --fps $FPS \
    --slowfast_alpha $SLOWFAST_ALPHA \
    --num_clips $NUM_CLIPS \
    --num_crops $NUM_CROPS \
    --model_zoo $MODEL_ZOO \
    --model_arch $MODEL_ARCH \
    --device $DEVICE


