"""
Built off of SlowFast's video inference script.

"""
import argparse
import ipdb
import os
from setproctitle import setproctitle
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

import torch
from pytorchvideo.data.encoded_video import EncodedVideo

from video_utils import transform_video

def main(args):

    # Set things up.
    setproctitle(args.exp_name)
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    VIDEOS_PATH = os.path.join("../../data/", args.split)

    # Initialize model.
    model = torch.hub.load(args.model_zoo, args.model_arch, pretrained=True)
    model = model.eval()
    model = model.to(args.device)

    # Inference setup.
    clip_duration = (args.num_frames * args.sampling_rate)/args.fps
    start_sec = 0
    end_sec = start_sec + clip_duration
    video_files = os.listdir(VIDEOS_PATH)
    transform_ = transform_video(args)

    embeddings = torch.Tensor(len(video_files), args.embedding_dim)
    filenames = []
    #ipdb.set_trace()

    # Inference.
    for idx, video_file in tqdm(enumerate(video_files)):

        video_path = os.path.join(VIDEOS_PATH, video_file)
        video = EncodedVideo.from_path(video_path)

        video_data = video.get_clip(start_sec=start_sec,
                                    end_sec=int(float(video.duration)))
        video_data = video_data["video"]
        transformed_video = transform_(video_data)

        inputs = [i.to(args.device)[None, ...] for i in transformed_video]

        preds = model(inputs)

        embeddings[idx] = preds.data.cpu()
        filenames.append(video_file)

    # Add embeddings and filenames to a dictionary and save to disk.
    embeddings_dict = {"embeddings": embeddings, "filenames": filenames}
    embedding_filename = os.path.join("../../data/embeddings/{}_{}_embeddings.pt".format(args.exp_name, args.split))
    torch.save(embeddings_dict, embedding_filename)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = "Video inference. Nothing more, nothing less, nothing else...")
    parser.add_argument("--exp_name", help="Identify experiments on nvidia-smi.",
                        default="an_expt_has_no_name", required=True)
    parser.add_argument("--split", help="Split for train/ test/ whatever.",
                        default="split", required=True)
    parser.add_argument("--gpu", help="Be nice. Use only one GPU.",
                        default="4", required=True)
    parser.add_argument("--embedding_dim", help="SlowFastR50 embedding dimension.",
                        default=400, type=int, required=False)
    parser.add_argument("--side_size", help="Video frame size.",
                        default=256, type=int, required=True)
    parser.add_argument("--mean", help="Mean of video frames.",
                        default=[0.45, 0.45, 0.45], type=list, required=False)
    parser.add_argument("--std", help="Mean of video frames.",
                        default=[0.225, 0.225, 0.225], type=list, required=False)
    parser.add_argument("--crop_size", help="Video frame crop size.",
                        default=256, type=int, required=True)
    parser.add_argument("--num_frames", help="Number of video frames.",
                        default=32, type=int, required=True)
    parser.add_argument("--sampling_rate", help="Video frame sampling rate.",
                        default=2, type=int, required=True)
    parser.add_argument("--fps", help="Frames per second.",
                        default=30, type=int, required=True)
    parser.add_argument("--slowfast_alpha", help="SlowFast alpha.",
                        default=4, type=int, required=True)
    parser.add_argument("--num_clips", help="Number of video clips.",
                        default=10, type=int, required=True)
    parser.add_argument("--num_crops", help="Number of video crops.",
                        default=3, type=int, required=True)
    parser.add_argument("--model_zoo", help="Model zoo.",
                        default="facebookresearch/pytorchvideo", required=True)
    parser.add_argument("--model_arch", help="Model architecture.",
                        default="slowfast_r50", required=True)
    parser.add_argument("--device", help="Device: cpu / gpu.",
                        default="cuda", required=True)
    args = parser.parse_args()

    for k, v in vars(args).items():
        print(f"{k}: {v}")

    main(args)


