# Import libraries.
import ipdb
import os
import random

from pytorchvideo.data import LabeledVideoDataset
from pytorchvideo.data.encoded_video import EncodedVideo
from pytorchvideo.transforms import (
    #ApplyTransformToKey,
    #Normalize,
    #RandomShortSideScale,
    #RemoveKey,
    ShortSideScale,
    UniformTemporalSubsample)

import torch
import torch.utils.data

from torchvision.transforms._transforms_video import (
     CenterCropVideo,
     NormalizeVideo)
from torchvision.transforms import (
     Compose,
     Lambda)

class LabeledVideoDataset(torch.utils.data.Dataset):

    def __init__(self, device):
        super().__init__()
        self._DATA_PATH = "../../data/pretrain"
        self._MEAN = [0.45, 0.45, 0.45]
        self._STD = [0.225, 0.225, 0.225]
        self._SHORT_SIDE_SCALE = 256
        self._CROP_SCALE = 256
        self._NUM_FRAMES = 32

        self.video_list = os.listdir(self._DATA_PATH)

        self.device = device
        #self.device = device if type(device)==int else "cpu"

        print("Number of videos: ", len(self.video_list))

    def __len__(self):

        return len(self.video_list)

    def transform(self, video):
        transform =  Compose(
        [
            UniformTemporalSubsample(self._NUM_FRAMES),
            Lambda(lambda x: x/255.0),
            NormalizeVideo(self._MEAN, self._STD),
            ShortSideScale(size=self._SHORT_SIDE_SCALE),
            CenterCropVideo(self._CROP_SCALE),
            PackPathway()
        ])
        return transform(video)
    
    def __getitem__(self, idx):

        video_path = os.path.join(self._DATA_PATH, self.video_list[idx])
        video = EncodedVideo.from_path(video_path)
        video_data = video.get_clip(start_sec=0,
                                    end_sec=int(float(video.duration)))
        video_data = video_data["video"]
        transformed_video = self.transform(video_data)
        # [3,8,256,256] and [3,32,256,256]
        
        label = random.randint(0,1)
        
        # shuffle the data
        if label==1:
            # Permute order of frames in both tensors for [CxNxHxW] tensor.
            for i in range(len(transformed_video)):
                transformed_video[i] = transformed_video[i][:, torch.randperm(transformed_video[i].size(1)), :, :]

        #x = [frames.to(self.device)[None, ...] for frames in x]
        
        return transformed_video, label


class PackPathway(torch.nn.Module):
    """
    Transform for converting video frames as a list of tensors. 
    """
    def __init__(self):
        super().__init__()
        self.slowfast_alpha = 4
        
    def forward(self, frames: torch.Tensor):
        fast_pathway = frames
        # Perform temporal sampling from the fast pathway.
        slow_pathway = torch.index_select(
            frames,
            1,
            torch.linspace(
                    0, frames.shape[1] - 1, frames.shape[1] // self.slowfast_alpha
            ).long(),
        )
        frame_list = [slow_pathway, fast_pathway]
        return frame_list

