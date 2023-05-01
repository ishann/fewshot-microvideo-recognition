import torch
from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo,
)
from pytorchvideo.transforms import (
    ShortSideScale,
    UniformTemporalSubsample
) 

class PackPathway(torch.nn.Module):
    """
    Transform for converting video frames as a list of tensors. 
    """
    def __init__(self, args):
        super().__init__()
        self.slowfast_alpha = args.slowfast_alpha
        
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

def transform_video(args):

    return Compose(
        [
            UniformTemporalSubsample(args.num_frames),
            Lambda(lambda x: x/255.0),
            NormalizeVideo(args.mean, args.std),
            ShortSideScale(size=args.side_size),
            CenterCropVideo(args.crop_size),
            PackPathway(args)
        ]
    )

