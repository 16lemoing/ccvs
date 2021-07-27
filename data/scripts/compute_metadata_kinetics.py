import argparse
from glob import glob
from torchvision.datasets.video_utils import VideoClips

import os, sys, inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
code_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, code_dir)

from data.cat import KINETICS600_CAT
from tools.utils import serialize, deserialize

def compute_data(args, data_path):
    assert args.data_folder is not None
    vid_paths = glob(f"{args.data_folder}**/*.mp4", recursive=True)
    vid_desc = [path.split('/')[-2] for path in vid_paths]
    vid_labels = [KINETICS600_CAT.index(s) for s in vid_desc]
    data = {"vid_paths":vid_paths, "vid_labels":vid_labels}
    serialize(data_path, data)
    return data

def compute_metadata(args):
    # Prepare paths
    dataroot = "datasets/kinetics"
    if args.name is not None:
        metadata_path = os.path.join(dataroot, f"{args.name}_{args.phase}_metadata.pkl")
        data_path = os.path.join(dataroot, f"{args.name}_{args.phase}_data.pkl")
    else:
        metadata_path = os.path.join(dataroot, f"{args.phase}_metadata.pkl")
        data_path = os.path.join(dataroot, f"{args.phase}_data.pkl")
    assert not os.path.exists(metadata_path), f"Metadata path {metadata_path} already exists, choose another name"

    # Retrieve data paths
    if not os.path.exists(data_path):
        data = compute_data(args, data_path)
        assert os.path.exists(data_path)
    else:
        data = deserialize(data_path)

    # Compute clips
    clips = VideoClips(data['vid_paths'], clip_length_in_frames=1, frames_between_clips=1, num_workers=args.num_workers)

    # Save metadata
    serialize(metadata_path, clips.metadata)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', type=str)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--name', type=str, default=None)
    parser.add_argument('--data_folder', type=str, default=None)
    args = parser.parse_args()
    compute_metadata(args)