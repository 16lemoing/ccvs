import argparse
import random
from torchvision.datasets.video_utils import VideoClips

import os, sys, inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
code_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, code_dir)

from tools.utils import serialize, deserialize
from data.folder_dataset import make_dataset
from data.cat import KINETICS600_CAT

def get_data(args):
    dataroot = "datasets/kinetics"
    data_path = os.path.join(dataroot, f"{args.name}_{args.phase}_data.pkl")
    if not os.path.exists(data_path):
        vid_folder = os.path.join(dataroot, "preprocessed_videos", f"{args.phase}_{args.name}")
        assert os.path.exists(vid_folder)
        vid_paths = make_dataset(vid_folder, recursive=True, from_vid=True)
        vid_desc = [s.split('/')[4] for s in vid_paths]
        vid_labels = [KINETICS600_CAT.index(s) for s in vid_desc]
        data = {"vid_paths": vid_paths, "vid_labels": vid_labels}
        serialize(data_path, data)
    else:
        data = deserialize(data_path)
    return data

def get_metadata(args, data):
    dataroot = "datasets/kinetics"
    metadata_path = os.path.join(dataroot, f"{args.name}_{args.phase}_metadata.pkl")
    if not os.path.exists(metadata_path):
        vid_clips = VideoClips(data["vid_paths"],
                               clip_length_in_frames=16,
                               frames_between_clips=16,
                               num_workers=8,
                               _precomputed_metadata=None)
        metadata = vid_clips.metadata
        serialize(metadata_path, metadata)
    else:
        metadata = deserialize(metadata_path)
    return metadata


def compute_folds(args):
    # Load data
    dataroot = "datasets/kinetics"

    print("Loading data")
    data = get_data(args)

    print("Loading metadata")
    metadata = get_metadata(args, data)

    assert metadata["video_paths"] == data["vid_paths"]

    # Prepare fold indices
    num_excerpts = len(metadata["video_paths"])
    num_excerpts_per_fold = num_excerpts // args.num_folds
    indices = list(range(num_excerpts))
    random.shuffle(indices)
    if args.max_per_fold is not None and args.num_folds * args.max_per_fold < num_excerpts:
        num_excerpts = args.num_folds * args.max_per_fold
        num_excerpts_per_fold = args.max_per_fold
        indices = indices[:num_excerpts]
    print("Total excerpts", num_excerpts)
    print("Num excerpts per fold", num_excerpts_per_fold)

    for i in range(args.num_folds):
        print(f"Computing fold data {i+1} / {args.num_folds}")
        fold_indices = indices[i * num_excerpts_per_fold: (i + 1) * num_excerpts_per_fold]
        fold_data = {k:[v[j] for j in fold_indices] for k,v in data.items()}
        fold_metadata = {k: [v[j] for j in fold_indices] for k, v in metadata.items()}
        fold_metadata_path = os.path.join(dataroot, f"folds/{args.name}_{i}_{args.phase}_metadata.pkl")
        fold_data_path = os.path.join(dataroot, f"folds/{args.name}_{i}_{args.phase}_data.pkl")
        print(f"Saving fold data {i + 1} / {args.num_folds}")
        serialize(fold_metadata_path, fold_metadata)
        serialize(fold_data_path, fold_data)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', type=str)
    parser.add_argument('num_folds', type=int)
    parser.add_argument('name', type=str)
    parser.add_argument('--max_per_fold', type=int, default=None)
    args = parser.parse_args()
    compute_folds(args)