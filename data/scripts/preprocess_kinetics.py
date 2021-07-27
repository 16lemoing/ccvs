import argparse
import cv2
import random
from copy import deepcopy

from joblib import Parallel
from joblib import delayed

import os, sys, inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
code_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, code_dir)

from tools.utils import mkdir
from data.folder_dataset import make_dataset

VID_EXTENSIONS = ['.avi', '.mp4']


def main(args):

    if not args.src_folder.endswith("/"):
        args.src_folder += "/"

    # prepare folders
    out_folder = os.path.join(args.out_root, args.out_name)
    mkdir(args.out_root)
    mkdir(out_folder)

    # retrieve video files
    vid_files = get_vid_files(args.src_folder)
    random.shuffle(vid_files)
    total = len(vid_files)

    Parallel(n_jobs=args.num_workers)(
        delayed(process_video)(vid_file, out_folder, i, total, args) for i, vid_file in enumerate(vid_files))


def process_video(vid_file, out_folder, i, total, args):
    out_file = None
    curr_file = vid_file.replace(args.src_folder, "")
    check_file = curr_file
    if os.path.exists(os.path.join(out_folder, check_file)):
        print(f'Processed {i + 1} out of {total} (*)', end='\r')
        return

    try:
        vidcap = cv2.VideoCapture(vid_file)
        width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # prepare resize function
        if args.resize is None:
            resize = lambda x: x
        else:
            if width > height:
                width = int(args.resize * width / height)
                height = args.resize
            else:
                height = int(args.resize * height / width)
                width = args.resize
            dim = deepcopy((width, height))
            resize = lambda x: cv2.resize(x, dim)

        # prepare crop function
        if args.square_crop:
            if width > height:
                s = (width - height) // 2
                crop = lambda x: x[:, s:s+height]
                width = height
            else:
                s = (height - width) // 2
                crop = lambda x: x[s:s + width]
                height = width
        else:
            crop = lambda x: x

        # prepare output
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = vidcap.get(cv2.CAP_PROP_FPS)
        vidwri, out_file = get_vidwri(args, out_folder, curr_file, fourcc, fps, width, height, idx=1)

        success, image = vidcap.read()
        count = 0

        assert success
        image = crop(resize(image))
        curr_len = 0
        curr_chunk = 1

        while True:
            if count % args.skip_frames == 0:
                curr_len += 1
                vidwri.write(image)

                if args.max_vid_len is not None and curr_len >= args.max_vid_len:
                    curr_chunk += 1
                    vidwri, out_file = get_vidwri(args, out_folder, curr_file, fourcc, fps, width, height, idx=curr_chunk)
                    curr_len = 0
            success, next_image = vidcap.read()
            if not success:
                break
            next_image = crop(resize(next_image))
            image = next_image
            count += 1
        vidcap.release()
        print(f'Processed {i + 1} out of {total}   ', end='\r')
    except:
        # remove potentially corrupted file
        if out_file is not None and os.path.exists(out_file):
            os.remove(out_file)
        print(f'Skipping {i + 1}                   ', end='\r')


def get_vid_files(src_folder):
    return make_dataset(src_folder, recursive=True, from_vid=True)


def get_path(out_folder, file, override, idx=None):
    out_file = os.path.join(out_folder, file)
    if idx is not None:
        f, ext = os.path.splitext(out_file)
        out_file = f + f"_{idx}" + ext
    if not override:
        assert not os.path.exists(out_file)
    return out_file


def get_vidwri(args, out_folder, curr_file, fourcc, fps, width, height, idx=None):
    out_file = get_path(out_folder, curr_file, args.override, idx)
    mkdir(os.path.dirname(out_file))
    vidwri = cv2.VideoWriter(out_file, fourcc, fps, (width, height))
    return vidwri, out_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_folder', type=str, required=True)
    parser.add_argument('--out_root', type=str, required=True)
    parser.add_argument('--out_name', type=str, required=True)
    parser.add_argument('--override', action='store_true')
    parser.add_argument('--skip_frames', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--max_vid_len', type=int, default=None)
    parser.add_argument('--resize', type=int, default=None)
    parser.add_argument('--square_crop', action='store_true')
    args = parser.parse_args()
    main(args)