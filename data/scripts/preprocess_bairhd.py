import argparse
import os
from glob import glob
import cv2

from joblib import Parallel
from joblib import delayed

def main(args):
    data_dir = os.path.join(args.data_root, "softmotion_0511")
    print("Preparing train")
    train_output_dir = os.path.join(args.data_root, f"original_frames_{args.dim}/train")
    extract_data(data_dir, train_output_dir, args.dim, init_k=0, end_k=43264)
    print("Preparing test")
    test_output_dir = os.path.join(args.data_root, f"original_frames_{args.dim}/test")
    extract_data(data_dir, test_output_dir, args.dim, init_k=44120, end_k=44376)

def get_frame_path(frames_dir, i):
    paths = glob(os.path.join(frames_dir, f"aux1_full_cropped_im{i}_*.jpg"))
    assert len(paths) == 1
    return paths[0]

def get_hd_frames(data_dir, k, dim):
    group = k // 1000
    frames_dir = os.path.join(data_dir, f"aux1/traj_group{group}/traj{k}/images")
    frame_paths = [get_frame_path(frames_dir, i) for i in range(30)]
    frames = []
    for path in frame_paths:
        im = cv2.imread(path)
        im = im[:,157:967]
        im = cv2.resize(im, dsize=(dim, dim))
        im = cv2.flip(im, 0)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        frames.append(im)
    return frames

def process_frames(k, output_dir, data_dir_hd, dim):
    frames_out_dir = os.path.join(output_dir, '{0:05}'.format(k))
    os.makedirs(frames_out_dir)
    aux1_frames = get_hd_frames(data_dir_hd, k, dim)
    for i, frame in enumerate(aux1_frames):
        filepath = os.path.join(frames_out_dir, f'{i:02}.png')
        cv2.imwrite(filepath, cv2.cvtColor(frame.astype('uint8'), cv2.COLOR_RGB2BGR))

def extract_data(data_dir, output_dir, dim, init_k, end_k):
    if os.path.exists(output_dir):
        if os.listdir(output_dir):
            raise RuntimeError('Directory not empty: {0}'.format(output_dir))
    else:
        os.makedirs(output_dir)

    Parallel(n_jobs=args.num_workers)(delayed(process_frames)(k, output_dir, data_dir, dim) for k in range(init_k, end_k))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--dim', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=8)
    args = parser.parse_args()
    main(args)