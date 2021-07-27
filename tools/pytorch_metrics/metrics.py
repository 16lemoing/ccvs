import piq
import argparse
from glob import glob
from joblib import Parallel, delayed
import numpy as np
import cv2
from tqdm import tqdm
import os
import torch
from skimage.metrics import structural_similarity as ssim_metric

def get_lpips(x, y):
    return piq.LPIPS(reduction='mean')(x, y)

def get_ssim(x, y):
    x = x.cpu().numpy()
    y = y.cpu().numpy()
    ssim = 0
    for i in range(len(x)):
        ssim += (ssim_metric(x[i, 0], y[i, 0]) + ssim_metric(x[i, 1], y[i, 1]) + ssim_metric(x[i, 2], y[i, 2])) / 3
    ssim = ssim / len(x)
    return torch.tensor(ssim)

def get_psnr(x, y):
    return piq.psnr(x, y, data_range=1., reduction='mean')

def metrics_from_files(real_video_files, generated_video_files, resize, num_workers, print_256, idx):
    # both have dimensionality [NUMBER_OF_VIDEOS, VIDEO_LENGTH, FRAME_WIDTH, FRAME_HEIGHT, 3] with values in 0-255
    batch_size = 16
    total_size = len(real_video_files)

    if len(idx) == 0:
        lpips, ms_ssim, ssim, psnr = [], [], [], []
    else:
        lpips, ms_ssim, ssim, psnr = [[] for _ in idx], [[] for _ in idx], [[] for _ in idx], [[] for _ in idx]

    with torch.no_grad():
        for i in tqdm(range(total_size // batch_size)):
            start = i * batch_size
            end = min(start + batch_size, total_size)
            real_videos_np = load_videos([real_video_files[i] for i in range(start, end)], resize, num_workers)
            generated_videos_np = load_videos([generated_video_files[i] for i in range(start, end)], resize, num_workers)
            real_videos = torch.tensor(real_videos_np).cuda() / 255
            generated_videos = torch.tensor(generated_videos_np).cuda() / 255
            if len(idx) == 0:
                real_videos = real_videos.view(-1, *real_videos.shape[2:]).permute(0, 3, 1, 2)
                generated_videos = generated_videos.view(-1, *generated_videos.shape[2:]).permute(0, 3, 1, 2)
                lpips.append(get_lpips(real_videos, generated_videos).cpu())
                ssim.append(get_ssim(real_videos, generated_videos).cpu())
                psnr.append(get_psnr(real_videos, generated_videos).cpu())
            else:
                for k, frame_idx in enumerate(idx):
                    real_frames = upscale(real_videos[:, frame_idx].permute(0, 3, 1, 2))
                    generated_frames = upscale(generated_videos[:, frame_idx].permute(0, 3, 1, 2))
                    lpips[k].append(get_lpips(real_frames, generated_frames).cpu())
                    ssim[k].append(get_ssim(real_frames, generated_frames).cpu())
                    psnr[k].append(get_psnr(real_frames, generated_frames).cpu())

    if print_256 and len(idx) == 0:
        lpips = torch.stack(lpips).view(-1, 256//batch_size)
        lpips_m = lpips.mean()
        lpips_std = lpips.mean(1).std()
        ssim = torch.stack(ssim).view(-1, 256 // batch_size)
        ssim_m = ssim.mean()
        ssim_std = ssim.mean(1).std()
        psnr = torch.stack(psnr).view(-1, 256//batch_size)
        psnr_m = psnr.mean()
        psnr_std = psnr.mean(1).std()
        print(f"(256) LPIPS is: {lpips_m} (+- {lpips_std}), SSIM is {ssim_m} (+- {ssim_std}), PSNR is {psnr_m} (+- {psnr_std})")
    elif len(idx) == 0:
        lpips_m = torch.stack(lpips).mean()
        ssim_m = torch.stack(ssim).mean()
        psnr_m = torch.stack(psnr).mean()
    else:
        lpips_m = [torch.stack(e).mean() for e in lpips]
        ssim_m = [torch.stack(e).mean() for e in ssim]
        psnr_m = [torch.stack(e).mean() for e in psnr]
    return lpips_m, ssim_m, psnr_m

def load_video(file, resize):
    vidcap = cv2.VideoCapture(file)
    success, image = vidcap.read()
    frames = []
    while success:
        if resize is not None:
            h, w = resize
            image = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)
        frames.append(image)
        success, image = vidcap.read()
    return np.stack(frames)

def get_video_files(folder):
    return sorted(glob(os.path.join(folder, "*.mp4")))

def load_videos(video_files, resize, num_workers):
    videos = Parallel(n_jobs=num_workers)(delayed(load_video)(file, resize) for file in video_files)
    return np.stack(videos)

def get_folder(exp_tag, fold_i=None):
    if fold_i is not None:
        exp_tag += f"_{fold_i}"
    all_folders = glob(f"results/*{exp_tag}")
    assert len(all_folders) == 1, f"Too many possibilities for this tag {exp_tag}:\n{all_folders}"
    return all_folders[0]

def get_folders(exp_tag, num_folds):
    if num_folds is not None:
        folders= []
        for i in range(num_folds):
            folders.append(get_folder(exp_tag, i))
        return folders
    else:
        return [get_folder(exp_tag)]

def upscale(videos, min_size=161):
    h, w = videos.shape[-2:]
    if h >= min_size and w >= min_size:
        return videos
    else:
        if h < w:
            size = [min_size, int(min_size * w / h)]
        else:
            size = [int(min_size * h / w), min_size]
        return torch.nn.functional.interpolate(videos, size=size, mode='bilinear')

def print_scores(scores, name):
    print(f"Individual {name} scores")
    print(scores)
    print(f"Mean/std of {name} across {len(scores)} runs")
    print(np.mean(scores), np.std(scores))

def main(args):
    fake_folders = get_folders(args.exp_tag, args.num_folds)
    real_tag = args.exp_tag if args.real_tag is None else args.real_tag
    real_folders = get_folders(real_tag, args.num_folds)

    if len(args.idx) == 0:
        lpips, ssim, psnr = [], [], []
    else:
        lpips, ssim, psnr = [[] for _ in args.idx], [[] for _ in args.idx], [[] for _ in args.idx]
    for i, (real_root, fake_root) in tqdm(enumerate(zip(sorted(real_folders), sorted(fake_folders)))):
        print(f"[{i}] Loading real")
        real_video_files = get_video_files(os.path.join(real_root, args.real_folder))
        print(f"Found {len(real_video_files)} {args.real_folder} video files")

        print(f"[{i}] Loading fake")
        fake_video_files = get_video_files(os.path.join(fake_root, args.fake_folder))
        print(f"Found {len(fake_video_files)} {args.fake_folder} video files")

        assert len(real_video_files) == len(fake_video_files)

        print(f"[{i}] Computing metrics")
        lpips_i, ssim_i, psnr_i = metrics_from_files(real_video_files, fake_video_files, args.resize, args.num_workers, args.print_256, args.idx)
        if len(args.idx) == 0:
            lpips.append(lpips_i)
            ssim.append(ssim_i)
            psnr.append(psnr_i)
        else:
            for k in range(len(args.idx)):
                lpips[k].append(lpips_i[k])
                ssim[k].append(ssim_i[k])
                psnr[k].append(psnr_i[k])

    if len(args.idx) == 0:
        print_scores(lpips, "LPIPS")
        print_scores(ssim, "SSIM")
        print_scores(psnr, "PSNR")
    else:
        for k in range(len(args.idx)):
            print_scores(lpips[k], f"LPIPS-{k}")
            print_scores(ssim[k], f"SSIM-{k}")
            print_scores(psnr[k], f"PSNR-{k}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_tag', type=str, default=None)
    parser.add_argument('--real_tag', type=str, default=None)
    parser.add_argument('--real_folder', type=str, default="real")
    parser.add_argument('--fake_folder', type=str, default="fake")
    parser.add_argument('--num_folds', type=int, default=None)
    parser.add_argument('--idx', type=int, nargs="+", default=[])
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--print_256', action='store_true')
    parser.add_argument('--resize', type=int, nargs="+", default=None)
    args = parser.parse_args()
    main(args)