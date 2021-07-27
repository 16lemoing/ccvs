# Adapted from https://github.com/google-research/google-research/blob/master/frechet_video_distance/frechet_video_distance.py

"""Minimal Reference implementation for the Frechet Video Distance (FVD).
FVD is a metric for the quality of video generation models. It is inspired by
the FID (Frechet Inception Distance) used for images, but uses a different
embedding to be better suitable for videos.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow.compat.v1 as tf
import tensorflow_gan as tfgan
import tensorflow_hub as hub

import argparse
from glob import glob
from joblib import Parallel, delayed
import numpy as np
import cv2
from tqdm import tqdm

import os, sys, inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
code_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, code_dir)

from tools.utils import calculate_frechet_distance

def preprocess(videos, target_resolution):
    """Runs some preprocessing on the videos for I3D model.
    Args:
        videos: <T>[batch_size, num_frames, height, width, depth] The videos to be
            preprocessed. We don't care about the specific dtype of the videos, it can
            be anything that tf.image.resize_bilinear accepts. Values are expected to
            be in the range 0-255.
        target_resolution: (width, height): target video resolution
    Returns:
        videos: <float32>[batch_size, num_frames, height, width, depth]
    """
    videos_shape = videos.shape.as_list()
    all_frames = tf.reshape(videos, [-1] + videos_shape[-3:])
    resized_videos = tf.image.resize_bilinear(all_frames, size=target_resolution)
    target_shape = [videos_shape[0], -1] + list(target_resolution) + [3]
    output_videos = tf.reshape(resized_videos, target_shape)
    scaled_videos = 2. * tf.cast(output_videos, tf.float32) / 255. - 1
    return scaled_videos


def _is_in_graph(tensor_name):
    """Checks whether a given tensor does exists in the graph."""
    try:
        tf.get_default_graph().get_tensor_by_name(tensor_name)
    except KeyError:
        return False
    return True


def create_id3_embedding(videos):
    """Embeds the given videos using the Inflated 3D Convolution network.
    Downloads the graph of the I3D from tf.hub and adds it to the graph on the
    first call.
    Args:
        videos: <float32>[batch_size, num_frames, height=224, width=224, depth=3].
            Expected range is [-1, 1].
    Returns:
        embedding: <float32>[batch_size, embedding_size]. embedding_size depends
                             on the model used.
    Raises:
        ValueError: when a provided embedding_layer is not supported.
    """

    batch_size = 16
    module_spec = "https://tfhub.dev/deepmind/i3d-kinetics-400/1"


    # Making sure that we import the graph separately for
    # each different input video tensor.
    module_name = "fvd_kinetics-400_id3_module_" + six.ensure_str(
            videos.name).replace(":", "_")

    assert_ops = [
            tf.Assert(
                    tf.reduce_max(videos) <= 1.001,
                    ["max value in frame is > 1", videos]),
            tf.Assert(
                    tf.reduce_min(videos) >= -1.001,
                    ["min value in frame is < -1", videos]),
            tf.assert_equal(
                    tf.shape(videos)[0],
                    batch_size, ["invalid frame batch size: ",
                                 tf.shape(videos)],
                    summarize=6),
    ]
    with tf.control_dependencies(assert_ops):
        videos = tf.identity(videos)

    module_scope = "%s_apply_default/" % module_name

    # To check whether the module has already been loaded into the graph, we look
    # for a given tensor name. If this tensor name exists, we assume the function
    # has been called before and the graph was imported. Otherwise we import it.
    # Note: in theory, the tensor could exist, but have wrong shapes.
    # This will happen if create_id3_embedding is called with a frames_placehoder
    # of wrong size/batch size, because even though that will throw a tf.Assert
    # on graph-execution time, it will insert the tensor (with wrong shape) into
    # the graph. This is why we need the following assert.
    video_batch_size = int(videos.shape[0])
    assert video_batch_size in [batch_size, -1, None], "Invalid batch size"
    tensor_name = module_scope + "RGB/inception_i3d/Mean:0"
    if not _is_in_graph(tensor_name):
        i3d_model = hub.Module(module_spec, name=module_name)
        i3d_model(videos)

    # gets the kinetics-i3d-400-logits layer
    tensor_name = module_scope + "RGB/inception_i3d/Mean:0"
    tensor = tf.get_default_graph().get_tensor_by_name(tensor_name)
    return tensor


def calculate_fvd(real_activations, fake_activations):
    """Returns a list of ops that compute metrics as funcs of activations.
    Args:
        real_activations: <float32>[num_samples, embedding_size]
        fake_activations: <float32>[num_samples, embedding_size]
    Returns:
        A scalar that contains the requested FVD.
    """
    return tfgan.eval.frechet_classifier_distance_from_activations(
            real_activations, fake_activations)

def compute_fvd_given_acts(acts_1, acts_2):
    """Computes the FVD of two paths"""
    m1 = np.mean(acts_1, axis=0)
    s1 = np.cov(acts_1, rowvar=False)
    m2 = np.mean(acts_2, axis=0)
    s2 = np.cov(acts_2, rowvar=False)
    fvd_value = calculate_frechet_distance(m1, s1, m2, s2)
    return fvd_value

def emb_from_files(real_video_files, fake_video_files, resize, num_workers):
    # both have dimensionality [NUMBER_OF_VIDEOS, VIDEO_LENGTH, FRAME_WIDTH, FRAME_HEIGHT, 3] with values in 0-255
    batch_size = 16
    total_size = len(real_video_files)
    with tf.Graph().as_default():
        real_emb = []
        fake_emb = []

        ph_videos = tf.convert_to_tensor(load_videos([real_video_files[i] for i in range(batch_size)], resize, num_workers))
        ph_emb = create_id3_embedding(preprocess(ph_videos, (224, 224)))

        ph_videos2 = tf.convert_to_tensor(load_videos([fake_video_files[i] for i in range(batch_size)], resize, num_workers))
        ph_emb2 = create_id3_embedding(preprocess(ph_videos2, (224, 224)))

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.tables_initializer())

            for i in tqdm(range(total_size // batch_size)):
                start = i * batch_size
                end = min(start + batch_size, total_size)
                real_videos_np = load_videos([real_video_files[i] for i in range(start, end)], resize, num_workers)
                fake_videos_np = load_videos([fake_video_files[i] for i in range(start, end)], resize, num_workers)
                real_emb.append(sess.run(ph_emb, feed_dict={ph_videos: real_videos_np}))
                fake_emb.append(sess.run(ph_emb2, feed_dict={ph_videos2: fake_videos_np}))

            real_emb = np.concatenate(real_emb, axis=0)
            fake_emb = np.concatenate(fake_emb, axis=0)
    return real_emb, fake_emb

def fvd_from_files(real_video_files, fake_video_files, num_workers):
    real_emb, fake_emb = emb_from_files(real_video_files, fake_video_files, num_workers)
    fvd = compute_fvd_given_acts(real_emb, fake_emb)
    print("FVD is: %.2f." % fvd)
    return fvd

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

def fvd_size(real_emb, fake_emb, size):
    fvds = []
    n = real_emb.shape[0] // size
    for i in tqdm(range(n)):
        r = real_emb[i * size:(i + 1) * size]
        f = fake_emb[i * size:(i + 1) * size]
        fvds.append(compute_fvd_given_acts(r, f))
    print("Individual FVD scores")
    print(fvds)
    print(f"Mean/std of FVD across {n} runs of size {size}")
    print(np.mean(fvds), np.std(fvds))

def fvd_full(real_emb, fake_emb):
    fvd = compute_fvd_given_acts(real_emb, fake_emb)
    print(f"FVD score: {fvd}")

def main(args):
    fake_folders = get_folders(args.exp_tag, args.num_folds)
    real_tag = args.exp_tag if args.real_tag is None else args.real_tag
    real_folders = get_folders(real_tag, args.num_folds)

    real_emb, fake_emb = [], []
    for i, (real_root, fake_root) in tqdm(enumerate(zip(sorted(real_folders), sorted(fake_folders)))):
        print(f"[{i}] Loading real")
        real_video_files = get_video_files(os.path.join(real_root, args.real_folder))
        print(f"Found {len(real_video_files)} {args.real_folder} video files")

        print(f"[{i}] Loading fake")
        fake_video_files = get_video_files(os.path.join(fake_root, args.fake_folder))
        print(f"Found {len(fake_video_files)} {args.fake_folder} video files")

        assert len(real_video_files) == len(fake_video_files)

        print(f"[{i}] Computing embeddings")
        real_emb_i, fake_emb_i = emb_from_files(real_video_files, fake_video_files, args.resize, args.num_workers)
        real_emb.append(real_emb_i)
        fake_emb.append(fake_emb_i)

    print(f"Computing FVD with {args.mode} mode")
    real_emb = np.concatenate(real_emb, axis=0)
    fake_emb = np.concatenate(fake_emb, axis=0)
    if args.mode == "size" or args.mode == "both":
        fvd_size(real_emb, fake_emb, args.size)
    if args.mode == "full" or args.mode == "both":
        fvd_full(real_emb, fake_emb)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_tag', type=str, default=None)
    parser.add_argument('--real_tag', type=str, default=None)
    parser.add_argument('--real_folder', type=str, default="real")
    parser.add_argument('--fake_folder', type=str, default="fake")
    parser.add_argument('--num_folds', type=int, default=None)
    parser.add_argument('--mode', type=str, default="size", help="(size | full | both)")
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--resize', type=int, nargs="+", default=None)
    parser.add_argument('--num_workers', type=int, default=8)
    args = parser.parse_args()
    main(args)