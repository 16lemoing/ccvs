# Adapted from  https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py

import torch.utils.data as data
from PIL import Image
import os

IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tiff', '.webp']
VID_EXTENSIONS = ['.avi', '.mp4']

def is_file(filename, from_vid=False):
    return is_vid_file(filename) if from_vid else is_img_file(filename)

def is_img_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def is_vid_file(filename):
    return any(filename.endswith(extension) for extension in VID_EXTENSIONS)

def make_dataset_rec(dir, files, from_vid):
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, dnames, fnames in sorted(os.walk(dir, followlinks=True)):
        for fname in fnames:
            if is_file(fname, from_vid):
                path = os.path.join(root, fname)
                files.append(path)


def make_dataset(dir, recursive=False, read_cache=False, write_cache=False, from_vid=False):
    files = []

    if read_cache:
        possible_filelist = os.path.join(dir, 'files.list')
        if os.path.isfile(possible_filelist):
            with open(possible_filelist, 'r') as f:
                files = f.read().splitlines()
                return files

    if recursive:
        make_dataset_rec(dir, files, from_vid)
    else:
        assert os.path.isdir(dir) or os.path.islink(dir), '%s is not a valid directory' % dir

        for root, dnames, fnames in sorted(os.walk(dir)):
            for fname in fnames:
                if is_file(fname, from_vid):
                    path = os.path.join(root, fname)
                    files.append(path)

    if write_cache:
        filelist_cache = os.path.join(dir, 'files.list')
        with open(filelist_cache, 'w') as f:
            for path in files:
                f.write("%s\n" % path)
            print('wrote filelist cache at %s' % filelist_cache)

    return files


def default_loader(path):
    return Image.open(path).convert('RGB')


class ImageFolder(data.Dataset):

    def __init__(self, root, transform=None, return_paths=False,
                 loader=default_loader):
        imgs = make_dataset(root)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in: " + root + "\n"
                               "Supported image extensions are: " +
                               ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.return_paths = return_paths
        self.loader = loader

    def __getitem__(self, index):
        path = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.return_paths:
            return img, path
        else:
            return img

    def __len__(self):
        return len(self.imgs)