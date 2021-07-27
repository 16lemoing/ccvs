import os
import PIL
import numpy as np
from matplotlib.colors import ListedColormap
from scipy import linalg
from collections import OrderedDict
import gzip, pickle, pickletools

import torch
from torchvision import transforms

def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def serialize(path, obj):
    with gzip.open(path, "wb") as f:
        pickled = pickle.dumps(obj)
        optimized_pickle = pickletools.optimize(pickled)
        f.write(optimized_pickle)

def deserialize(path):
    with gzip.open(path, 'rb') as f:
        p = pickle.Unpickler(f)
        return p.load()

def get_vprint(verbose):
    if verbose:
        return lambda s: print(s)
    else:
        return lambda s: None

def to_cuda(tensor_dic, key, flatten_empty=True):
    if key in tensor_dic:
        if isinstance(tensor_dic[key], list):
            return [t.cuda() for t in tensor_dic[key]]
        else:
            if 0 in tensor_dic[key].size() and flatten_empty:
                tensor_dic[key] = torch.Tensor([])
            return tensor_dic[key].cuda()
    return torch.Tensor([])

def flatten_vid(x, vid_ndim=5):
    vid_size = None
    if x.ndim == vid_ndim:
        vid_size = x.shape[:2]
        x = x.view(-1, *x.shape[2:])
    return x, vid_size

def unflatten_vid(x, vid_size):
    if vid_size is not None and x.size(0) != 0:
        b, t = vid_size
        return x.view(b, t, *x.shape[1:])
    else:
        return x

# Taken from https://github.com/mseitzer/pytorch-fid/
def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        # msg = ('fid calculation produces singular product; '
        #        'adding %s to diagonal of cov estimates') % eps
        # print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)

def load_model(model, path, remove_string=None):
    state_dict = torch.load(path)['state_dict']
    if remove_string is not None:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_state_dict[k.replace(remove_string, "")] = v
    else:
        new_state_dict = state_dict
    model.load_state_dict(new_state_dict)

class DummyOpt:
    def __init__(self):
        pass

    def zero_grad(self):
        pass

    def step(self):
        pass

def color_transfer(im, colormap):
    im = im.cpu().numpy()
    im_new = torch.Tensor(im.shape[0], 3, im.shape[2], im.shape[3])
    newcmp = ListedColormap(colormap)
    for i in range(im.shape[0]):
        img = (im[i, 0, :, :]).astype('uint8')
        rgba_img = newcmp(img)
        rgb_img = PIL.Image.fromarray((255 * np.delete(rgba_img, 3, 2)).astype('uint8'))
        tt = transforms.Compose([transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        rgb_img = tt(rgb_img)
        im_new[i, :, :, :] = rgb_img
    return im_new