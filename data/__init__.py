import importlib
import torch.utils.data
from torch.utils.data._utils.collate import default_collate
import numpy as np

from data.base_dataset import BaseDataset
from tools.utils import get_vprint


def find_dataset_using_name(dataset_name):
    # Given the option --dataset [datasetname],
    # the file "datasets/datasetname_dataset.py"
    # will be imported. 
    dataset_filename = "data." + dataset_name + "_dataset"
    datasetlib = importlib.import_module(dataset_filename)

    # In the file, the class called DatasetNameDataset() will
    # be instantiated. It has to be a subclass of BaseDataset,
    # and it is case-insensitive.
    dataset = None
    target_dataset_name = dataset_name.replace('_', '') + 'dataset'
    for name, cls in datasetlib.__dict__.items():
        if name.lower() == target_dataset_name.lower() \
           and issubclass(cls, BaseDataset):
            dataset = cls
            
    if dataset is None:
        raise ValueError(f"In {dataset_filename}.py, there should be a subclass of BaseDataset "
                         f"with class name that matches {target_dataset_name} in lowercase.")

    return dataset


def create_dataset(opt, phase='train', fold=None, from_vid=False, load_vid=False, verbose=True):
    dataset = find_dataset_using_name(opt.dataset)
    instance = dataset(opt, phase=phase, fold=fold, from_vid=from_vid, load_vid=load_vid, verbose=verbose)
    file_type = "video" if from_vid else "img"
    load_type = "video" if load_vid else "img"
    vprint = get_vprint(verbose)
    fold_str = f"-{fold}" if fold is not None else ""
    vprint(f"Creation of dataset [{type(instance).__name__}-{phase}{fold_str}] of size {len(instance)} "
           f"loading {load_type} data from {file_type} data")
    return instance


def create_dataloader(dataset, batch_size, num_workers, is_train):
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=num_workers,
        drop_last=is_train,
        worker_init_fn=lambda _: np.random.seed(),
        collate_fn=custom_collate_fn
    )
    return dataloader


def custom_collate_fn(batch):
    input_dict = {}
    elem = batch[0]
    for key in elem:
        if key in ["img", "mask_img", "flow_img", "layout"] and elem[key].ndim == 4:
            input_dict[key] = torch.cat([d[key] for d in batch], dim=0)
        else:
            input_dict[key] = default_collate([d[key] for d in batch])
    return input_dict