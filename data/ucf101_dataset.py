import os

from data.base_dataset import BaseDataset
from data.folder_dataset import make_dataset


class Ucf101Dataset(BaseDataset):

    def get_data(self, opt, phase="train", from_vid=False):
        root = opt.dataroot
        assert from_vid
        vid_paths = make_dataset(os.path.join(root, "videos"), recursive=True, from_vid=True)
        return {"vid_paths": vid_paths}
