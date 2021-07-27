import os

from data.base_dataset import BaseDataset
from data.folder_dataset import make_dataset


class DrumDataset(BaseDataset):

    def get_data(self, opt, phase="train", from_vid=False):
        root = opt.dataroot

        phase = 'test' if phase == 'valid' else 'train'
        assert from_vid
        vid_paths = make_dataset(os.path.join(root, "AudioSet_Dataset", phase, "mp4"), recursive=True, from_vid=True)
        stft_paths = [p.replace("/mp4/", "/stft_pickle/").replace(".mp4", ".pickle") for p in vid_paths]
        vid_id = [int(os.path.basename(p).split(".")[0]) for p in vid_paths]
        return {"vid_paths": vid_paths, "stft_paths": stft_paths, "vid_id": vid_id}
