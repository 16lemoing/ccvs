import os
import pandas as pd
from glob import glob

from data.base_dataset import BaseDataset


class Kinetics600Dataset(BaseDataset):

    def get_data(self, opt, phase="train", from_vid=False):
        raise NotImplementedError("should go through manual preprocessing")
        phase = "val" if phase == "valid" else phase
        assert from_vid
        root = opt.dataroot
        csv_file = os.path.join(root, f"data/kinetics_600_{phase}.csv")
        csv = pd.read_csv(csv_file)
        data_folder = os.path.join(root, f"data/kinetics_700_{phase}")
        found = 0
        tot = 0
        vid_desc = []
        vid_paths = []
        for idx, row in csv.iterrows():
            tot += 1
            if phase == "test":
                paths = glob(os.path.join(data_folder, f"test/{row['youtube_id']}*"))
            else:
                paths = glob(os.path.join(data_folder, f"{row['label']}/{row['youtube_id']}*"))
            if len(paths) == 1:
                found += 1
                print(f"{found}/{tot}")
                vid_paths.append(paths[0])
                vid_desc.append(row['label'])
        print(f"Found {found} videos out of the {tot} videos of Kinetics600")
        vid_labels = [opt.categories.index(s) for s in vid_desc]
        return {"vid_paths": vid_paths, "vid_labels": vid_labels}