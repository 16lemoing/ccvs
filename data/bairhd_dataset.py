import os

from data.base_dataset import BaseDataset
from data.folder_dataset import make_dataset


class BairhdDataset(BaseDataset):

    def get_data(self, opt, phase="train", from_vid=False):
        phase = "test" if phase == "valid" else phase
        root = opt.dataroot
        assert not from_vid

        if self.opt.load_state:
            frame_paths = make_dataset(os.path.join(root, "annotated_frames"), recursive=True, from_vid=False)
            if phase == 'train':
                frame_paths = [p for p in frame_paths if self.get_id(p) % 5 != 0]
            else:
                frame_paths = [p for p in frame_paths if self.get_id(p) % 5 == 0]
            frame_states = [self.get_state(p) for p in frame_paths]
            return {"frame_paths": frame_paths, "frame_states": frame_states}
        else:
            frame_paths = make_dataset(os.path.join(root, "original_frames_256", phase), recursive=True, from_vid=False)
            frame_dic = {}
            for path in sorted(frame_paths):
                seq = os.path.dirname(path)
                if seq in frame_dic:
                    frame_dic[seq].append(path)
                else:
                    frame_dic[seq] = [path]
            vid_frame_paths = list(frame_dic.values())
            return {"frame_paths": frame_paths, "vid_frame_paths": vid_frame_paths}

    def get_id(self, path):
        return int(os.path.basename(path).split("_")[0])

    def get_state(self, path):
        x, y = os.path.basename(path).split(".")[0].split("_")[1:3]
        x, y = int(x) / 256, int(y) / 256
        return [x, y]
