import argparse
import cv2
import random
from glob import glob
import time

import os, sys, inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
code_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, code_dir)

from tools.utils import mkdir
from data.folder_dataset import make_dataset

def main(args):
    frame_paths = make_dataset(args.data_root, recursive=True, from_vid=False)
    random.shuffle(frame_paths)
    mkdir(args.out_dir)
    i = len(glob(os.path.join(args.out_dir, "*.png")))
    global frame
    for path in frame_paths:
        try:
            original_frame = cv2.imread(path)
            frame = original_frame.copy()
            success, (x, y) = annotate()
            if success:
                out_path = os.path.join(args.out_dir, f"{i:05d}_{x}_{y}.jpg")
                cv2.imwrite(out_path, original_frame)
                i += 1
        except:
            print(f"Skipping {path}")
            time.sleep(3)

def annotate():
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', get_x_y)
    while True:
        cv2.imshow('image', frame)
        k = cv2.waitKey(20) & 0xFF
        if k == 27:
            break
        elif k == ord('a'):
            return True, (mouseX, mouseY)
    return False, (None, None)

def get_x_y(event, x, y, flags, param):
    global mouseX, mouseY
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(frame, (x, y), 10, (255, 0, 0), -1)
        mouseX, mouseY = x, y

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--out_dir', type=str, required=True)
    args = parser.parse_args()
    main(args)