import random

import numpy as np
from collections import OrderedDict
import os
import glob
import cv2
import torch.utils.data as data
import re

rng = np.random.RandomState(2020)


def np_load_frame(filename, resize_height, resize_width):
    """
    Load image path and convert it to numpy.ndarray. Notes that the color channels are BGR and the color space
    is normalized from [0, 255] to [-1, 1].

    :param filename: the full path of image
    :param resize_height: resized height
    :param resize_width: resized width
    :return: numpy.ndarray
    """
    image_decoded = cv2.imread(filename)
    image_resized = cv2.resize(image_decoded, (resize_width, resize_height))
    image_resized = image_resized.astype(dtype=np.float32)
    image_resized = (image_resized / 127.5) - 1.0
    return image_resized



class DataLoader(data.Dataset):
    def __init__(self, video_folder, transform, resize_height, resize_width, train=True, time_step=4, num_pred=1):
        self.dir = video_folder
        self.transform = transform
        self.videos = OrderedDict()
        self._resize_height = resize_height
        self._resize_width = resize_width
        self._time_step = time_step
        self._num_pred = num_pred
        self.train = train
        self.setup()
        self.samples = self.get_all_samples()
        self.sample = []

    def setup(self):
        npy_files = glob.glob(os.path.join(self.dir, "*.npy"))
        npy_files.sort()

        self.videos = OrderedDict()

        for idx, file_path in enumerate(npy_files):
            file_name = os.path.basename(file_path)

            self.videos[file_name] = {}
            self.videos[file_name]['path'] = file_path
            self.videos[file_name]['idx'] = idx
            self.videos[file_name]['length'] = 1  # one sample per file



    def get_all_samples(self):
        npy_files = glob.glob(os.path.join(self.dir, "*.npy"))
        npy_files.sort()
        return npy_files

    def __getitem__(self, index):
        # a = self.samples[index].split('/')[-2]
        # dataset_name = self.samples[index].split('/')[2]
        video_name = self.samples[index].split('/')[-1].split('\\')[-2]
        # a = self.samples[index].split('.')[-2].split('\\')[-1]
        frame_name = int(self.samples[index].split('.')[-2].split('\\')[-1])
        # b = self.videos[video_name]['idx']  # int(re.findall(r'\d+', video_name)[0])-1
        # bkg = self.bkg if dataset_name != 'shanghaitech' else self.bkg[b]
        batch_forward = []
        batch_backward = []
        # print('----------------------------------------')
        # print(video_name)
        # print(frame_name)
        #
        # print('----------------------------------------')

        foreground = []
        noisy = []
        if self.train:
            for i in range(0, self._time_step, 2):
                # print(frame_name + i - 1, flush=True)

                image_f = np_load_frame(self.videos[video_name]['frame'][frame_name + i - 1], self._resize_height,
                                          self._resize_width)
                if self.transform is not None:
                    batch_forward.append(self.transform(image_f))
            for j in range(self._time_step-1, -1, -2):
                # print(frame_name + j - 1, flush=True)

                image_b = np_load_frame(self.videos[video_name]['frame'][frame_name + j - 1], self._resize_height,
                                      self._resize_width)

                if self.transform is not None:
                    batch_backward.append(self.transform(image_b))
            return np.concatenate(batch_forward, axis=0), np.concatenate(batch_backward, axis=0)
        else:
            for i in range(self._time_step//2):
                # print(frame_name + i - 1, flush=True)

                image_f = np_load_frame(self.videos[video_name]['frame'][frame_name + i - 1],
                                          self._resize_height, self._resize_width)
                if self.transform is not None:
                    batch_forward.append(self.transform(image_f))
            for j in range(self._time_step-1, 3, -1):
                # print(frame_name + j - 1, flush=True)

                image_b = np_load_frame(self.videos[video_name]['frame'][frame_name + j - 1],
                                          self._resize_height, self._resize_width)
                if self.transform is not None:
                    batch_backward.append(self.transform(image_b))
            gt_pred = np_load_frame(self.videos[video_name]['frame'][frame_name + 3 - 1],
                                          self._resize_height, self._resize_width)
            return np.concatenate(batch_forward, axis=0), np.concatenate(batch_backward, axis=0), self.transform(gt_pred)

    def __len__(self):
        return len(self.samples)
