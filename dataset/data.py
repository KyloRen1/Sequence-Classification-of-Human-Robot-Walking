import ast
import os
from typing import List, Tuple

import numpy as np
import torch
import torchvision
from PIL import Image
from torch.utils.data import Dataset


class StairnetVideoDataset(Dataset):
    def __init__(self, split_file:str, dataset_path:str, many_to_one:bool, image_size:int):
        """ init

        Args:
            split_file (str): validation split file with samples
            dataset_path (str): path to the frames directory
            many_to_one (bool): switch for seq-to-seq or seq-to-one labels
            image_size (int): size of frames
        """

        self.split_file = split_file
        self.split_samples = self._read_sample_file(self.split_file)
        self.label_setting_many_to_one = many_to_one
        self.dataset_path = dataset_path
        self.image_size = image_size
        self.labels_mapping = ['IS', 'ISLG', 'LG', 'LGIS']
        self.normalize_video = torchvision.transforms.Normalize(
            mean=(127.,),
            std=(32.,)
        )

    def _read_sample_file(self, path:str) -> List[str]:
        samples = open(path).readlines()
        return samples

    def _preprocess_labels(self, labels: List) -> np.ndarray:
        """ mapping string labels to index

        Args:
            labels (List): array of string labels

        Returns:
            np.ndarray: array of label indices
        """
        return np.array([self.labels_mapping.index(el) for el in labels])

    def _read_video_sample(self, sample: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """ reading video clip from frames path

        Args:
            sample (str): string representation of sample dict, with frames path and labels array

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: torch tensors of frames and labels
        """
        sample = ast.literal_eval(sample)
        frames = np.empty((len(sample['sequence']), self.image_size, self.image_size, 3))
        labels = sample['labels']
        for i, img_path in enumerate(sample['sequence']):
            img_class = labels[i]
            img_new_path = os.path.join(
                self.dataset_path, img_class, 'preprocessed ' + img_path.split('/')[-1])
            frames[i, :, :] = np.array(Image.open(img_new_path))

        labels = self._preprocess_labels(labels)
        frames = torch.from_numpy(frames)
        labels = torch.tensor(labels)

        if self.normalize_video is not None:
            frames = self.normalize_video(frames)
        return frames.to(torch.float32), labels.to(torch.long)


    def __getitem__(self, idx:int):
        sample = self.split_samples[idx]
        sample, label = self._read_video_sample(sample)
        sample = sample.permute(0, 3, 1, 2)
        if self.label_setting_many_to_one:
            return sample, label[-1]
        return sample, label

    def __len__(self):
        return len(self.split_samples)
