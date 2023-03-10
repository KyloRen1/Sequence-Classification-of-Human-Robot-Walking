import os
from collections import Counter

import numpy as np
import torch
from torch.utils import data
from tqdm import trange

from .data import StairnetVideoDataset


def create_sampler(dataset: torch.utils.data, many_to_one: bool, upsample: bool = False) -> torch.utils.data.sampler.Sampler:
    """ DataLoader data sampler

    Args:
        dataset (torch.utils.data): dataset class 
        many_to_one (bool): switcher for using seq-2-seq labels or seq-to-one lanels
        upsample (bool, optional): weighted data sampling to balance classes. Defaults to True. 

    Returns:
        torch.utils.data.sampler.Sampler: data sampler
    """    
    if not upsample:
        return None
    else:
        os.makedirs('.cache', exist_ok=True)

        if os.path.exists('.cache/labels_counter.txt'):
            labels = np.loadtxt('.cache/labels_counter.txt', dtype=int)
        else:
            print('No file labels_counter.txt found')
            if many_to_one:
                labels = [dataset[i][1].item() for i in trange(len(dataset), desc='many to one labels')]
            else:
                labels = [dataset[i][1][-1].item() for i in trange(len(dataset), desc='many to many labels')]
            np.savetxt('.cache/labels_counter.txt', np.array(labels), fmt='%d')
        
        if os.path.exists('.cache/class_counts.txt'):
            class_counts = np.loadtxt('.cache/class_counts.txt', dtype=int)
        else:
            print('No file class_counts.txt found')
            counts = Counter(labels)
            class_counts = np.array([counts[0], counts[1], counts[2], counts[3]])
            np.savetxt('.cache/class_counts.txt', class_counts, fmt='%d')
        
        weights = 1. / class_counts
        samples_weight = np.array([weights[t] for t in labels])
        samples_weight = torch.from_numpy(samples_weight)
        sampler = data.WeightedRandomSampler(samples_weight, len(samples_weight))
    return sampler


def create_dataset(samples_file: str, dataset_path: str, batch_size: int, many_to_one: bool, image_size: int, 
                   upsample:int = True, split: str = 'train') -> torch.utils.data.DataLoader:
    """ Dataset builder method

    Args:
        samples_file (str): split samples file
        dataset_path (str): path to the dataset frames
        batch_size (int): number of samples in each batch
        many_to_one (bool): seq-2-seq or seq-to-one labels
        image_size (int): image size
        upsample (int, optional): weighted data sampling to balance classes. Defaults to True.
        split (str, optional): dataset split. Defaults to 'train'.

    Returns:
        torch.utils.data.DataLoader
    """    
    
    dataset = StairnetVideoDataset(samples_file, dataset_path, many_to_one, image_size)

    sampler = create_sampler(dataset, many_to_one, upsample)

    dataloader = data.DataLoader(
        dataset, 
        batch_size, 
        shuffle=True if (split == 'train' and not upsample) else False,
        num_workers=4, #8
        sampler = sampler
    )
    return dataloader