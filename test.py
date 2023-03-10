import argparse
import os

import torch

from dataset import create_dataset
from metrics import compute_model_metrics
from models import create_model
from utils.config import read_config_from_file
from utils.plot import display_confusion_matrix
from utils.torch import test_model

os.environ["TORCH_HOME"] = "./.cache"


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_folder", type=str, required=True)
    parser.add_argument("--val_samples_file", type=str, required=True)
    parser.add_argument("--test_samples_file", type=str, required=True)

    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--experiment_cfg", type=str, required=True)
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        help="path to save model checkpoints",
        default="checkpoints/",
    )

    args = parser.parse_args()
    experiment_config = read_config_from_file(args.experiment_cfg)
    return args, experiment_config


if __name__ == "__main__":
    args, experiment_config = parse_arguments()

    val_dataset = create_dataset(
        args.val_samples_file,
        args.dataset_folder,
        experiment_config.data_kwargs.batch_size,
        experiment_config.data_kwargs.many_to_one_setting,
        experiment_config.data_kwargs.image_size,
        upsample=False,
        split="val",
    )
    test_dataset = create_dataset(
        args.test_samples_file,
        args.dataset_folder,
        experiment_config.data_kwargs.batch_size,
        experiment_config.data_kwargs.many_to_one_setting,
        experiment_config.data_kwargs.image_size,
        upsample=False,
        split="test",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = create_model(experiment_config).to(device)
    model.load_state_dict(torch.load(args.checkpoint_path))

    test_model(
        experiment_config,
        val_dataset,
        model,
        device,
        f"validation_{experiment_config.model_kwargs.encoder.name}_{experiment_config.model_kwargs.temporal.name}",
        compute_model_metrics,
        display_confusion_matrix,
    )

    test_model(
        experiment_config,
        test_dataset,
        model,
        device,
        f"test_{experiment_config.model_kwargs.encoder.name}_{experiment_config.model_kwargs.temporal.name}",
        compute_model_metrics,
        display_confusion_matrix,
    )
