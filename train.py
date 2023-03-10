import argparse
import os

import torch

from dataset import create_dataset
from loss import create_loss_function
from metrics import compute_model_metrics
from models import create_model
from optim import create_optimizer, create_scheduler
from utils.config import read_config_from_file
from utils.plot import display_confusion_matrix
from utils.torch import (
    current_datatime,
    save_model_weights,
    test_model,
    train_single_epoch,
)

os.environ["TORCH_HOME"] = "./.cache"


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_folder", type=str, required=True)
    parser.add_argument("--train_samples_file", type=str, required=True)
    parser.add_argument("--val_samples_file", type=str, required=True)
    parser.add_argument("--test_samples_file", type=str, required=True)

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
    start_date, start_time = current_datatime()

    train_dataset = create_dataset(
        args.train_samples_file,
        args.dataset_folder,
        experiment_config.data_kwargs.batch_size,
        experiment_config.data_kwargs.many_to_one_setting,
        experiment_config.data_kwargs.image_size,
        upsample=experiment_config.data_kwargs.class_balancing,
        split="train",
    )
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

    optimizer = create_optimizer(experiment_config.optimizer_kwargs, model)
    lr_scheduler = create_scheduler(
        experiment_config.optimizer_kwargs, optimizer)
    criterion = create_loss_function(experiment_config)

    metrics_folder = os.path.join(
        args.checkpoint_dir, "metrics", f"{start_date}_{start_time}"
    )
    os.makedirs(metrics_folder, exist_ok=True)

    train_loss_arr, val_loss_arr = list(), list()
    train_acc_arr, val_acc_arr = list(), list()
    for epoch in range(
        experiment_config.experiment_kwargs.start_epoch,
        experiment_config.experiment_kwargs.end_epoch,
    ):
        train_loss, train_acc = train_single_epoch(
            experiment_config,
            train_dataset,
            model,
            optimizer,
            lr_scheduler,
            criterion,
            epoch,
            device,
        )
        print(f"Train loss: {train_loss} | Train acc: {train_acc}")
        torch.cuda.empty_cache()

        # val_loss, val_acc = evaluate_model(
        #    experiment_config, val_dataset, model, criterion,
        #    epoch, device, compute_model_metrics)
        # print(f'Val loss: {val_loss} | Val acc: {val_acc}')
        # torch.cuda.empty_cache()

        confusion_matrix_path = "epoch:{}_val_{}_{}".format(
            epoch,
            experiment_config.model_kwargs.encoder.name,
            experiment_config.model_kwargs.temporal.name,
        )

        test_model(
            experiment_config,
            val_dataset,
            model,
            device,
            os.path.join(metrics_folder, confusion_matrix_path),
            compute_model_metrics,
            display_confusion_matrix,
        )

        print("==" * 20)

        train_loss_arr.append(train_loss)
        # val_loss_arr.append(val_loss)
        train_acc_arr.append(train_acc)
        # val_acc_arr.append(val_acc)

        save_model_weights(
            experiment_config,
            model.state_dict(),
            args.checkpoint_dir,
            epoch,
            start_date,
            start_time,
        )

    with open(
        os.path.join(
            args.checkpoint_dir,
            "model_weights",
            f"{start_date}_{start_time}",
            "config.txt",
        ),
        "w",
    ) as text_file:
        text_file.write(str(experiment_config))
