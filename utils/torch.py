import gc
import math
import os
from datetime import datetime

import numpy as np
import torch
from tqdm import tqdm


def calculate_acc(gt, pred):
    accuracy = (gt == pred).sum() / len(gt)
    return accuracy


def train_single_epoch(
    config, dataloader, model, optimizer, scheduler, criterion, epoch, device
):
    model.train()
    num_updates = epoch * len(dataloader)

    train_loss = list()
    gt_arr, pred_arr = np.array([]), np.array([])
    with tqdm(dataloader, unit="batch") as tqdm_epoch:
        for idx, (samples, labels) in enumerate(tqdm_epoch):
            tqdm_epoch.set_description(f"Epoch {epoch} train step")

            im = samples.to(device)
            gt_labels = labels.to(device)

            if config.model_kwargs.encoder.name == "movinet":
                pred_labels = model(im.permute(0, 2, 1, 3, 4))
            else:
                pred_labels = model(im)

            if not config.data_kwargs.many_to_one_setting:
                BS, S, C = pred_labels.shape
                pred_labels = pred_labels.reshape(BS, C, S)

            loss = criterion(
                input=pred_labels,  # (C), (N, C), (N, C, d1, ...)
                target=gt_labels,  # (), (N), (N, d1, ...)
            )

            if not math.isfinite(loss.item()):
                print("Loss is {}, stopping training".format(
                    loss.item()), force=True)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            num_updates += 1
            scheduler.step()

            tqdm_epoch.set_postfix(
                lr=round(optimizer.param_groups[0]["lr"], 10), loss=loss.item()
            )

            if config.data_kwargs.many_to_one_setting:
                pred_flatten = torch.argmax(pred_labels, dim=1)
            else:
                pred_flatten = torch.argmax(pred_labels, dim=1)

            gt_arr = np.append(gt_arr, gt_labels.detach().cpu().numpy())
            pred_arr = np.append(pred_arr, pred_flatten.detach().cpu().numpy())
            train_loss.append(loss.item())

    epoch_acc = calculate_acc(gt_arr, pred_arr)

    epoch_loss = np.mean(train_loss)
    del gt_arr, pred_arr, train_loss
    gc.collect()
    return epoch_loss, epoch_acc


def evaluate_model(config, dataloader, model, criterion, epoch, device, metrics):
    model.eval()

    val_loss = list()
    gt_arr, pred_arr = np.array([]), np.array([])
    with tqdm(dataloader, unit="batch") as tqdm_epoch:
        for idx, (samples, labels) in enumerate(tqdm_epoch):
            tqdm_epoch.set_description(f"Epoch {epoch} eval step")

            im = samples.to(device)
            gt_labels = labels.to(device)

            if config.model_kwargs.encoder.name == "movinet":
                pred_labels = model(im.permute(0, 2, 1, 3, 4))
            else:
                pred_labels = model(im)

            if not config.data_kwargs.many_to_one_setting:
                BS, S, C = pred_labels.shape
                pred_labels = pred_labels.reshape(BS, C, S)

            loss = criterion(
                input=pred_labels,  # (C), (N, C), (N, C, d1, ...)
                target=gt_labels,  # (), (N), (N, d1, ...)
            )

            tqdm_epoch.set_postfix(loss=loss.item())

            if config.data_kwargs.many_to_one_setting:
                pred_flatten = torch.argmax(pred_labels, dim=1)
            else:
                pred_flatten = torch.argmax(pred_labels, dim=1)

            gt_arr = np.append(gt_arr, gt_labels.detach().cpu().numpy())
            pred_arr = np.append(pred_arr, pred_flatten.detach().cpu().numpy())
            val_loss.append(loss.item())

    epoch_acc = metrics(
        gt_arr.flatten(), pred_arr.flatten(), dataloader.dataset.labels_mapping
    )[0]

    epoch_loss = np.mean(val_loss)
    del gt_arr, pred_arr, val_loss
    gc.collect()
    return epoch_loss, epoch_acc


def test_model(config, dataloader, model, device, split, metrics, plot_cm):
    model.eval()

    if config.data_kwargs.many_to_one_setting:
        gt_arr, pred_arr = np.array([]), np.array([])
    else:
        gt_arr, pred_arr = np.empty((0, config.data_kwargs.seq_len), int), np.empty(
            (0, config.data_kwargs.seq_len), int
        )

    with tqdm(dataloader, unit="batch") as tqdm_epoch:
        for idx, (samples, labels) in enumerate(tqdm_epoch):

            im = samples.to(device)
            gt_labels = labels.to(device)

            if config.model_kwargs.encoder.name == "movinet":
                pred_labels = model(im.permute(0, 2, 1, 3, 4))
            else:
                pred_labels = model(im)

            if config.data_kwargs.many_to_one_setting:
                pred_flatten = torch.argmax(pred_labels, dim=1)
                gt_arr = np.append(gt_arr, gt_labels.detach().cpu().numpy())
                pred_arr = np.append(
                    pred_arr, pred_flatten.detach().cpu().numpy())
            else:
                pred_flatten = torch.argmax(pred_labels, dim=2)
                gt_arr = np.append(
                    gt_arr, gt_labels.detach().cpu().numpy(), axis=0)
                pred_arr = np.append(
                    pred_arr, pred_flatten.detach().cpu().numpy(), axis=0
                )

    if config.data_kwargs.many_to_one_setting:
        accuracy, f1_score, precision, recall, cmat = metrics(
            gt_arr, pred_arr, dataloader.dataset.labels_mapping
        )
        metrics_results = (
            "Accuracy: {} | F1-score: {} | Precision: {} | Recall {}".format(
                accuracy, f1_score, precision, recall
            )
        )

        print()
        print(f"========== {split} ==========")
        print("Many to one classification")
        print(metrics_results)
        plot_cm(
            cmat,
            accuracy,
            f1_score,
            precision,
            recall,
            f"{split}_confusion_matrix_{metrics_results}.png",
            dataloader.dataset.labels_mapping,
        )
    else:
        print()
        print(f"========== {split} ==========")
        print("Many to many classification")
        print("Many to one labels")
        accuracy, f1_score, precision, recall, cmat = metrics(
            gt_arr[:, -1], pred_arr[:, -1], dataloader.dataset.labels_mapping
        )
        metrics_results = (
            "Accuracy: {} | F1-score: {} | Precision: {} | Recall {}".format(
                accuracy, f1_score, precision, recall
            )
        )
        print(metrics_results)
        plot_cm(
            cmat,
            accuracy,
            f1_score,
            precision,
            recall,
            f"{split}_confusion_matrix_M2O_{metrics_results}.png",
            dataloader.dataset.labels_mapping,
        )

        print()
        print("Many to many labels")
        accuracy, f1_score, precision, recall, cmat = metrics(
            gt_arr.flatten(), pred_arr.flatten(), dataloader.dataset.labels_mapping
        )
        metrics_results = (
            "Accuracy: {} | F1-score: {} | Precision: {} | Recall {}".format(
                accuracy, f1_score, precision, recall
            )
        )
        print(metrics_results)
        plot_cm(
            cmat,
            accuracy,
            f1_score,
            precision,
            recall,
            f"{split}_confusion_matrix_M2M_{metrics_results}.png",
            dataloader.dataset.labels_mapping,
        )


def current_datatime():
    now = datetime.now()
    return now.strftime("%b-%d-%Y"), now.strftime("%H:%M:%S")


def save_model_weights(
    config, model_state_dict, checkpoints_folder, epoch, start_date, start_time
):
    curr_date, curr_time = current_datatime()

    model_name = "epoch:{}_{}_{}_{}_{}.pth".format(
        epoch,
        config.model_kwargs.encoder.name,
        config.model_kwargs.temporal.name,
        curr_date,
        curr_time,
    )
    folder_path = os.path.join(
        checkpoints_folder, "model_weights", f"{start_date}_{start_time}"
    )
    os.makedirs(folder_path, exist_ok=True)
    path = os.path.join(folder_path, model_name)
    torch.save(model_state_dict, path)
    print(f"Saved model weights in {path}")
