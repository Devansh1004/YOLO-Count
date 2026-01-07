from tqdm import tqdm
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from yolo_count.utils.fn import wrap_hw


def evaluate_on_fsc(
    model: nn.Module,
    dataloader: DataLoader,
    large_threshold: int = float("inf"),
    confidence_threshold: float = 0.0,
    save_file=None,
) -> None:
    model.cuda()
    model.eval()
    MAE = 0
    MSE = 0
    BIAS = 0
    f = open(save_file, "w") if save_file is not None else None
    for batch in tqdm(dataloader, desc="Validating"):
        images = batch["image"].cuda()
        texts = [[word] for word in batch["text_label"]]
        counts = batch["count"].cuda()
        original_hw = wrap_hw(batch["original_hw"])
        with torch.no_grad():
            pred_counts_normal = model.predict(
                images,
                texts,
                original_hw=original_hw,
                large_threshold=large_threshold,
                confidence_threshold=confidence_threshold,
            )[0]
            flipped_images = torch.flip(images, dims=[3])
            pred_counts_flipped = model.predict(
                flipped_images,
                texts,
                original_hw=original_hw,
                large_threshold=large_threshold,
                confidence_threshold=confidence_threshold,
            )[0]
            pred_counts = (pred_counts_normal + pred_counts_flipped) / 2
            gt_counts = torch.round(counts)
            pred_counts = torch.round(pred_counts)
            MAE += (pred_counts - gt_counts).abs().sum().item()
            MSE += ((pred_counts - gt_counts) ** 2).sum().item()
            BIAS += (pred_counts - gt_counts).sum().item()
    if f is not None:
        f.close()
    MAE /= len(dataloader.dataset)
    RMSE = math.sqrt(MSE / len(dataloader.dataset))
    BIAS /= len(dataloader.dataset)
    return MAE, RMSE, BIAS


def evaluate_on_lvis(
    model: nn.Module, dataloader: DataLoader, confidence_threshold: float = 0.0
) -> None:
    model.cuda()
    model.eval()
    MAE = 0
    MSE = 0
    i = 0
    num_images = len(dataloader.dataset)
    for batch in tqdm(dataloader, desc="Validating"):
        images = batch["image"].cuda()
        texts = [[word] for word in batch["text_label"]]
        counts = batch["count"].cuda()
        i += 1
        with torch.no_grad():
            pred_counts = model.predict(
                images, texts, confidence_threshold=confidence_threshold
            )[0]

            gt_counts = torch.round(counts)
            pred_counts = torch.round(pred_counts)
            MAE += (pred_counts - gt_counts).abs().sum().item()
            MSE += ((pred_counts - gt_counts) ** 2).sum().item()
    MAE = MAE / num_images
    RMSE = math.sqrt(MSE / num_images)
    return MAE, RMSE


def evaluate_on_obj365(
    model: nn.Module, dataloader: DataLoader, confidence_threshold: float = 0.0
) -> None:
    model.cuda()
    model.eval()
    MAE = 0
    MSE = 0
    i = 0
    num_images = len(dataloader.dataset)
    for batch in tqdm(dataloader, desc="Validating"):
        images = batch["image"].cuda()
        texts = [[word] for word in batch["text_label"]]
        counts = batch["count"].cuda()
        i += 1
        with torch.no_grad():
            pred_counts = model.predict(
                images, texts, confidence_threshold=confidence_threshold
            )[0]
            gt_counts = torch.round(counts)
            pred_counts = torch.round(pred_counts)
            MAE += (pred_counts - gt_counts).abs().sum().item()
            MSE += ((pred_counts - gt_counts) ** 2).sum().item()
    MAE = MAE / num_images
    RMSE = math.sqrt(MSE / num_images)
    return MAE, RMSE


def evaluate_on_oimgv7(
    model: nn.Module, dataloader: DataLoader, confidence_threshold: float = 0.0
) -> None:
    model.cuda()
    model.eval()
    MAE = 0
    MSE = 0
    i = 0
    num_images = len(dataloader.dataset)
    for batch in tqdm(dataloader, desc="Validating"):
        images = batch["image"].cuda()
        texts = [[word] for word in batch["text_label"]]
        counts = batch["count"].cuda()
        i += 1
        with torch.no_grad():
            pred_counts = model.predict(
                images, texts, confidence_threshold=confidence_threshold
            )[0]
            gt_counts = torch.round(counts)
            pred_counts = torch.round(pred_counts)
            MAE += (pred_counts - gt_counts).abs().sum().item()
            MSE += ((pred_counts - gt_counts) ** 2).sum().item()
    MAE = MAE / num_images
    RMSE = math.sqrt(MSE / num_images)
    return MAE, RMSE
