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
    """Evaluate on FSC-147.

    predict() returns pred_counts of shape [B, K].  For FSC-147 K=1, so we
    squeeze the class dimension to get a [B] tensor before accumulating.
    """
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
            # pred_counts_normal / _flipped: [B, K=1] → squeeze to [B]
            pred_counts = (pred_counts_normal + pred_counts_flipped) / 2
            pred_counts = pred_counts.squeeze(1)   # [B, 1] → [B]

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
    """Evaluate on LVIS (single class per sample, K=1)."""
    model.cuda()
    model.eval()
    MAE = 0
    MSE = 0
    num_images = len(dataloader.dataset)
    for batch in tqdm(dataloader, desc="Validating"):
        images = batch["image"].cuda()
        texts = [[word] for word in batch["text_label"]]
        counts = batch["count"].cuda()
        with torch.no_grad():
            # pred_counts: [B, 1] → squeeze to [B]
            pred_counts = model.predict(
                images, texts, confidence_threshold=confidence_threshold
            )[0].squeeze(1)

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
    """Evaluate on Objects365 (single class per sample, K=1)."""
    model.cuda()
    model.eval()
    MAE = 0
    MSE = 0
    num_images = len(dataloader.dataset)
    for batch in tqdm(dataloader, desc="Validating"):
        images = batch["image"].cuda()
        texts = [[word] for word in batch["text_label"]]
        counts = batch["count"].cuda()
        with torch.no_grad():
            pred_counts = model.predict(
                images, texts, confidence_threshold=confidence_threshold
            )[0].squeeze(1)
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
    """Evaluate on OpenImagesV7 (single class per sample, K=1)."""
    model.cuda()
    model.eval()
    MAE = 0
    MSE = 0
    num_images = len(dataloader.dataset)
    for batch in tqdm(dataloader, desc="Validating"):
        images = batch["image"].cuda()
        texts = [[word] for word in batch["text_label"]]
        counts = batch["count"].cuda()
        with torch.no_grad():
            pred_counts = model.predict(
                images, texts, confidence_threshold=confidence_threshold
            )[0].squeeze(1)
            gt_counts = torch.round(counts)
            pred_counts = torch.round(pred_counts)
            MAE += (pred_counts - gt_counts).abs().sum().item()
            MSE += ((pred_counts - gt_counts) ** 2).sum().item()
    MAE = MAE / num_images
    RMSE = math.sqrt(MSE / num_images)
    return MAE, RMSE


def evaluate_multiclass(
    model: nn.Module,
    dataloader: DataLoader,
    confidence_threshold: float = 0.0,
) -> dict:
    """Evaluate multiclass counting.

    Expects each batch to contain:
        "image":       [B, 3, H, W]
        "text_labels": List[List[str]]  — B lists of K class names
        "counts":      [B, K]           — ground-truth count per class

    Returns a dict with per-class and overall MAE / RMSE.
    """
    model.cuda()
    model.eval()

    all_pred = []
    all_gt = []

    for batch in tqdm(dataloader, desc="Validating (multiclass)"):
        images = batch["image"].cuda()
        texts = batch["text_labels"]          # List[List[str]], shape [B, K]
        counts = batch["counts"].cuda()       # [B, K]

        with torch.no_grad():
            pred_counts = model.predict(
                images, texts, confidence_threshold=confidence_threshold
            )[0]                              # [B, K]

        all_pred.append(torch.round(pred_counts))
        all_gt.append(torch.round(counts))

    all_pred = torch.cat(all_pred, dim=0)     # [N, K]
    all_gt = torch.cat(all_gt, dim=0)         # [N, K]

    error = all_pred - all_gt                 # [N, K]
    mae_per_class = error.abs().mean(dim=0)   # [K]
    rmse_per_class = (error ** 2).mean(dim=0).sqrt()  # [K]

    return {
        "MAE_per_class": mae_per_class.tolist(),
        "RMSE_per_class": rmse_per_class.tolist(),
        "MAE_overall": mae_per_class.mean().item(),
        "RMSE_overall": rmse_per_class.mean().item(),
    }
