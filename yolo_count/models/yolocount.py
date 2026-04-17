from typing import List, Tuple, Optional, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import matplotlib.pyplot as plt
from PIL import Image
from yolo_count.models.backbone import (
    MultiModalYOLOBackbone,
    HuggingCLIPLanguageBackbone,
    YOLOv8CSPDarknet,
)
from yolo_count.models.neck import YOLOCountPAFPN
from yolo_count.models.head import ProportionCountingHead
from yolo_count.utils.fn import aspect_based_center_mask


def flatten_texts(texts):
    return [t for group in texts for t in group]


def build_yolocount_model_base():
    head_cfg = {
        "in_channels": [256, 512, 512],
        "embed_dims": 512,
        "freeze_all": False,
    }
    neck_cfg = {
        "in_channels": [256, 512, 512],
        "out_channels": [256, 512, 512],
        "guide_channels": 512,
        "embed_channels": [128, 256, 256],
        "num_heads": [4, 8, 8],
    }
    backbone_cfg = {
        "image_model": {
            "arch": "P5",
            "last_stage_out_channels": 512,
        },
        "text_model": {
            "model_name": "openai/clip-vit-base-patch32",
            "frozen_modules": ["all"],
        },
    }
    return YOLOCount(backbone_cfg, neck_cfg, head_cfg)


def build_yolocount_model_large():
    widen_factor = 1.0
    head_cfg = {
        "in_channels": [256, 512, 512],
        "embed_dims": 512,
        "freeze_all": False,
        "widen_factor": widen_factor,
    }
    neck_cfg = {
        "in_channels": [256, 512, 512],
        "out_channels": [256, 512, 512],
        "guide_channels": 512,
        "embed_channels": [128, 256, 256],
        "num_heads": [4, 8, 8],
        "deepen_factor": 3.0,
        "widen_factor": widen_factor,
    }
    backbone_cfg = {
        "image_model": {
            "arch": "P5",
            "last_stage_out_channels": 512,
            "deepen_factor": 1.0,
            "widen_factor": widen_factor,
        },
        "text_model": {
            "model_name": "openai/clip-vit-base-patch32",
            "frozen_modules": ["all"],
        },
    }
    return YOLOCount(backbone_cfg, neck_cfg, head_cfg)


class YOLOCount(nn.Module):
    """Multimodal object counting network

    Supports multi-class counting: given an image and K text queries the model
    returns a count for each query class independently.

    Args:
        backbone_cfg (dict): Backbone network configuration
        neck_cfg (dict): Feature pyramid network configuration
        head_cfg (dict): Density estimation head configuration
    """

    def __init__(self, backbone_cfg: dict, neck_cfg: dict, head_cfg: dict) -> None:
        super().__init__()

        # Build backbone network
        self.backbone = self._build_backbone(backbone_cfg)

        # Build feature pyramid network
        self.neck = self._build_neck(neck_cfg)

        # Build density estimation head
        self.head = self._build_head(head_cfg)

        self.texts = None
        self.text_feats = None

        self.gradient_checkpointing = False

    def _build_backbone(self, cfg: dict) -> nn.Module:
        """Build backbone network"""
        image_model = YOLOv8CSPDarknet(**cfg.get("image_model", {}))
        text_model = None
        if cfg.get("text_model"):
            text_cfg = cfg["text_model"]
            text_model = HuggingCLIPLanguageBackbone(**text_cfg)

        return MultiModalYOLOBackbone(
            image_model=image_model, text_model=text_model, **cfg.get("kwargs", {})
        )

    def _build_neck(self, cfg: dict) -> nn.Module:
        """Build feature pyramid network"""
        return YOLOCountPAFPN(**cfg)

    def _build_head(self, cfg: dict) -> nn.Module:
        """Build density estimation head"""
        return ProportionCountingHead(**cfg)

    def extract_feat(
        self, batch_inputs: torch.Tensor, texts: Optional[List[List[str]]] = None
    ) -> Tuple[Tuple[torch.Tensor], torch.Tensor]:
        """Extract features

        Args:
            batch_inputs: input image tensor
            texts: list of text descriptions

        Returns:
            img_feats1: tuple of image feature tensors (classification branch)
            img_feats2: tuple of image feature tensors (density branch)
            txt_feats: text features [B, K, 512]
        """
        img_feats, txt_feats = self.backbone(batch_inputs, texts)
        img_feats1, img_feats2 = self.neck(img_feats, txt_feats)
        return img_feats1, img_feats2, txt_feats

    def _forward(
        self, batch_inputs: torch.Tensor, texts: Optional[List[List[str]]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass

        Args:
            batch_inputs: input image tensor [B, 3, H, W]
            texts: list of text descriptions, shape [B, K] of strings
                   (each inner list has K class names for that image)

        Returns:
            cls_logits:    classification logits  [B, K, H, W]
            density_pred:  per-class density maps [B, K, H, W]
        """
        if self.gradient_checkpointing:
            img_feats, txt_feats = checkpoint(
                self.backbone, batch_inputs, texts, use_reentrant=False
            )
            # neck returns (results1, results2) — two feature-pyramid tuples
            img_feats1, img_feats2 = checkpoint(
                self.neck, img_feats, txt_feats, use_reentrant=False
            )
            cls_logits, density_pred = checkpoint(
                self.head, img_feats1, img_feats2, txt_feats, use_reentrant=False
            )
        else:
            img_feats, txt_feats = self.backbone(batch_inputs, texts)
            img_feats1, img_feats2 = self.neck(img_feats, txt_feats)
            cls_logits, density_pred = self.head(img_feats1, img_feats2, txt_feats)
        return cls_logits, density_pred

    def reparameterize(self, texts: List[List[str]]) -> None:
        """Cache text features

        Args:
            texts: list of text descriptions
        """
        self.texts = texts
        self.text_feats = self.backbone.forward_text(texts)

    def strong_loss(
        self,
        image_inputs: torch.Tensor,
        texts: List[List[str]],
        gt_category_labels: torch.Tensor,
        gt_proportion_labels: torch.Tensor,
    ) -> torch.Tensor:
        cls_logits, proportion_pred = self._forward(image_inputs, texts)
        loss_category = F.binary_cross_entropy_with_logits(
            cls_logits, gt_category_labels, reduction="mean"
        )
        loss_proportion = (
            F.l1_loss(proportion_pred, gt_proportion_labels, reduction="sum")
            / image_inputs.shape[0]
        )
        loss = loss_category + loss_proportion
        loss_dict = {
            "loss": loss,
            "loss_category": loss_category,
            "loss_proportion": loss_proportion,
        }
        return loss, loss_dict

    def weak_loss_for_strong_labels(
        self,
        image_inputs: torch.Tensor,
        texts: List[List[str]],
        gt_category_labels: torch.Tensor,
        gt_proportion_labels: torch.Tensor,
    ) -> torch.Tensor:

        cls_logits, proportion_pred = self._forward(image_inputs, texts)
        loss_category = F.binary_cross_entropy_with_logits(
            cls_logits, gt_category_labels, reduction="mean"
        )
        loss_proportion = (
            F.l1_loss(proportion_pred, gt_proportion_labels, reduction="sum")
            / image_inputs.shape[0]
        )
        loss = loss_category + 0.0 * loss_proportion
        loss_dict = {
            "loss": loss,
            "loss_category": loss_category,
            "loss_proportion": loss_proportion,
        }
        return loss, loss_dict

    def weak_loss(
        self,
        image_inputs: torch.Tensor,
        texts: List[List[str]],
        gt_category_labels: torch.Tensor,
        gt_valid_label_masks: torch.Tensor,
    ) -> torch.Tensor:
        """Weakly supervised loss.

        gt_category_labels and gt_valid_label_masks are expected to have shape
        [B, K, H, W] where K is the number of query classes.  For existing
        single-class datasets K=1 so the shapes remain backward-compatible.
        """
        cls_logits, proportion_pred = self._forward(image_inputs, texts)

        valid_positions = gt_valid_label_masks == 1
        if valid_positions.sum() == 0:
            loss_category = torch.tensor(0.0).to(image_inputs.device)
        else:
            gamma = 2.0
            alpha = 0.25
            bce_loss = F.binary_cross_entropy_with_logits(
                cls_logits[valid_positions],
                gt_category_labels[valid_positions],
                reduction="none",
            )
            p = torch.sigmoid(cls_logits[valid_positions])
            p_t = p * gt_category_labels[valid_positions] + (1 - p) * (
                1 - gt_category_labels[valid_positions]
            )
            alpha_t = alpha * gt_category_labels[valid_positions] + (1 - alpha) * (
                1 - gt_category_labels[valid_positions]
            )
            loss_category = (alpha_t * (1 - p_t) ** gamma * bce_loss).mean()

        # gt_counts / pred_counts: [B, K]
        gt_counts = gt_category_labels.sum(dim=(2, 3))    # [B, K]
        pred_counts = proportion_pred.sum(dim=(2, 3))     # [B, K]

        error = pred_counts - gt_counts
        abs_error = torch.abs(error)
        weights = 1.0 / (abs_error.detach() + 1e-6)
        loss_proportion = (weights * abs_error).mean()

        loss = 0.1 * loss_category + loss_proportion
        loss_dict = {
            "loss": loss,
            "loss_category": loss_category,
            "loss_proportion": loss_proportion,
        }
        return loss, loss_dict

    def weak_loss_fake(
        self,
        image_inputs: torch.Tensor,
        texts: List[List[str]],
        gt_category_labels: torch.Tensor,
        gt_proportion_labels: torch.Tensor,
    ) -> torch.Tensor:

        cls_logits, proportion_pred = self._forward(image_inputs, texts)
        cls_probs = torch.sigmoid(cls_logits)

        positive_mask = gt_category_labels == 1
        loss_category = -torch.log(cls_probs[positive_mask] + 1e-6).mean()

        # Cardinality loss — works for any K
        gt_counts = gt_category_labels.sum(dim=(2, 3))    # [B, K]
        pred_counts = proportion_pred.sum(dim=(2, 3))     # [B, K]

        error = pred_counts - gt_counts
        abs_error = torch.abs(error)
        weights = 1.0 / (abs_error.detach() + 1e-6)
        loss_proportion = (weights * abs_error).mean()

        loss = loss_category + loss_proportion
        loss_dict = {
            "loss": loss,
            "loss_category": loss_category,
            "loss_proportion": loss_proportion,
        }
        return loss, loss_dict

    def predict(
        self,
        image_inputs: torch.Tensor,
        texts: List[List[str]],
        original_hw: List[Tuple[int, int]] = None,
        large_threshold: float = float("inf"),
        confidence_threshold: float = 0.0,
        demonstrate: bool = False,
        gt_count: Optional[float] = None,
    ) -> Union[Tuple[torch.Tensor, str], Tuple[torch.Tensor, str, Image.Image]]:
        """Predict per-class counts.

        Args:
            image_inputs:         [B, 3, H, W]
            texts:                [B, K] list-of-lists of class name strings
            original_hw:          original (height, width) before padding
            large_threshold:      if any count exceeds this, apply tiled
                                  re-inference (single-class only; ignored for K>1)
            confidence_threshold: threshold on sigmoid cls probability
            demonstrate:          if True, return a debug image (batch=1 only)
            gt_count:             ground-truth count for annotation in demo mode

        Returns:
            pred_counts: [B, K] tensor of per-class counts
            status:      "no_adaptation" or "adapted"
            demo_image:  (only when demonstrate=True)
        """
        if demonstrate:
            assert (
                image_inputs.shape[0] == len(texts) == 1
            ), "Demonstrate mode only supports batch size 1"

        cls_logits, proportion_pred = self._forward(image_inputs, texts)
        # cls_logits:    [B, K, H, W]
        # proportion_pred: [B, K, H, W]
        cls_probs = torch.sigmoid(cls_logits)

        unmasked_proportion = proportion_pred.clone() if demonstrate else None

        # Apply aspect-ratio mask to ignore padding regions.
        # aspect_based_center_mask returns [B, 1, 80, 80] — broadcasts over K.
        if original_hw is not None and isinstance(original_hw[0], (list, tuple)):
            mask = aspect_based_center_mask(original_hw).to(image_inputs.device)
            # mask shape: [B, 1, 80, 80] — broadcasts correctly over [B, K, H, W]
            cls_probs = cls_probs * mask
            proportion_pred = proportion_pred * mask

        batch_size = image_inputs.shape[0]
        K = cls_probs.shape[1]  # number of query classes

        # Threshold each class channel independently: [B, K, H, W]
        pred_masks = (cls_probs > confidence_threshold).float()

        # Masked density sum → per-class count: [B, K]
        pred_counts = (proportion_pred * pred_masks).sum(dim=(2, 3))

        # ── Multi-class path: skip tiled adaptation (not supported for K>1) ──
        if K > 1:
            if demonstrate:
                # Show only the first class in demo mode
                demo_image = self._generate_demonstration(
                    image_inputs[0],
                    texts[0],
                    cls_probs[0],
                    unmasked_proportion[0],
                    proportion_pred[0],
                    pred_masks[0],
                    pred_counts[0].tolist(),
                    gt_count,
                    confidence_threshold,
                )
                return pred_counts, "no_adaptation", demo_image
            return pred_counts, "no_adaptation"

        # ── Single-class path: optional tiled adaptation for large counts ────
        if pred_counts.max() < large_threshold:
            if demonstrate:
                demo_image = self._generate_demonstration(
                    image_inputs[0],
                    texts[0],
                    cls_probs[0],
                    unmasked_proportion[0],
                    proportion_pred[0],
                    pred_masks[0],
                    pred_counts[0].tolist(),
                    gt_count,
                    confidence_threshold,
                )
                return pred_counts, "no_adaptation", demo_image
            return pred_counts, "no_adaptation"

        # Tiled re-inference (single-class, K=1)
        final_counts = torch.zeros_like(pred_counts)  # [B, 1]

        for b in range(batch_size):
            h, w = original_hw[b]
            if h > w:
                w = int(w * 640 / h)
                h = 640
            else:
                h = int(h * 640 / w)
                w = 640
            pad_h = (640 - h) // 2
            pad_w = (640 - w) // 2

            # Work on class 0 (K=1 guaranteed in this branch)
            density_map = torch.zeros((h, w), device=image_inputs.device)
            valid_pred = (proportion_pred[b, 0] * pred_masks[b, 0])[
                pad_h // 8 : (pad_h + h) // 8, pad_w // 8 : (pad_w + w) // 8
            ]
            density_map = F.interpolate(
                valid_pred.unsqueeze(0).unsqueeze(0),
                size=(h, w),
                mode="bilinear",
                align_corners=False,
            )[0, 0]
            if torch.sum(density_map) != 0:
                density_map = density_map * (
                    torch.sum(valid_pred) / torch.sum(density_map)
                )
            h_mid = h // 2
            w_mid = w // 2

            sub_regions = [
                (slice(0, h_mid), slice(0, w_mid)),
                (slice(0, h_mid), slice(w_mid, w)),
                (slice(h_mid, h), slice(0, w_mid)),
                (slice(h_mid, h), slice(w_mid, w)),
            ]

            for y_slice, x_slice in sub_regions:
                img_h_slice = slice(pad_h + y_slice.start, pad_h + y_slice.stop)
                img_w_slice = slice(pad_w + x_slice.start, pad_w + x_slice.stop)
                sub_img = image_inputs[b : b + 1, :, img_h_slice, img_w_slice]
                h_sub, w_sub = sub_img.shape[2:]
                max_size = max(h_sub, w_sub)
                pad_h_sub = (max_size - h_sub) // 2

                padded_img = F.pad(
                    sub_img, (0, 0, pad_h_sub, pad_h_sub), mode="constant", value=0
                )
                resized_img = F.interpolate(
                    padded_img, size=(640, 640), mode="bilinear", align_corners=False
                )

                sub_cls_logits, sub_proportion_pred = self._forward(
                    resized_img, texts[b : b + 1]
                )
                sub_cls_probs = torch.sigmoid(sub_cls_logits)

                # K=1 in this branch
                pred_mask = sub_cls_probs[0, 0] > confidence_threshold
                sub_pred = sub_proportion_pred[0, 0] * pred_mask

                sub_h = y_slice.stop - y_slice.start
                sub_w = x_slice.stop - x_slice.start

                if sub_pred.sum() > 0:
                    sub_resized = F.interpolate(
                        sub_pred[
                            pad_h // 8 : (pad_h + sub_h) // 8,
                            pad_w // 8 : (pad_w + sub_w) // 8,
                        ]
                        .unsqueeze(0)
                        .unsqueeze(0),
                        size=(sub_h, sub_w),
                        mode="bilinear",
                        align_corners=False,
                    )[0, 0]

                    if torch.sum(sub_resized) != 0:
                        sub_resized = (
                            sub_resized * torch.sum(sub_pred) / torch.sum(sub_resized)
                        )

                    orig_region = density_map[y_slice, x_slice]
                    if torch.sum(sub_resized) > torch.sum(orig_region):
                        density_map[y_slice, x_slice] = sub_resized

            # Store as [B, 1] to match pred_counts shape
            final_counts[b, 0] = torch.sum(density_map)

        if demonstrate:
            demo_image = self._generate_demonstration(
                image_inputs[0],
                texts[0],
                cls_probs[0],
                unmasked_proportion[0],
                proportion_pred[0],
                pred_masks[0],
                final_counts[0].tolist(),
                gt_count,
                confidence_threshold,
            )
            return final_counts, "adapted", demo_image

        return final_counts, "adapted"

    def forward(self, mode: str = "default", **kwargs):
        if mode == "strong_loss":
            return self.strong_loss(**kwargs)
        elif mode == "weak_loss":
            return self.weak_loss(**kwargs)
        elif mode == "weak_loss_fake":
            return self.weak_loss_fake(**kwargs)
        elif mode == "weak_strong_loss":
            return self.weak_loss_for_strong_labels(**kwargs)
        elif mode == "predict":
            return self.predict(**kwargs)
        else:
            return self._forward(**kwargs)

    def _generate_demonstration(
        self,
        image: torch.Tensor,
        text: List[str],
        cls_probs: torch.Tensor,
        unmasked_proportion: torch.Tensor,
        proportion_pred: torch.Tensor,
        pred_mask: torch.Tensor,
        pred_count,
        gt_count: Optional[float],
        confidence_threshold: float,
    ) -> Image.Image:
        """Generate a debug visualisation for a single image.

        Args:
            image:              [3, H, W] float tensor
            text:               list of K class name strings
            cls_probs:          [K, H, W] classification probabilities
            unmasked_proportion:[K, H, W] raw density predictions
            proportion_pred:    [K, H, W] density predictions
            pred_mask:          [K, H, W] binary masks
            pred_count:         scalar or list of K counts
            gt_count:           optional ground-truth count
            confidence_threshold: float
        """
        img = image.permute(1, 2, 0).cpu().numpy()
        img = (img * 255).astype(np.uint8)

        # Average over classes for visualisation
        cls_prob_map = cls_probs.mean(0).detach().cpu().numpy()
        unmasked_density_map = unmasked_proportion.mean(0).detach().cpu().numpy()
        density_map = (proportion_pred * pred_mask).mean(0).detach().cpu().numpy()

        fig, axes = plt.subplots(1, 4, figsize=(20, 6))

        axes[0].imshow(img)
        label_str = ", ".join(text) if isinstance(text, (list, tuple)) else str(text)
        axes[0].set_title(f"Input Image\nText: {label_str}")

        im1 = axes[1].imshow(cls_prob_map, cmap="viridis")
        plt.colorbar(im1, ax=axes[1])
        axes[1].set_title(
            f"Classification Probability\nThreshold: {confidence_threshold:.3f}"
        )

        im2 = axes[2].imshow(unmasked_density_map, cmap="viridis")
        plt.colorbar(im2, ax=axes[2])
        axes[2].set_title("Proportion Logits")

        im3 = axes[3].imshow(density_map, cmap="viridis")
        plt.colorbar(im3, ax=axes[3])
        if isinstance(pred_count, (list, tuple)):
            count_str = ", ".join(f"{c:.2f}" for c in pred_count)
        else:
            count_str = f"{pred_count:.2f}"
        title = f"Proportion Prediction\nCount: {count_str}"
        if gt_count is not None:
            title += f"\nGT Count: {gt_count:.2f}"
        axes[3].set_title(title)

        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])

        plt.tight_layout()
        fig.canvas.draw()
        plt_image = Image.frombytes(
            "RGB", fig.canvas.get_width_height(), fig.canvas.tostring_rgb()
        )
        plt.close()
        return plt_image
