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

    Args:
        backbone_cfg (dict): Backbone network configuration
        neck_cfg (dict): Feature pyramid network configuration
        head_cfg (dict): Density estimation head configuration
        mm_neck (bool): Whether to use multimodal features in the neck
        num_train_classes (int): Number of classes used for training
        num_test_classes (int): Number of classes used for testing
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
            img_feats: tuple of image feature tensors
            txt_feats: text features
        """

        img_feats, txt_feats = self.backbone(batch_inputs, texts)
        img_feats1, img_feats2 = self.neck(img_feats, txt_feats)
        return img_feats1, img_feats2, txt_feats

    def _forward(
        self, batch_inputs: torch.Tensor, texts: Optional[List[List[str]]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass

        Args:
            batch_inputs: input image tensor
            texts: list of text descriptions

        Returns:
            cls_logits: classification logits
            proportion_pred: density prediction map
        """
        if self.gradient_checkpointing:
            img_feats, txt_feats = checkpoint(
                self.backbone, batch_inputs, texts, use_reentrant=False
            )
            img_feats1, img_feats2 = checkpoint(
                self.neck, img_feats, txt_feats, use_reentrant=False
            )
            cls_logits, proportion_pred = checkpoint(
                self.head, img_feats1, img_feats2, txt_feats, use_reentrant=False
            )
        else:
            img_feats, txt_feats = self.backbone(batch_inputs, texts)
            img_feats1, img_feats2 = self.neck(img_feats, txt_feats)
            cls_logits, proportion_pred = self.head(img_feats1, img_feats2, txt_feats)
        return cls_logits, proportion_pred

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

        gt_counts = torch.sum(gt_category_labels, dim=(1, 2, 3))
        pred_counts = torch.sum(proportion_pred, dim=(1, 2, 3))
        loss_proportion = F.l1_loss(pred_counts, gt_counts, reduction="mean")

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
        loss_category = -torch.log(cls_probs[positive_mask]).mean()

        batch_size = image_inputs.shape[0]
        thresholds = np.zeros(batch_size)
        pred_masks = torch.zeros_like(cls_probs).to(image_inputs.device)

        for i in range(batch_size):
            thresholds[i] = 0.0
            pred_masks[i, 0] = cls_probs[i, 0] > thresholds[i].item()
        gt_counts = torch.sum(gt_proportion_labels, dim=(1, 2, 3))
        pred_counts = torch.sum(proportion_pred * pred_masks, dim=(1, 2, 3))
        loss_proportion = F.l1_loss(pred_counts, gt_counts, reduction="mean")

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

        if demonstrate:
            assert (
                image_inputs.shape[0] == len(texts) == len(original_hw) == 1
            ), "Demonstrate mode only supports batch size 1"

        cls_logits, proportion_pred = self._forward(image_inputs, texts)
        cls_probs = torch.sigmoid(cls_logits)

        unmasked_proportion = proportion_pred.clone() if demonstrate else None

        if original_hw is not None:
            mask = aspect_based_center_mask(original_hw).to(image_inputs.device)
            cls_probs = cls_probs * mask
            proportion_pred = proportion_pred * mask

        batch_size = image_inputs.shape[0]
        pred_masks = torch.zeros_like(cls_probs).to(image_inputs.device)
        for i in range(batch_size):
            pred_masks[i, 0] = cls_probs[i, 0] > confidence_threshold

        pred_counts = torch.sum(proportion_pred * pred_masks, dim=(1, 2, 3))

        if not demonstrate and pred_counts.max() < large_threshold:
            return pred_counts, "no_adaptation"

        if demonstrate and pred_counts.max() < large_threshold:
            demo_image = self._generate_demonstration(
                image_inputs[0],
                texts[0],
                cls_probs[0],
                unmasked_proportion[0],
                proportion_pred[0],
                pred_masks[0],
                pred_counts[0].item(),
                gt_count,
                confidence_threshold,
            )
            return pred_counts, "no_adaptation", demo_image

        final_counts = torch.zeros_like(pred_counts)

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

            final_counts[b] = torch.sum(density_map)

        if demonstrate:
            demo_image = self._generate_demonstration(
                image_inputs[0],
                texts[0],
                cls_probs[0],
                unmasked_proportion[0],
                proportion_pred[0],
                pred_masks[0],
                final_counts[0].item(),
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
        pred_count: float,
        gt_count: Optional[float],
        confidence_threshold: float,
    ) -> Image.Image:

        img = image.permute(1, 2, 0).cpu().numpy()
        img = (img * 255).astype(np.uint8)

        cls_prob_map = cls_probs.squeeze().detach().cpu().numpy()
        unmasked_density_map = unmasked_proportion.squeeze().detach().cpu().numpy()
        density_map = (proportion_pred * pred_mask).squeeze().detach().cpu().numpy()

        fig, axes = plt.subplots(1, 4, figsize=(20, 6))

        axes[0].imshow(img)
        axes[0].set_title(f"Input Image\nText: {text[0]}")

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
        title = f"Proportion Prediction\nCount: {pred_count:.2f}"
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
