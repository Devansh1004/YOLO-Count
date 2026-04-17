from typing import List, Union
import torch
import torch.nn as nn
from yolo_count.models.module import (
    ConvModule,
    make_divisible,
    make_round,
    MaxSigmoidCSPLayerWithTwoConv,
)


class YOLOCountPAFPN(nn.Module):
    """Path Aggregation Network used in YOLO World."""

    def __init__(
        self,
        in_channels: List[int],
        out_channels: Union[List[int], int],
        guide_channels: int,
        embed_channels: List[int],
        num_heads: List[int],
        deepen_factor: float = 1.0,
        widen_factor: float = 1.0,
        num_csp_blocks: int = 3,
        freeze_all: bool = False,
        norm_cfg: dict = dict(type="BN", momentum=0.03, eps=0.001),
        act_cfg: dict = dict(type="SiLU", inplace=True),
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = (
            [out_channels] * len(in_channels)
            if isinstance(out_channels, int)
            else out_channels
        )
        self.guide_channels = guide_channels
        self.embed_channels = embed_channels
        self.num_heads = num_heads
        self.deepen_factor = deepen_factor
        self.widen_factor = widen_factor
        self.num_csp_blocks = num_csp_blocks
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg

        # build layers
        self.reduce_layers = nn.ModuleList()
        self.top_down_layers = nn.ModuleList()
        self.bottom_up_layers = nn.ModuleList()
        self.top_down_layers2 = nn.ModuleList()
        self.downsample_layers = nn.ModuleList()
        self.upsample_layers = nn.ModuleList()
        self.out_layers = nn.ModuleList()
        self.out_layers2 = nn.ModuleList()

        # build reduce conv
        for idx in range(len(in_channels)):
            self.reduce_layers.append(nn.Identity())

        # build top-down blocks
        for idx in range(len(in_channels) - 1, 0, -1):
            self.upsample_layers.append(nn.Upsample(scale_factor=2, mode="nearest"))
            self.top_down_layers.append(self.build_top_down_layer(idx))

        # build bottom-up blocks
        for idx in range(len(in_channels) - 1):
            self.downsample_layers.append(
                ConvModule(
                    make_divisible(self.out_channels[idx], widen_factor),
                    make_divisible(self.out_channels[idx], widen_factor),
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                )
            )
            self.bottom_up_layers.append(self.build_bottom_up_layer(idx))

        # build second top-down blocks
        for idx in range(len(in_channels) - 1, 0, -1):
            self.top_down_layers2.append(self.build_top_down_layer(idx))

        # build out layers
        for _ in range(len(in_channels)):
            self.out_layers.append(nn.Identity())
            self.out_layers2.append(nn.Identity())

        self.upsample_feats_cat_first = True

        if freeze_all:
            for m in self.modules():
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def build_top_down_layer(self, idx: int) -> nn.Module:
        """build top down layer."""
        return MaxSigmoidCSPLayerWithTwoConv(
            in_channels=make_divisible(
                (self.in_channels[idx - 1] + self.in_channels[idx]), self.widen_factor
            ),
            out_channels=make_divisible(self.out_channels[idx - 1], self.widen_factor),
            guide_channels=self.guide_channels,
            embed_channels=make_round(self.embed_channels[idx - 1], self.widen_factor),
            num_heads=make_round(self.num_heads[idx - 1], self.widen_factor),
            num_blocks=make_round(self.num_csp_blocks, self.deepen_factor),
            add_identity=False,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
        )

    def build_bottom_up_layer(self, idx: int) -> nn.Module:
        """build bottom up layer."""
        return MaxSigmoidCSPLayerWithTwoConv(
            in_channels=make_divisible(
                (self.out_channels[idx] + self.out_channels[idx + 1]), self.widen_factor
            ),
            out_channels=make_divisible(self.out_channels[idx + 1], self.widen_factor),
            guide_channels=self.guide_channels,
            embed_channels=make_round(self.embed_channels[idx + 1], self.widen_factor),
            num_heads=make_round(self.num_heads[idx + 1], self.widen_factor),
            num_blocks=make_round(self.num_csp_blocks, self.deepen_factor),
            add_identity=False,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
        )

    def forward(
        self, img_feats: List[torch.Tensor], txt_feats: torch.Tensor = None
    ) -> tuple:
        """Forward function."""
        
        if txt_feats is not None:
            assert txt_feats.dim() == 3, \
                f"Expected txt_feats shape [B, K, D], got {txt_feats.shape}"
                
        if txt_feats is not None:
            assert txt_feats.shape[0] == img_feats[0].shape[0], \
                "Batch size mismatch between image and text"
        
        assert len(img_feats) == len(
            self.in_channels
        ), f"The length of img_feats must be equal to the length of in_channels, but got {len(img_feats)} and {len(self.in_channels)}"
        # reduce layers
        reduce_outs = []
        for idx in range(len(self.in_channels)):
            reduce_outs.append(self.reduce_layers[idx](img_feats[idx]))

        # top-down path
        inner_outs = [reduce_outs[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_high = inner_outs[0]
            feat_low = reduce_outs[idx - 1]
            upsample_feat = self.upsample_layers[len(self.in_channels) - 1 - idx](
                feat_high
            )
            if self.upsample_feats_cat_first:
                top_down_layer_inputs = torch.cat([upsample_feat, feat_low], 1)
            else:
                top_down_layer_inputs = torch.cat([feat_low, upsample_feat], 1)
            inner_out = self.top_down_layers[len(self.in_channels) - 1 - idx](
                top_down_layer_inputs, txt_feats
            )
            inner_outs.insert(0, inner_out)

        # bottom-up path
        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_high = inner_outs[idx + 1]
            downsample_feat = self.downsample_layers[idx](feat_low)
            out = self.bottom_up_layers[idx](
                torch.cat([downsample_feat, feat_high], 1), txt_feats
            )
            outs.append(out)

        # first results
        results1 = []
        for idx in range(len(self.in_channels)):
            results1.append(self.out_layers[idx](outs[idx]))

        # second top-down path
        inner_outs2 = [outs[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_high = inner_outs2[0]
            feat_low = outs[idx - 1]
            upsample_feat = self.upsample_layers[len(self.in_channels) - 1 - idx](
                feat_high
            )
            if self.upsample_feats_cat_first:
                top_down_layer_inputs = torch.cat([upsample_feat, feat_low], 1)
            else:
                top_down_layer_inputs = torch.cat([feat_low, upsample_feat], 1)
            inner_out = self.top_down_layers2[len(self.in_channels) - 1 - idx](
                top_down_layer_inputs, txt_feats
            )
            inner_outs2.insert(0, inner_out)

        # second results
        results2 = []
        for idx in range(len(self.in_channels)):
            results2.append(self.out_layers2[idx](inner_outs2[idx]))

        return tuple(results1), tuple(results2)
