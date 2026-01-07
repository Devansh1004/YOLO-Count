from typing import List, Tuple, Union, Sequence, Optional
import itertools
import torch
import torch.nn as nn

from transformers import (
    AutoTokenizer,
    CLIPTextConfig,
    CLIPTextModelWithProjection as CLIPTP,
)
from yolo_count.models.module import (
    ConvModule,
    make_divisible,
    make_round,
    CSPLayerWithTwoConv,
    SPPFBottleneck,
)


class HuggingCLIPLanguageBackbone(nn.Module):
    def __init__(
        self,
        model_name: str,
        frozen_modules: Sequence[str] = (),
        dropout: float = 0.0,
        training_use_cache: bool = False,
    ) -> None:
        super().__init__()

        self.frozen_modules = frozen_modules
        self.training_use_cache = training_use_cache
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        clip_config = CLIPTextConfig.from_pretrained(
            model_name, attention_dropout=dropout
        )
        self.model = CLIPTP.from_pretrained(model_name, config=clip_config)
        self._freeze_modules()

    def forward_tokenizer(self, texts):
        if not hasattr(self, "text"):
            text = list(itertools.chain(*texts))
            text = self.tokenizer(text=text, return_tensors="pt", padding=True)
            self.text = text.to(device=self.model.device)
        return self.text

    def forward(self, text: List[List[str]]) -> torch.Tensor:
        num_per_batch = [len(t) for t in text]
        assert max(num_per_batch) == min(
            num_per_batch
        ), "number of sequences not equal in batch"
        text = list(itertools.chain(*text))
        text = self.tokenizer(text=text, return_tensors="pt", padding=True)
        text = text.to(device=self.model.device)
        txt_outputs = self.model(**text)
        txt_feats = txt_outputs.text_embeds
        txt_feats = txt_feats / txt_feats.norm(p=2, dim=-1, keepdim=True)
        txt_feats = txt_feats.reshape(-1, num_per_batch[0], txt_feats.shape[-1])
        return txt_feats

    def _freeze_modules(self):
        if len(self.frozen_modules) == 0:
            # not freeze
            return
        if self.frozen_modules[0] == "all":
            self.model.eval()
            for _, module in self.model.named_modules():
                module.eval()
                for param in module.parameters():
                    param.requires_grad = False
            return
        for name, module in self.model.named_modules():
            for frozen_name in self.frozen_modules:
                if name.startswith(frozen_name):
                    module.eval()
                    for param in module.parameters():
                        param.requires_grad = False
                    break

    def train(self, mode=True):
        super().train(mode)
        self._freeze_modules()


class YOLOv8CSPDarknet(nn.Module):
    """CSP-Darknet backbone used in YOLOv8.

    Args:
        arch (str): Architecture of CSP-Darknet, from {P5}.
            Defaults to P5.
        last_stage_out_channels (int): Final layer output channel.
            Defaults to 1024.
        plugins (list[dict]): List of plugins for stages, each dict contains:
            - cfg (dict, required): Cfg dict to build plugin.
            - stages (tuple[bool], optional): Stages to apply plugin, length
              should be same as 'num_stages'.
        deepen_factor (float): Depth multiplier, multiply number of
            blocks in CSP layer by this amount. Defaults to 1.0.
        widen_factor (float): Width multiplier, multiply number of
            channels in each layer by this amount. Defaults to 1.0.
        input_channels (int): Number of input image channels. Defaults to: 3.
        out_indices (Tuple[int]): Output from which stages.
            Defaults to (2, 3, 4).
        frozen_stages (int): Stages to be frozen (stop grad and set eval
            mode). -1 means not freezing any parameters. Defaults to -1.
        norm_cfg (dict): Dictionary to construct and config norm layer.
            Defaults to dict(type='BN', momentum=0.03, eps=0.001).
        act_cfg (dict): Config dict for activation layer.
            Defaults to dict(type='SiLU', inplace=True).
    """

    # From left to right:
    # in_channels, out_channels, num_blocks, add_identity, use_spp
    arch_settings = {
        "P5": [
            [64, 128, 3, True, False],
            [128, 256, 6, True, False],
            [256, 512, 6, True, False],
            [512, None, 3, True, True],
        ],
    }

    def __init__(
        self,
        arch: str = "P5",
        last_stage_out_channels: int = 1024,
        plugins: Union[dict, List[dict]] = None,
        deepen_factor: float = 1.0,
        widen_factor: float = 1.0,
        input_channels: int = 3,
        out_indices: Tuple[int] = (2, 3, 4),
        frozen_stages: int = -1,
        norm_cfg: dict = dict(type="BN", momentum=0.03, eps=0.001),
        act_cfg: dict = dict(type="SiLU", inplace=True),
    ):
        super().__init__()

        self.arch_setting = self.arch_settings[arch].copy()
        self.arch_setting[-1][1] = last_stage_out_channels

        self.plugins = plugins
        self.deepen_factor = deepen_factor
        self.widen_factor = widen_factor
        self.input_channels = input_channels
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg

        # build stem
        self.stem = self.build_stem_layer()

        # build stages
        self.stages = nn.ModuleList()
        for i, setting in enumerate(self.arch_setting):
            stage = self.build_stage_layer(i, setting)
            self.stages.append(nn.Sequential(*stage))

        self._freeze_stages()

    def build_stem_layer(self) -> nn.Module:
        """Build a stem layer."""
        return ConvModule(
            self.input_channels,
            make_divisible(self.arch_setting[0][0], self.widen_factor),
            kernel_size=3,
            stride=2,
            padding=1,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
        )

    def build_stage_layer(self, stage_idx: int, setting: list) -> list:
        """Build a stage layer.

        Args:
            stage_idx (int): The index of a stage layer.
            setting (list): The architecture setting of a stage layer.
        """
        in_channels, out_channels, num_blocks, add_identity, use_spp = setting

        in_channels = make_divisible(in_channels, self.widen_factor)
        out_channels = make_divisible(out_channels, self.widen_factor)
        num_blocks = make_round(num_blocks, self.deepen_factor)

        stage = []
        # build conv layer
        conv_layer = ConvModule(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
        )
        stage.append(conv_layer)

        # build CSP layer
        csp_layer = CSPLayerWithTwoConv(
            out_channels,
            out_channels,
            num_blocks=num_blocks,
            add_identity=add_identity,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
        )
        stage.append(csp_layer)

        # build SPP layer
        if use_spp:
            spp = SPPFBottleneck(
                out_channels,
                out_channels,
                kernel_sizes=5,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
            )
            stage.append(spp)

        return stage

    def _freeze_stages(self):
        """Freeze stages param and norm stats."""
        if self.frozen_stages >= 0:
            self.stem.eval()
            for param in self.stem.parameters():
                param.requires_grad = False

        for i in range(self.frozen_stages):
            if i < len(self.stages):
                stage = self.stages[i]
                stage.eval()
                for param in stage.parameters():
                    param.requires_grad = False

    def forward(self, x):
        x = self.stem(x)
        outs = []

        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i + 1 in self.out_indices:
                outs.append(x)

        return tuple(outs)

    def train(self, mode=True):
        super().train(mode)
        self._freeze_stages()


class MultiModalYOLOBackbone(nn.Module):
    def __init__(
        self,
        image_model: nn.Module,
        text_model: Optional[nn.Module] = None,
        frozen_stages: int = -1,
        with_text_model: bool = True,
    ) -> None:
        super().__init__()

        self.with_text_model = with_text_model
        self.image_model = image_model
        self.text_model = text_model if with_text_model else None
        self.frozen_stages = frozen_stages

        if frozen_stages >= 0:
            self._freeze_stages()

    def _freeze_stages(self) -> None:
        if hasattr(self.image_model, "layers"):
            for i in range(self.frozen_stages + 1):
                layer = getattr(self.image_model, self.image_model.layers[i])
                layer.eval()
                for param in layer.parameters():
                    param.requires_grad = False

    def train(self, mode: bool = True) -> None:
        super().train(mode)
        if self.frozen_stages >= 0:
            self._freeze_stages()

    def forward(
        self, image: torch.Tensor, text: List[List[str]]
    ) -> Tuple[Tuple[torch.Tensor], Optional[torch.Tensor]]:
        img_feats = self.image_model(image)

        if self.with_text_model:
            txt_feats = self.text_model(text)
            return img_feats, txt_feats

        return img_feats, None

    def forward_text(self, text: List[List[str]]) -> torch.Tensor:
        return self.text_model(text)

    def forward_image(self, image: torch.Tensor) -> Tuple[torch.Tensor]:
        return self.image_model(image)
