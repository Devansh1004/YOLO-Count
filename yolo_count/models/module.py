from typing import Union, Tuple, Optional, Sequence, Dict
import math
import torch
import torch.nn as nn


def make_divisible(x: float, widen_factor: float = 1.0, divisor: int = 8) -> int:
    """Make sure that x*widen_factor is divisible by divisor."""
    return math.ceil(x * widen_factor / divisor) * divisor


def make_round(x: float, deepen_factor: float = 1.0) -> int:
    """Make sure that x*deepen_factor becomes an integer not less than 1."""
    return max(round(x * deepen_factor), 1) if x > 1 else x


class ConvModule(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        groups: int = 1,
        bias: Union[bool, str] = "auto",
        conv_cfg: Optional[Dict] = None,
        norm_cfg: Optional[Dict] = None,
        act_cfg: Optional[Dict] = dict(type="ReLU"),
        inplace: bool = True,
        with_spectral_norm: bool = False,
        padding_mode: str = "zeros",
        order: tuple = ("conv", "norm", "act"),
    ):
        super().__init__()

        self.with_explicit_padding = padding_mode not in ["zeros", "circular"]
        if self.with_explicit_padding:
            self.padding_layer = (
                nn.ReflectionPad2d(padding)
                if padding_mode == "reflect"
                else nn.ReplicationPad2d(padding)
            )
            conv_padding = 0
        else:
            conv_padding = padding

        self.with_norm = norm_cfg is not None
        self.with_activation = act_cfg is not None
        if bias == "auto":
            bias = not self.with_norm

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=conv_padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode if not self.with_explicit_padding else "zeros",
        )

        # Spectral Norm
        if with_spectral_norm:
            self.conv = nn.utils.spectral_norm(self.conv)

        if self.with_norm:
            norm_channels = (
                out_channels
                if order.index("norm") > order.index("conv")
                else in_channels
            )
            if norm_cfg["type"] == "BN":
                self.norm = nn.BatchNorm2d(norm_channels, momentum=0.03, eps=0.001)
            elif norm_cfg["type"] == "IN":
                self.norm = nn.InstanceNorm2d(norm_channels)
        else:
            self.norm = None

        if self.with_activation:
            if act_cfg["type"] == "ReLU":
                self.activation = nn.ReLU(inplace=inplace)
            elif act_cfg["type"] == "LeakyReLU":
                self.activation = nn.LeakyReLU(inplace=inplace)
            elif act_cfg["type"] == "GELU":
                self.activation = nn.GELU()
            elif act_cfg["type"] == "SiLU":
                self.activation = nn.SiLU(inplace=inplace)
        else:
            self.activation = None

        self.order = order
        self.init_weights()

    def init_weights(self):
        if (
            self.with_activation
            and hasattr(self, "act_cfg")
            and self.act_cfg["type"] == "LeakyReLU"
        ):
            nn.init.kaiming_normal_(
                self.conv.weight,
                a=self.act_cfg.get("negative_slope", 0.01),
                nonlinearity="leaky_relu",
            )
        else:
            nn.init.kaiming_normal_(self.conv.weight, nonlinearity="relu")

        if self.norm is not None:
            nn.init.constant_(self.norm.weight, 1)
            nn.init.constant_(self.norm.bias, 0)

    def forward(
        self, x: torch.Tensor, activate: bool = True, norm: bool = True
    ) -> torch.Tensor:
        for layer in self.order:
            if layer == "conv":
                if self.with_explicit_padding:
                    x = self.padding_layer(x)
                x = self.conv(x)
            elif layer == "norm" and norm and self.norm is not None:
                x = self.norm(x)
            elif layer == "act" and activate and self.activation is not None:
                x = self.activation(x)
        return x


class DepthwiseSeparableConvModule(nn.Module):
    """Depthwise separable convolution module.

    See https://arxiv.org/pdf/1704.04861.pdf for details.

    This module can replace a ConvModule with the conv block replaced by two
    conv block: depthwise conv block and pointwise conv block. The depthwise
    conv block contains depthwise-conv/norm/activation layers. The pointwise
    conv block contains pointwise-conv/norm/activation layers.

    Args:
        in_channels (int): Number of channels in the input feature map.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int | tuple[int]): Size of the convolving kernel.
        stride (int | tuple[int]): Stride of the convolution. Default: 1.
        padding (int | tuple[int]): Zero-padding added to both sides of
            the input. Default: 0.
        dilation (int | tuple[int]): Spacing between kernel elements. Default: 1.
        norm_cfg (dict): Default norm config for both depthwise ConvModule and
            pointwise ConvModule. Default: None.
        act_cfg (dict): Default activation config for both depthwise ConvModule
            and pointwise ConvModule. Default: dict(type='ReLU').
        dw_norm_cfg (dict): Norm config of depthwise ConvModule. If it is
            'default', it will be the same as `norm_cfg`. Default: 'default'.
        dw_act_cfg (dict): Activation config of depthwise ConvModule. If it is
            'default', it will be the same as `act_cfg`. Default: 'default'.
        pw_norm_cfg (dict): Norm config of pointwise ConvModule. If it is
            'default', it will be the same as `norm_cfg`. Default: 'default'.
        pw_act_cfg (dict): Activation config of pointwise ConvModule. If it is
            'default', it will be the same as `act_cfg`. Default: 'default'.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        norm_cfg: Optional[Dict] = None,
        act_cfg: Dict = dict(type="ReLU"),
        dw_norm_cfg: Union[Dict, str] = "default",
        dw_act_cfg: Union[Dict, str] = "default",
        pw_norm_cfg: Union[Dict, str] = "default",
        pw_act_cfg: Union[Dict, str] = "default",
        **kwargs
    ):
        super().__init__()
        assert "groups" not in kwargs, "groups should not be specified"

        # if norm/activation config of depthwise/pointwise ConvModule is not
        # specified, use default config.
        dw_norm_cfg = dw_norm_cfg if dw_norm_cfg != "default" else norm_cfg
        dw_act_cfg = dw_act_cfg if dw_act_cfg != "default" else act_cfg
        pw_norm_cfg = pw_norm_cfg if pw_norm_cfg != "default" else norm_cfg
        pw_act_cfg = pw_act_cfg if pw_act_cfg != "default" else act_cfg

        # depthwise convolution
        self.depthwise_conv = ConvModule(
            in_channels,
            in_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            norm_cfg=dw_norm_cfg,
            act_cfg=dw_act_cfg,
            **kwargs
        )

        self.pointwise_conv = ConvModule(
            in_channels,
            out_channels,
            1,
            norm_cfg=pw_norm_cfg,
            act_cfg=pw_act_cfg,
            **kwargs
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        return x


class DarknetBottleneck(nn.Module):
    """The basic bottleneck block used in Darknet.

    Each ResBlock consists of two ConvModules and the input is added to the
    final output. Each ConvModule is composed of Conv, BN, and LeakyReLU.
    The first convLayer has filter size of k1Xk1 and the second one has the
    filter size of k2Xk2.

    Args:
        in_channels (int): The input channels of this Module.
        out_channels (int): The output channels of this Module.
        expansion (float): The kernel size for hidden channel.
            Defaults to 0.5.
        kernel_size (Sequence[int]): The kernel size of the convolution.
            Defaults to (1, 3).
        padding (Sequence[int]): The padding size of the convolution.
            Defaults to (0, 1).
        add_identity (bool): Whether to add identity to the out.
            Defaults to True
        use_depthwise (bool): Whether to use depthwise separable convolution.
            Defaults to False
        conv_cfg (dict): Config dict for convolution layer. Default: None,
            which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to dict(type='BN', momentum=0.03, eps=0.001).
        act_cfg (dict): Config dict for activation layer.
            Defaults to dict(type='SiLU', inplace=True).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expansion: float = 0.5,
        kernel_size: Sequence[int] = (1, 3),
        padding: Sequence[int] = (0, 1),
        add_identity: bool = True,
        use_depthwise: bool = False,
        conv_cfg: Optional[dict] = None,
        norm_cfg: dict = dict(type="BN", momentum=0.03, eps=0.001),
        act_cfg: dict = dict(type="SiLU", inplace=True),
    ) -> None:
        super().__init__()

        hidden_channels = int(out_channels * expansion)
        conv = DepthwiseSeparableConvModule if use_depthwise else ConvModule
        assert isinstance(kernel_size, Sequence) and len(kernel_size) == 2

        self.conv1 = ConvModule(
            in_channels,
            hidden_channels,
            kernel_size[0],
            padding=padding[0],
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )

        self.conv2 = conv(
            hidden_channels,
            out_channels,
            kernel_size[1],
            stride=1,
            padding=padding[1],
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )

        self.add_identity = add_identity and in_channels == out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)

        if self.add_identity:
            return out + identity
        return out


class CSPLayerWithTwoConv(nn.Module):
    """Cross Stage Partial Layer with 2 convolutions.

    Args:
        in_channels (int): The input channels of the CSP layer.
        out_channels (int): The output channels of the CSP layer.
        expand_ratio (float): Ratio to adjust the number of channels of the
            hidden layer. Defaults to 0.5.
        num_blocks (int): Number of blocks. Defaults to 1
        add_identity (bool): Whether to add identity in blocks.
            Defaults to True.
        conv_cfg (dict, optional): Config dict for convolution layer.
            Defaults to None, which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to dict(type='BN', momentum=0.03, eps=0.001).
        act_cfg (dict): Config dict for activation layer.
            Defaults to dict(type='SiLU', inplace=True).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expand_ratio: float = 0.5,
        num_blocks: int = 1,
        add_identity: bool = True,  # shortcut
        conv_cfg: Optional[dict] = None,
        norm_cfg: dict = dict(type="BN", momentum=0.03, eps=0.001),
        act_cfg: dict = dict(type="SiLU", inplace=True),
    ) -> None:
        super().__init__()

        self.mid_channels = int(out_channels * expand_ratio)
        self.main_conv = ConvModule(
            in_channels,
            2 * self.mid_channels,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )

        self.final_conv = ConvModule(
            (2 + num_blocks) * self.mid_channels,
            out_channels,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )

        self.blocks = nn.ModuleList(
            DarknetBottleneck(
                self.mid_channels,
                self.mid_channels,
                expansion=1,
                kernel_size=(3, 3),
                padding=(1, 1),
                add_identity=add_identity,
                use_depthwise=False,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
            )
            for _ in range(num_blocks)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward process."""
        x_main = self.main_conv(x)
        x_main = list(x_main.split((self.mid_channels, self.mid_channels), 1))
        x_main.extend(blocks(x_main[-1]) for blocks in self.blocks)
        return self.final_conv(torch.cat(x_main, 1))


class SPPFBottleneck(nn.Module):
    """Spatial pyramid pooling - Fast (SPPF) layer for YOLOv5, YOLOX and PPYOLOE.

    Args:
        in_channels (int): The input channels of this Module.
        out_channels (int): The output channels of this Module.
        kernel_sizes (int, tuple[int]): Sequential or number of kernel
            sizes of pooling layers. Defaults to 5.
        use_conv_first (bool): Whether to use conv before pooling layer.
            In YOLOv5 and YOLOX, the para set to True.
            In PPYOLOE, the para set to False.
            Defaults to True.
        mid_channels_scale (float): Channel multiplier, multiply in_channels
            by this amount to get mid_channels. This parameter is valid only
            when use_conv_fist=True. Defaults to 0.5.
        conv_cfg (dict, optional): Config dict for convolution layer.
            Defaults to None, which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to dict(type='BN', momentum=0.03, eps=0.001).
        act_cfg (dict): Config dict for activation layer.
            Defaults to dict(type='SiLU', inplace=True).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_sizes: Union[int, Sequence[int]] = 5,
        use_conv_first: bool = True,
        mid_channels_scale: float = 0.5,
        conv_cfg: Optional[dict] = None,
        norm_cfg: dict = dict(type="BN", momentum=0.03, eps=0.001),
        act_cfg: dict = dict(type="SiLU", inplace=True),
    ):
        super().__init__()

        if use_conv_first:
            mid_channels = int(in_channels * mid_channels_scale)
            self.conv1 = ConvModule(
                in_channels,
                mid_channels,
                1,
                stride=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
            )
        else:
            mid_channels = in_channels
            self.conv1 = None

        self.kernel_sizes = kernel_sizes
        if isinstance(kernel_sizes, int):
            self.poolings = nn.MaxPool2d(
                kernel_size=kernel_sizes, stride=1, padding=kernel_sizes // 2
            )
            conv2_in_channels = mid_channels * 4
        else:
            self.poolings = nn.ModuleList(
                [
                    nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks // 2)
                    for ks in kernel_sizes
                ]
            )
            conv2_in_channels = mid_channels * (len(kernel_sizes) + 1)

        self.conv2 = ConvModule(
            conv2_in_channels,
            out_channels,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward process
        Args:
            x (torch.Tensor): The input tensor.
        """
        if self.conv1:
            x = self.conv1(x)

        if isinstance(self.kernel_sizes, int):
            y1 = self.poolings(x)
            y2 = self.poolings(y1)
            x = torch.cat([x, y1, y2, self.poolings(y2)], dim=1)
        else:
            x = torch.cat([x] + [pooling(x) for pooling in self.poolings], dim=1)

        x = self.conv2(x)
        return x


class MaxSigmoidAttnBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        guide_channels: int,
        embed_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
        num_heads: int = 1,
        use_depthwise: bool = False,
        with_scale: bool = False,
        conv_cfg: Optional[dict] = None,
        norm_cfg: dict = dict(type="BN", momentum=0.03, eps=0.001),
        use_einsum: bool = True,
    ) -> None:
        super().__init__()

        assert (
            out_channels % num_heads == 0 and embed_channels % num_heads == 0
        ), "out_channels and embed_channels should be divisible by num_heads."

        self.num_heads = num_heads
        self.head_channels = embed_channels // num_heads
        self.use_einsum = use_einsum

        self.embed_conv = (
            None
            if embed_channels == in_channels
            else ConvModule(
                in_channels,
                embed_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=None,
            )
        )

        self.guide_fc = nn.Linear(guide_channels, embed_channels)

        self.bias = nn.Parameter(torch.zeros(num_heads))
        self.scale = nn.Parameter(torch.ones(1, num_heads, 1, 1)) if with_scale else 1.0

        conv_layer = DepthwiseSeparableConvModule if use_depthwise else ConvModule
        self.project_conv = conv_layer(
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=padding,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=None,
        )

    def forward(self, x: torch.Tensor, guide: torch.Tensor) -> torch.Tensor:
        """Forward process."""
        B, _, H, W = x.shape

        guide = self.guide_fc(guide)
        guide = guide.reshape(B, -1, self.num_heads, self.head_channels)
        embed = self.embed_conv(x) if self.embed_conv is not None else x
        embed = embed.reshape(B, self.num_heads, self.head_channels, H, W)

        if self.use_einsum:
            attn_weight = torch.einsum("bmchw,bnmc->bmhwn", embed, guide)
        else:
            batch, m, channel, height, width = embed.shape
            _, n, _, _ = guide.shape
            embed = embed.permute(0, 1, 3, 4, 2)
            embed = embed.reshape(batch, m, -1, channel)
            guide = guide.permute(0, 2, 3, 1)
            attn_weight = torch.matmul(embed, guide)
            attn_weight = attn_weight.reshape(batch, m, height, width, n)

        attn_weight = attn_weight.max(dim=-1)[0]
        attn_weight = attn_weight / (self.head_channels**0.5)
        attn_weight = attn_weight + self.bias[None, :, None, None]
        attn_weight = attn_weight.sigmoid() * self.scale

        x = self.project_conv(x)
        x = x.reshape(B, self.num_heads, -1, H, W)
        x = x * attn_weight.unsqueeze(2)
        x = x.reshape(B, -1, H, W)
        return x


class MaxSigmoidCSPLayerWithTwoConv(nn.Module):
    """Sigmoid-attention based CSP layer with two convolution layers."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        guide_channels: int,
        embed_channels: int,
        num_heads: int = 1,
        expand_ratio: float = 0.5,
        num_blocks: int = 1,
        with_scale: bool = False,
        add_identity: bool = True,
        conv_cfg: Optional[dict] = None,
        norm_cfg: dict = dict(type="BN", momentum=0.03, eps=0.001),
        act_cfg: dict = dict(type="SiLU", inplace=True),
        use_einsum: bool = True,
    ) -> None:
        super().__init__()

        self.mid_channels = int(out_channels * expand_ratio)

        # Main trunk convolution
        self.main_conv = ConvModule(
            in_channels,
            2 * self.mid_channels,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )

        self.blocks = nn.ModuleList(
            [
                DarknetBottleneck(
                    self.mid_channels,
                    self.mid_channels,
                    expansion=1,
                    kernel_size=(3, 3),
                    padding=(1, 1),
                    add_identity=add_identity,
                    use_depthwise=False,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                )
                for _ in range(num_blocks)
            ]
        )

        self.attn_block = MaxSigmoidAttnBlock(
            self.mid_channels,
            self.mid_channels,
            guide_channels=guide_channels,
            embed_channels=embed_channels,
            num_heads=num_heads,
            with_scale=with_scale,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            use_einsum=use_einsum,
        )

        self.final_conv = ConvModule(
            (3 + num_blocks) * self.mid_channels,
            out_channels,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )

    def forward(self, x: torch.Tensor, guide: torch.Tensor) -> torch.Tensor:
        """Forward process."""
        x_main = self.main_conv(x)
        x_main = list(x_main.split((self.mid_channels, self.mid_channels), 1))
        x_main.extend(blocks(x_main[-1]) for blocks in self.blocks)
        x_main.append(self.attn_block(x_main[-1], guide))
        return self.final_conv(torch.cat(x_main, 1))
