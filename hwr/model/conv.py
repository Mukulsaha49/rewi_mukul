import torch
import torch.nn as nn

__all__ = ['BLCNN']


class PatchEmbed(nn.Module):
    '''ConvNeXt-style patch embedding layer.
    Downsamples by stride (default = 2).'''
    def __init__(
        self, in_chan: int, out_chan: int, kernel: int = 2, stride: int = 2
    ) -> None:
        super().__init__()
        self.conv = nn.Conv1d(in_chan, out_chan, kernel, stride)
        self.norm = nn.InstanceNorm1d(out_chan)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        return x


class Conv(nn.Module):
    '''Depthwise + pointwise conv block with GELU and dropout.'''
    def __init__(
        self,
        dim: int,
        kernel: int = 5,
        r_drop: float = 0.2,
    ) -> None:
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim * 2, kernel, padding='same', groups=dim)
        self.pwconv = nn.Conv1d(dim * 2, dim, 1)
        self.norm = nn.InstanceNorm1d(dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(r_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dwconv(x)
        x = self.pwconv(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.drop(x)
        return x


class BLCNN(nn.Module):
    '''Convolutional baseline encoder with downsampling via PatchEmbed.'''

    def __init__(
        self,
        in_chan: int,
        depths: list[int] = [3, 3, 3],
        dims: list[int] = [128, 256, 512],
    ) -> None:
        super().__init__()
        # keep track of input + hidden dims
        self.dims = [in_chan] + dims

        # number of downsampling stages = one PatchEmbed per block
        self.num_downsamples = len(depths)

        # build layers
        self.layers = nn.ModuleList([])
        for i, depth in enumerate(depths):
            # each block starts with a stride-2 PatchEmbed
            self.layers.append(PatchEmbed(self.dims[i], self.dims[i + 1]))
            # followed by `depth` Conv modules
            for _ in range(depth):
                self.layers.append(Conv(self.dims[i + 1]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, channels, seq_len)
        for layer in self.layers:
            x = layer(x)
        # transpose to (batch, seq_len, features)
        return x.transpose(1, 2)

    @property
    def size_out(self) -> int:
        '''Number of output features per time-step.'''
        return self.dims[-1]

    def calculate_output_length(self, input_lengths: torch.Tensor) -> torch.Tensor:
        '''Compute the downsampled time-length after all PatchEmbeds.

        We downsample by factor=2 at each of the num_downsamples PatchEmbed stages,
        so total factor = 2**num_downsamples. We floor-divide the original lengths
        by that factor.
        '''
        factor = 2 ** self.num_downsamples
        # Floor divide and keep same dtype (e.g. torch.int64)
        return (input_lengths // factor).to(input_lengths.dtype)
