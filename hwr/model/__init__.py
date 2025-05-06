import torch
import torch.nn as nn
from .conv import BLCNN
from .lstm import LSTM


class BaseModel(nn.Module):
    """Modular encoder-decoder model wrapper."""
    def __init__(self, encoder: nn.Module, decoder: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)  # shape: [B, C, T] → encoder → [B, T, D]
        x = self.decoder(x)  # decoder → [B, T, num_classes]
        return x

    def fuse(self):
        if hasattr(self.encoder, "fuse"):
            self.encoder.fuse()
        if hasattr(self.decoder, "fuse"):
            self.decoder.fuse()

    def calculate_output_length(self, input_lengths: torch.Tensor) -> torch.Tensor:
        """Optionally compute output sequence lengths (for CTC, etc.)"""
        if hasattr(self.encoder, "calculate_output_length"):
            return self.encoder.calculate_output_length(input_lengths)
        return input_lengths  # fallback


def get_model(
    arch_en: str,
    arch_de: str,
    in_chan: int,
    num_cls: int,
    ratio_ds: int = 8,
    len_seq: int = 0,
    **kwargs
) -> nn.Module:
    # === Encoder selection ===
    if arch_en == "blcnn":
        encoder = BLCNN(in_chan)
    else:
        raise ValueError(f"Unknown encoder type: {arch_en}")

    # === Decoder selection ===
    if arch_de == "lstm":
        decoder = LSTM(encoder.size_out, num_cls)
    else:
        raise ValueError(f"Unknown decoder type: {arch_de}")

    return BaseModel(encoder, decoder)
