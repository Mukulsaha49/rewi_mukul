import torch
import torch.nn as nn


def smooth_probs(probs: torch.Tensor, alpha: float = 1e-6) -> torch.Tensor:
    '''Smooth a probability distribution for stable CTC training.'''
    num_cls = probs.shape[-1]
    uniform = torch.full_like(probs, 1.0 / num_cls)
    probs = (1 - alpha) * probs + alpha * uniform
    probs = probs / probs.sum(dim=-1, keepdim=True)
    return probs


class CTCLoss(nn.Module):
    """
    CTC loss wrapper that smooths probs and
    falls back to CPU if the GPU kernel’s 81-frame limit is exceeded.
    """
    def __init__(
        self,
        alpha_smooth: float = 1e-6,
        blank: int = 0,
        reduction: str = 'mean',
        zero_infinity: bool = False,
    ) -> None:
        super().__init__()
        self.alpha_smooth = alpha_smooth
        self.blank = blank
        self.reduction = reduction
        self.zero_infinity = zero_infinity

    def forward(
        self,
        probs: torch.Tensor,
        targets: torch.Tensor,
        input_lengths: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> torch.Tensor:
        # 1) optionally smooth
        if self.alpha_smooth:
            probs = smooth_probs(probs, self.alpha_smooth)

        log_probs = probs.log()
        max_len = int(input_lengths.max().item())

        # 2) if on CUDA and too long for the 81-frame CTC kernel, run on CPU
        if log_probs.device.type == 'cuda' and max_len > 81:
            device = log_probs.device
            lp = log_probs.cpu()
            tgt = targets.cpu()
            il = input_lengths.cpu()
            tl = target_lengths.cpu()
            loss_cpu = nn.functional.ctc_loss(
                lp, tgt, il, tl,
                blank=self.blank,
                reduction=self.reduction,
                zero_infinity=self.zero_infinity,
            )
            return loss_cpu.to(device)

        # 3) otherwise run normally (GPU or CPU)
        return nn.functional.ctc_loss(
            log_probs,
            targets,
            input_lengths,
            target_lengths,
            blank=self.blank,
            reduction=self.reduction,
            zero_infinity=self.zero_infinity,
        )
