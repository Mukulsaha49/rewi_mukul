import torch
from torch.nn.utils.rnn import pad_sequence


def fn_collate(batch: list[tuple[torch.Tensor | list[torch.Tensor], torch.Tensor]]):
    """
    Flattens per-sample window lists into one batch of windows,
    pads sequences & labels, and returns lengths.
    """
    seqs, labels, lens_x, lens_y = [], [], [], []

    for x, y in batch:
        if isinstance(x, list):
            for win in x:
                seqs.append(win)
                labels.append(y)
                lens_x.append(win.size(0))
                lens_y.append(y.size(0))
        else:
            seqs.append(x)
            labels.append(y)
            lens_x.append(x.size(0))
            lens_y.append(y.size(0))

    # pad sequences to same time-length
    seqs_padded = pad_sequence(seqs, batch_first=True)      # (B_all, T_max, C)
    seqs_padded = seqs_padded.permute(0, 2, 1).contiguous()  # (B_all, C, T_max)

    # pad labels to same token-length
    labels_padded = pad_sequence(labels, batch_first=True)   # (B_all, L_max)

    return (
        seqs_padded,
        labels_padded,
        torch.tensor(lens_x, dtype=torch.long),
        torch.tensor(lens_y, dtype=torch.long),
    )
