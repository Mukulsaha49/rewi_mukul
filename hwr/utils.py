# hwr/utils.py

import os
import random
import time
import yaml

import numpy as np
import torch


def seed_everything(seed: int = 42) -> None:
    """Seed everything for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def seed_worker(worker_id: int) -> None:
    """Seed DataLoader workers."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def sec2time(time_sec: float) -> str:
    """Convert elapsed seconds into H:MM:SS string."""
    s = str(int(time_sec % 60)).zfill(2)
    m = str(int((time_sec // 60) % 60)).zfill(2)
    h = int(time_sec // 3600)
    return f"{h}:{m}:{s}"


def stitch_windows(partials: list[str], overlap: int) -> str:
    """
    Merge overlapping decoded window‐predictions into one full transcript.
    
    Args:
        partials: list of decoded substrings, each from one window
        overlap:  maximum number of tokens any two adjacent windows share

    Returns:
        A single, merged string.
    """
    if not partials:
        return ""

    full = partials[0]
    for part in partials[1:]:
        max_olap = min(len(full), len(part), overlap)
        cut = 0
        # find the longest suffix of `full` that matches a prefix of `part`
        for k in range(max_olap, 0, -1):
            if full.endswith(part[:k]):
                cut = k
                break
        full += part[cut:]
    return full
