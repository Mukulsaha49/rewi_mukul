import os
import json

import numpy as np
import torch
from loguru import logger
from torch.nn.functional import pad
from torch.utils.data import Dataset
from tqdm import tqdm

from .transforms import AddNoise, Drift, Dropout, TimeWarp
from hwr.tokenizer import Tokenizer

SENSOR = {
    'AF': [0, 1, 2],    # front accelerometer
    'AR': [3, 4, 5],    # rear accelerometer
    'G':  [6, 7, 8],    # gyroscope
    'M':  [9, 10, 11],  # magnetometer
    'F':  [12],         # force sensor
}


class HRDataset(Dataset):
    """
    Dataset for bigram‐tokenized handwriting signals.
    Keeps the full raw sequence and relies on CTC’s CPU fallback for very long inputs.
    """

    def __init__(
        self,
        path_anno: str,
        categories: list[str],
        sensors: list[str],
        ratio_ds: int,
        idx_cv: str | int,
        size_window: int = 1,
        aug: bool = False,
        len_seq: int = 0,
        cache: bool = False,
    ) -> None:
        self.dir_ds = os.path.dirname(path_anno)
        self.categories = categories
        self.ratio_ds = ratio_ds
        self.size_window = size_window
        self.len_seq = len_seq
        self.cache = cache

        # build augmentation list
        self.augs = (
            [
                AddNoise(scale=0.05, kind='multiplicative'),
                Drift(0.1, 40, 'multiplicative'),
                Dropout(size=(5, 10), per_channel=True),
                TimeWarp(5, 4),
            ]
            if aug else None
        )

        # pick sensor channels
        self.idx_channel = []
        for s in sensors:
            self.idx_channel.extend(SENSOR[s])

        # load annotations for this fold
        with open(path_anno, 'r', encoding='utf-8') as f:
            full = json.load(f)
        self.annos = full['annotations'][str(idx_cv)]

        # load our bigram Tokenizer (with <BLANK> and <UNK>)
        vocab_path = os.path.join(self.dir_ds, 'token_vocab.json')
        self.tokenizer = Tokenizer(vocab_path)

        # optional in-RAM cache
        if cache:
            self.data_cache = [
                (
                    np.loadtxt(os.path.join(self.dir_ds, a['filename']), delimiter=';', dtype=np.float32),
                    a['label'],
                )
                for a in tqdm(self.annos, desc='Caching dataset')
            ]
            logger.info(f'✅ Cached dataset: {path_anno}')

    def __len__(self) -> int:
        return len(self.annos)

    def __getitem__(self, idx: int):
        # 1) load raw signal & text label
        if self.cache:
            seq_raw, text = self.data_cache[idx]
        else:
            anno = self.annos[idx]
            seq_raw = np.loadtxt(
                os.path.join(self.dir_ds, anno['filename']),
                delimiter=';',
                dtype=np.float32,
            )
            text = anno['label']

        # 2) bigram‐encode the label
        label_indices = self.tokenizer.encode(text)
        label = torch.tensor(label_indices, dtype=torch.int32)

        # 3) select sensor channels
        seq = seq_raw[:, self.idx_channel]

        # 4) normalize / augment / pad (no forced cropping!)
        seq = self.process(seq, len(label_indices))

        return seq, label

    def process(self, seq: np.ndarray, len_label: int) -> torch.Tensor:
        # a) optional augment
        if self.augs is not None:
            for a in self.augs:
                if np.random.rand() < 0.25:
                    seq = a(seq)

        # b) channel-wise normalization
        seq = (seq - seq.mean(0)) / (seq.std(0) + 1e-6)
        seq = torch.from_numpy(seq).float()

        # c) padding / truncation
        if self.len_seq > 0:
            # fixed-length mode: truncate then pad
            if len(seq) > self.len_seq:
                seq = seq[:self.len_seq]
            else:
                pad_amt = self.len_seq - len(seq)
                seq = pad(seq.T, (0, pad_amt)).T
        else:
            # dynamic CTC-safe padding:
            # ensure at least 2×label_length × downsampling_factor
            min_len = len_label * 2 * self.ratio_ds
            if len(seq) < min_len:
                pad_amt = min_len - len(seq)
                seq = pad(seq.T, (0, pad_amt)).T

            # make final length divisible by ratio_ds * size_window
            rem = len(seq) % (self.ratio_ds * self.size_window)
            if rem:
                pad_amt = (self.ratio_ds * self.size_window) - rem
                seq = pad(seq.T, (0, pad_amt)).T

        return seq
