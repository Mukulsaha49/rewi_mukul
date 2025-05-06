import torch
from typing import Any
from hwr.tokenizer import Tokenizer


class BestPathCTCDecoder:
    """
    Greedy CTC decoder using best path decoding.
    Designed to work with bigram tokenization and Tokenizer class.
    """
    def __init__(self, vocab_path: str):
        """
        Args:
            vocab_path (str): Path to token_vocab.json used to initialize Tokenizer
        """
        self.tokenizer = Tokenizer(vocab_path)

    def decode(self, seq: torch.Tensor, label: bool = False) -> str:
        """
        Decodes a sequence of logits or indices into a string.

        Args:
            seq (torch.Tensor): If prediction, shape [T, C] with logits;
                                If label, shape [T] with token indices
            label (bool): Whether input is a label (True) or model output (False)

        Returns:
            str: Decoded output string
        """
        if not label:
            # Prediction: logits → best path (argmax) → collapse repeats
            seq = torch.argmax(seq, dim=-1)
            seq = torch.unique_consecutive(seq, dim=-1)

        seq = seq.tolist() if isinstance(seq, torch.Tensor) else seq
        return self.tokenizer.decode(seq, remove_duplicates=False, ignore_blank=True)


def build_ctc_decoder(vocab_path: str, decoder_type: str = "best_path") -> Any:
    """
    Factory to build the CTC decoder.

    Args:
        vocab_path (str): Path to token_vocab.json
        decoder_type (str): Currently only supports 'best_path'

    Returns:
        Instance of a CTC decoder
    """
    match decoder_type:
        case "best_path":
            return BestPathCTCDecoder(vocab_path)
        case _:
            raise ValueError(f"Unknown decoder type: {decoder_type}")
