import jiwer
import Levenshtein
import numpy as np


def get_levenshtein_distance(
    preds: list[str], labels: list[str]
) -> tuple[float, float]:
    """
    Calculate average Levenshtein distance between predicted and ground truth strings.

    Args:
        preds (list[str]): List of decoded predicted strings.
        labels (list[str]): List of decoded ground-truth strings.

    Returns:
        tuple[float, float]: (Average Levenshtein distance, Average label length)
    """
    dist_leven = []
    len_label_avg = []

    for pred, label in zip(preds, labels):
        dist = Levenshtein.distance(pred, label)
        dist_leven.append(dist)
        len_label_avg.append(len(label))

    return np.mean(dist_leven), np.mean(len_label_avg)


def evaluate(
    preds: list[str] | str,
    labels: list[str] | str,
    use_ld: bool = True,
    use_cer: bool = True,
    use_wer: bool = True,
) -> dict:
    """
    Evaluate model predictions using Levenshtein distance, CER, and WER.

    Args:
        preds (list[str] | str): List of decoded predicted strings or a single string.
        labels (list[str] | str): List of decoded ground-truth strings or a single string.
        use_ld (bool): Include Levenshtein distance. Defaults to True.
        use_cer (bool): Include Character Error Rate. Defaults to True.
        use_wer (bool): Include Word Error Rate. Defaults to True.

    Returns:
        dict: Evaluation metrics.
    """
    if isinstance(preds, str):
        preds = [preds]
    if isinstance(labels, str):
        labels = [labels]

    dist_leven, len_sent_avg = (-1, -1)
    if use_ld:
        dist_leven, len_sent_avg = get_levenshtein_distance(preds, labels)

    cer = jiwer.cer(labels, preds) if use_cer else -1
    wer = jiwer.wer(labels, preds) if use_wer else -1

    return {
        "levenshtein_distance": dist_leven,
        "average_sentence_length": len_sent_avg,
        "character_error_rate": cer,
        "word_error_rate": wer,
    }
