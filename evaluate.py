import argparse
import json
import os
from glob import glob

import numpy as np
import torch
import yaml
from thop import profile

from hwr.model import get_model


def get_mean_std_cv(cfgs: dict, results: dict = {}) -> dict:
    '''Calculate mean/std of CER and WER across CV folds.'''
    cer, wer = {}, {}

    # collect all fold result files
    pattern = 'test_*.json' if cfgs['test'] else 'train_*.json'
    paths = glob(os.path.join(cfgs['dir_work'], '*', pattern))
    if not paths:
        return results

    for i, path in enumerate(sorted(paths)):
        with open(path, 'r') as f:
            result_fd = json.load(f)
        if cfgs['test']:
            best = result_fd.get('-1', {}).get('evaluation', {})
        else:
            # on train mode, fold JSON has 'best' epoch key
            best_epoch = result_fd.get('best', {}).get('character_error_rate', [None])[0]
            best = result_fd.get(str(best_epoch), {}).get('evaluation', {})
        cer[str(i)] = best.get('character_error_rate', -1)
        wer[str(i)] = best.get('word_error_rate', -1)

    results['cer'] = {
        'raw': cer,
        'mean': float(np.mean(list(cer.values()))),
        'std': float(np.std(list(cer.values()))),
    }
    results['wer'] = {
        'raw': wer,
        'mean': float(np.mean(list(wer.values()))),
        'std': float(np.std(list(wer.values()))),
    }
    return results


def get_macs_params(cfgs: dict, results: dict = {}) -> dict:
    '''Compute MACs and parameter count for the model.'''
    model = get_model(
        cfgs['arch_en'],
        cfgs['arch_de'],
        cfgs['in_chan'],
        cfgs['num_cls'],
        cfgs['ratio_ds'],
        cfgs['len_seq'],
    ).eval()
    # dummy input: batch=1, channels=in_chan, length=1024
    x = torch.randn(1, cfgs['in_chan'], 1024)
    macs, params = profile(model, inputs=(x,), verbose=False)
    results['macs'] = int(macs)
    results['params'] = int(params)
    return results


def main(path_cfg: str) -> None:
    '''Evaluate and summarize the results of all cross-validation folds.'''
    # Load config
    with open(path_cfg, 'r') as f:
        cfgs = yaml.safe_load(f)

    # Defaults if missing
    cfgs.setdefault('ratio_ds', 8)
    cfgs.setdefault('len_seq', 0)

    # Ensure work directory exists
    os.makedirs(cfgs['dir_work'], exist_ok=True)

    # Load existing results or start fresh
    results_path = os.path.join(cfgs['dir_work'], 'results.json')
    if os.path.isfile(results_path):
        with open(results_path, 'r') as f:
            results = json.load(f)
    else:
        results = {}

    # Compute metrics
    results = get_mean_std_cv(cfgs, results)
    results = get_macs_params(cfgs, results)

    # Save and print
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(json.dumps(results, indent=4))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluate handwriting recognition model.'
    )
    parser.add_argument(
        '-c', '--config',
        default='configs/train_mukul.yaml',
        help='Path to configuration YAML (default: configs/train_mukul.yaml)'
    )
    args = parser.parse_args()
    main(args.config)
