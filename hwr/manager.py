import argparse
import json
import os
import time
from datetime import datetime
from typing import Any

import torch
import yaml
from loguru import logger

from .utils import sec2time


class RunManager:
    '''Manage work directory, logging, losses, evaluation results and checkpoints.'''

    def __init__(self, cfgs: argparse.Namespace) -> None:
        self.dir_work   = cfgs.dir_work
        self.epoch_max  = cfgs.epoch
        self.freq_eval  = cfgs.freq_eval
        self.freq_log   = cfgs.freq_log
        self.freq_save  = cfgs.freq_save
        self.test       = cfgs.test

        self.results = {}
        self.ts      = datetime.now().strftime('%Y%m%d%H%M%S')

        self.initialize_directory(cfgs)

    def check_step(self, scur: int, freq: int, smax: int) -> bool:
        return scur % freq == 0 or scur == smax

    def initialize_directory(self, cfgs: argparse.Namespace) -> None:
        tag        = 'test' if self.test else 'train'
        idx_cv     = str(cfgs.idx_cv)
        self.dir_ckp   = os.path.join(cfgs.dir_work, idx_cv, 'checkpoints')
        path_cfg   = os.path.join(cfgs.dir_work, idx_cv, f'{tag}_{self.ts}.yaml')
        path_log   = os.path.join(cfgs.dir_work, idx_cv, f'{tag}_{self.ts}.log')
        self.path_result = os.path.join(cfgs.dir_work, idx_cv, f'{tag}_{self.ts}.json')

        os.makedirs(self.dir_ckp, exist_ok=True)
        os.makedirs(os.path.dirname(path_cfg), exist_ok=True)

        with open(path_cfg, 'w') as f:
            yaml.safe_dump(vars(cfgs), f)

        logger.add(path_log)
        logger.info(f'Initialized work directory at {cfgs.dir_work} for fold {idx_cv}.')

    def initialize_epoch(self, epoch: int, num_iter: int, val: bool) -> None:
        self.epoch    = epoch
        self.loss     = []
        self.num_iter = num_iter
        self.t_start  = time.time()
        self.tag      = 'test' if val else 'train'

        if epoch not in self.results:
            self.results[epoch] = {}
        self.results[epoch][self.tag] = []

    def log(self, message: str) -> None:
        logger.info(message)

    def save_checkpoint(
        self,
        state_model: dict = None,
        state_optimizer: dict = None,
        state_lr_scheduler: dict = None,
    ) -> None:
        torch.save(
            {
                'epoch': self.epoch,
                'model': state_model,
                'optimizer': state_optimizer,
                'lr_scheduler': state_lr_scheduler,
            },
            os.path.join(self.dir_ckp, f'{self.epoch}.pth'),
        )
        logger.info(f'Saved checkpoint of epoch {self.epoch}')

    def save_results(self) -> None:
        with open(self.path_result, 'w') as f:
            json.dump(self.results, f)

    def summarize_evaluation(self) -> None:
        results_eval = [
            [epoch, result['evaluation']]
            for epoch, result in self.results.items()
            if 'evaluation' in result
        ]
        if not results_eval:
            return

        metrics = results_eval[0][1].keys()
        best = {m: [-1, float('inf')] for m in metrics}
        for epoch, eval_res in results_eval:
            for m in metrics:
                if eval_res[m] < best[m][1]:
                    best[m] = [epoch, eval_res[m]]

        self.results['best'] = best
        logger.info(f'best: {best}')
        self.save_results()

    def summarize_epoch(self) -> float | None:
        t_end = time.time() - self.t_start

        if self.loss:
            loss_avg = sum(self.loss) / len(self.loss)
            logger.info(
                f'{self.tag}, epoch: {self.epoch}, '
                f'loss avg: {loss_avg:.7f}, time: {sec2time(t_end)}'
            )
            entry = {'loss_avg': loss_avg, 'time': t_end}
        else:
            logger.info(f'{self.tag}, epoch: {self.epoch}, time: {sec2time(t_end)}')
            entry = {'time': t_end}
            loss_avg = None

        self.results[self.epoch][self.tag].append(entry)
        self.save_results()

        return loss_avg

    def update_evaluation(
        self, result: dict, preds: Any = None, labels: Any = None
    ) -> None:
        self.results[self.epoch]['evaluation'] = result
        self.save_results()
        msg = ', '.join(f'{k}: {v:.7f}' for k, v in result.items())
        logger.info(msg)
        if preds:
            logger.info(f'predictions: {preds}')
        if labels:
            logger.info(f'labels: {labels}')

    def update_iteration(
        self,
        iter: int,
        loss: float,
        lr: float = -1,
    ) -> None:
        self.loss.append(loss)
        if self.check_step(iter + 1, self.freq_log, self.num_iter):
            t_inter = time.time() - self.t_start
            result = {
                'lr': lr,
                'iters': iter + 1,
                'loss': loss,
                'time': t_inter,
            }
            self.results[self.epoch][self.tag].append(result)
            logger.info(
                f'{self.tag}, epoch: {self.epoch}, '
                f'iters: {iter+1}/{self.num_iter}, '
                f'lr: {lr:.7f}, loss: {loss:.7f}, '
                f'time: {sec2time(t_inter)}'
            )
