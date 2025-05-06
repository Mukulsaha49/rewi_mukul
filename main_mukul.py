import argparse, os, json, yaml
import torch
import torch.nn as nn
from torch.amp import GradScaler
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from torch.utils.data import DataLoader

from hwr.dataset      import HRDataset
from hwr.dataset.utils import fn_collate
from hwr.model        import get_model
from hwr.loss         import CTCLoss
from hwr.ctc_decoder  import build_ctc_decoder
from hwr.manager      import RunManager
from hwr.utils        import seed_everything, seed_worker
from hwr.evaluate     import evaluate


def train_one_epoch(dataloader, model, fn_loss, optimizer, scaler,
                    lr_scheduler, manager, device, epoch):
    manager.initialize_epoch(epoch, len(dataloader), val=False)
    model.train()
    for i, (x, y, len_x, len_y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)
        len_out = model.calculate_output_length(len_x)

        optimizer.zero_grad()
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            out  = model(x)
            loss = fn_loss(out.permute(1, 0, 2), y, len_out, len_y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        lr_scheduler.step()
        manager.update_iteration(i, loss.item(), lr_scheduler.get_last_lr()[0])

    manager.summarize_epoch()


def test(dataloader, model, fn_loss, manager, ctc_decoder, device, epoch):
    manager.initialize_epoch(epoch, len(dataloader), val=True)
    model.eval()
    preds, labels = [], []

    with torch.no_grad():
        for idx, (x, y, len_x, len_y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)
            len_out = model.calculate_output_length(len_x)

            out  = model(x)
            loss = fn_loss(out.permute(1, 0, 2), y, len_out, len_y)
            manager.update_iteration(idx, loss.item())

            if manager.check_step(idx + 1, manager.freq_eval, manager.num_iter):
                for p_logits, lo, lab in zip(out.cpu(), len_out, y.cpu()):
                    preds.append(ctc_decoder.decode(p_logits[:lo]))
                    labels.append(ctc_decoder.decode(lab.tolist(), label=True))

    manager.summarize_epoch()
    results = evaluate(preds, labels)
    manager.update_evaluation(results, preds[:20], labels[:20])


def main(cfg):
    # — load token vocab & sync
    vocab_path = os.path.join(cfg.dir_dataset, "token_vocab.json")
    with open(vocab_path, "r", encoding="utf-8") as vf:
        token_to_idx = json.load(vf)
    cfg.categories = list(token_to_idx.keys())
    cfg.num_cls    = len(cfg.categories)

    # — init everything
    manager     = RunManager(cfg)
    seed_everything(cfg.seed)
    decoder     = build_ctc_decoder(vocab_path, cfg.ctc_decoder)
    fn_loss     = CTCLoss(blank=cfg.categories.index("<BLANK>"))

    # only build train loader if not test-only
    if not cfg.test:
        train_ds = HRDataset(
            path_anno   = os.path.join(cfg.dir_dataset, "train.json"),
            categories  = cfg.categories,
            sensors     = cfg.sensors,
            ratio_ds    = cfg.ratio_ds,
            idx_cv      = cfg.idx_cv,
            size_window = cfg.size_window,
            aug         = cfg.aug,
            len_seq     = cfg.len_seq,
            cache       = cfg.cache,
        )
        train_loader = DataLoader(
            train_ds,
            batch_size     = cfg.size_batch,
            shuffle        = True,
            num_workers    = cfg.num_worker,
            collate_fn     = fn_collate,
            worker_init_fn = seed_worker,
            generator      = torch.Generator().manual_seed(cfg.seed),
        )

    # always build val loader
    val_ds = HRDataset(
        path_anno   = os.path.join(cfg.dir_dataset, "val.json"),
        categories  = cfg.categories,
        sensors     = cfg.sensors,
        ratio_ds    = cfg.ratio_ds,
        idx_cv      = cfg.idx_cv,
        size_window = cfg.size_window,
        aug         = False,
        len_seq     = cfg.len_seq,
        cache       = cfg.cache,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size  = cfg.size_batch,
        num_workers = cfg.num_worker,
        collate_fn  = fn_collate,
    )

    # model + optimizer + scheduler + resume
    model = get_model(
        arch_en  = cfg.arch_en,
        arch_de  = cfg.arch_de,
        in_chan  = cfg.in_chan,
        num_cls  = cfg.num_cls,
        ratio_ds = cfg.ratio_ds,
        len_seq  = cfg.len_seq,
    ).to(cfg.device)

    optimizer   = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    scaler      = GradScaler()
    total_steps = len(train_loader) * cfg.epoch if not cfg.test else 1
    warmup_steps= len(train_loader) * cfg.epoch_warmup if not cfg.test else 0
    lr_scheduler= SequentialLR(
        optimizer,
        [
            LinearLR(optimizer, start_factor=0.01, total_iters=warmup_steps),
            CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps),
        ],
        milestones=[warmup_steps],
    )

    epoch_start = 0
    if cfg.checkpoint:
        ckpt = torch.load(cfg.checkpoint, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        lr_scheduler.load_state_dict(ckpt["lr_scheduler"])
        epoch_start = ckpt["epoch"] + 1
        manager.log(f"Resumed from {cfg.checkpoint} at epoch {epoch_start}")

    # test‐only shortcut
    if cfg.test:
        test(val_loader, model, fn_loss, manager, decoder, cfg.device, epoch=-1)
        return

    # training loop with interim train‐set eval
    for epoch in range(epoch_start, cfg.epoch):
        train_one_epoch(train_loader, model, fn_loss,
                        optimizer, scaler, lr_scheduler,
                        manager, cfg.device, epoch)
        if (epoch + 1) % cfg.freq_eval == 0 or (epoch + 1) == cfg.epoch:
            test(train_loader, model, fn_loss,
                 manager, decoder, cfg.device, epoch)

    # final held‐out
    test(val_loader, model, fn_loss, manager, decoder, cfg.device, epoch=-1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train/evaluate bigram CTC model"
    )
    parser.add_argument(
        "-c", "--config", required=True,
        help="Path to train_mukul.yaml or test_mukul.yaml"
    )
    args = parser.parse_args()

    with open(args.config, "r") as yf:
        cfg_dict = yaml.safe_load(yf)
    cfg = argparse.Namespace(**cfg_dict)
    cfg.device = "cuda" if torch.cuda.is_available() else "cpu"

    main(cfg)
