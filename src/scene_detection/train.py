# training.py
import argparse
import logging
import time

import torch
from torch import nn

import timm
from data import build_loaders


def setup_logging(level: str = "INFO"):
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=(scaler is not None)):
            logits = model(x)
            loss = criterion(logits, y)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * x.size(0)
        correct += (logits.argmax(dim=1) == y).sum().item()
        total += x.size(0)

    return running_loss / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        loss = criterion(logits, y)

        running_loss += loss.item() * x.size(0)
        correct += (logits.argmax(dim=1) == y).sum().item()
        total += x.size(0)

    return running_loss / max(total, 1), correct / max(total, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default=None, help="ImageFolder root (contains class folders)")
    parser.add_argument("--model-name", type=str, default="mobilenetv3_large_100")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-path", type=str, default="best_scene_model.pth")
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision even on CUDA")
    parser.add_argument("--log-level", type=str, default="INFO", help="DEBUG, INFO, WARNING, ERROR")
    args = parser.parse_args()

    setup_logging(args.log_level)
    log = logging.getLogger("training")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (device.type == "cuda") and (not args.no_amp)

    log.info("Device=%s | AMP=%s", device, use_amp)
    log.info("Model=%s | epochs=%d | batch_size=%d | lr=%g | wd=%g",
             args.model_name, args.epochs, args.batch_size, args.lr, args.weight_decay)

    # --- Data ---
    train_loader, val_loader, test_loader, id_map, data_cfg = build_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        model_name=args.model_name,
        keep_classes=("ships", "bridges", "railways"),
        train_ratio=0.8,
        val_ratio=0.1,
    )

    log.info("id_map=%s", id_map)
    log.info("data_cfg=%s", data_cfg)
    log.info("Batches | train=%d val=%d test=%d",
             len(train_loader), len(val_loader), len(test_loader))

    # --- Model ---
    model = timm.create_model(args.model_name, pretrained=True, num_classes=3).to(device)

    # --- Loss / Optim / Sched ---
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp) if use_amp else None

    best_val_acc = -1.0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        log.info(
            "Epoch %02d/%02d | train loss %.4f acc %.3f | val loss %.4f acc %.3f | %.1fs",
            epoch, args.epochs, train_loss, train_acc, val_loss, val_acc, time.time() - t0
        )

        # save best checkpoint by validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_name": args.model_name,
                    "num_classes": 3,
                    "id_map": id_map,
                    "data_cfg": data_cfg,
                },
                args.save_path,
            )
            log.info("Saved new best checkpoint to %s (val_acc=%.3f)", args.save_path, best_val_acc)

    log.info("Best val acc: %.3f (saved to %s)", best_val_acc, args.save_path)

    # --- Final test (load best and evaluate once) ---
    ckpt = torch.load(args.save_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    log.info("TEST | loss %.4f acc %.3f", test_loss, test_acc)


if __name__ == "__main__":
    main()
