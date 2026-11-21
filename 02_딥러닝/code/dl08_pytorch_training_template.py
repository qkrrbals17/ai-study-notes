"""
[실습 08] 재사용 가능한 PyTorch 학습 템플릿
- 시드 고정, AMP(bf16), gradient clipping, gradient accumulation, warmup+cosine, early stopping, 체크포인트
- '한 배치 과적합' 디버그 모드 (--overfit_one_batch)
※ 실행 환경: pip install torch torchvision    예) python dl08_pytorch_training_template.py --epochs 10
"""
import argparse, math, os, random, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=10); p.add_argument("--bs", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-3); p.add_argument("--wd", type=float, default=0.05)
    p.add_argument("--accum", type=int, default=1); p.add_argument("--patience", type=int, default=3)
    p.add_argument("--overfit_one_batch", action="store_true"); p.add_argument("--seed", type=int, default=42)
    return p.parse_args()

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.f = nn.Sequential(nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
                               nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
                               nn.Flatten(), nn.Dropout(0.25), nn.Linear(64 * 7 * 7, 128), nn.ReLU(), nn.Linear(128, 10))
    def forward(self, x): return self.f(x)

@torch.no_grad()
def evaluate(model, dl, device):
    model.eval(); tot, correct, n = 0.0, 0, 0
    for x, y in dl:
        x, y = x.to(device), y.to(device)
        out = model(x); tot += F.cross_entropy(out, y, reduction="sum").item()
        correct += (out.argmax(1) == y).sum().item(); n += len(y)
    return tot / n, correct / n

def main():
    a = get_args(); seed_all(a.seed)
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    full = datasets.MNIST("./data", train=True, download=True, transform=tf)
    test = datasets.MNIST("./data", train=False, download=True, transform=tf)
    train_ds, val_ds = random_split(full, [55000, 5000], generator=torch.Generator().manual_seed(a.seed))
    train_dl = DataLoader(train_ds, a.bs, shuffle=True, num_workers=2, pin_memory=True)
    val_dl, test_dl = DataLoader(val_ds, 512), DataLoader(test, 512)

    model = Net().to(device)
    print(f"device={device}, params={sum(p.numel() for p in model.parameters()):,}")
    decay = [p for n, p in model.named_parameters() if p.ndim >= 2]           # 가중치만 weight decay
    no_decay = [p for n, p in model.named_parameters() if p.ndim < 2]         # bias, BN 제외
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": a.wd}, {"params": no_decay, "weight_decay": 0}], lr=a.lr)
    total = a.epochs * math.ceil(len(train_dl) / a.accum); warm = int(0.05 * total)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / (total - warm))))
    amp_dtype = torch.bfloat16 if device != "cpu" else torch.float32

    if a.overfit_one_batch:                       # 디버그: 손실이 0 근처로 가야 정상
        x, y = next(iter(train_dl)); x, y = x.to(device), y.to(device)
        for i in range(300):
            loss = F.cross_entropy(model(x), y); opt.zero_grad(); loss.backward(); opt.step()
            if i % 50 == 0: print(f"[overfit] step {i} loss {loss.item():.4f}")
        return

    best, bad, step = float("inf"), 0, 0
    for ep in range(a.epochs):
        model.train(); t0 = time.time(); run = 0.0
        for i, (x, y) in enumerate(train_dl):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type=device if device != "mps" else "cpu", dtype=amp_dtype, enabled=device == "cuda"):
                loss = F.cross_entropy(model(x), y, label_smoothing=0.05) / a.accum
            loss.backward(); run += loss.item() * a.accum
            if (i + 1) % a.accum == 0:
                gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); opt.zero_grad(set_to_none=True); sched.step(); step += 1
        vl, va = evaluate(model, val_dl, device)
        print(f"ep {ep+1:2d} | train {run/len(train_dl):.4f} | val {vl:.4f} acc {va:.4f} | lr {sched.get_last_lr()[0]:.2e} | grad {gn:.2f} | {time.time()-t0:.1f}s")
        if vl < best:
            best, bad = vl, 0; torch.save({"model": model.state_dict(), "epoch": ep, "val_loss": vl}, "best.pt")
        else:
            bad += 1
            if bad >= a.patience: print("early stopping"); break
    model.load_state_dict(torch.load("best.pt")["model"])
    print("TEST loss/acc:", evaluate(model, test_dl, device))

if __name__ == "__main__":
    main()
