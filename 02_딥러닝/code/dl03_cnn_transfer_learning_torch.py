"""
[실습 03-b] PyTorch: CIFAR-10 CNN 직접 학습 + ResNet18 전이학습 비교
※ 실행 환경: pip install torch torchvision  (GPU 권장, CPU면 epoch을 줄이세요)
"""
import torch, torch.nn as nn, torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
mean, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
train_tf = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
                               transforms.ToTensor(), transforms.Normalize(mean, std)])
test_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
train_ds = datasets.CIFAR10("./data", train=True, download=True, transform=train_tf)
test_ds = datasets.CIFAR10("./data", train=False, download=True, transform=test_tf)
train_dl = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=2)
test_dl = DataLoader(test_ds, batch_size=256, num_workers=2)

class SmallCNN(nn.Module):
    """conv-bn-relu x2 → pool 블록을 3번 + GAP + FC"""
    def __init__(self, n_cls=10):
        super().__init__()
        def block(cin, cout):
            return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(),
                                 nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(),
                                 nn.MaxPool2d(2))
        self.features = nn.Sequential(block(3, 64), block(64, 128), block(128, 256))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(256, n_cls))
    def forward(self, x): return self.head(self.features(x))

def run(model, epochs, lr):
    model.to(device)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * len(train_dl))
    for ep in range(epochs):
        model.train(); tot = 0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            loss = F.cross_entropy(model(x), y, label_smoothing=0.1)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step(); tot += loss.item()
        model.eval(); correct = 0
        with torch.no_grad():
            for x, y in test_dl:
                correct += (model(x.to(device)).argmax(1).cpu() == y).sum().item()
        print(f"epoch {ep+1}: loss={tot/len(train_dl):.3f}  test acc={correct/len(test_ds):.4f}")

print("== 1) 처음부터 학습하는 SmallCNN ==")
m = SmallCNN(); print("파라미터 수:", sum(p.numel() for p in m.parameters()))
run(m, epochs=15, lr=0.1)

print("== 2) ImageNet 사전학습 ResNet18 파인튜닝 ==")
r = models.resnet18(weights="IMAGENET1K_V1")
r.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)   # 32x32 작은 이미지용으로 첫 층 수정
r.maxpool = nn.Identity()
r.fc = nn.Linear(r.fc.in_features, 10)
run(r, epochs=5, lr=0.05)
