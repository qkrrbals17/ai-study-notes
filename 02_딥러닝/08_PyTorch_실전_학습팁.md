# PyTorch 실전 & 학습 팁

## 1. 표준 학습 루프
```python
import torch
from torch.utils.data import DataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-4,
            total_steps=EPOCHS * len(train_loader))
scaler = torch.amp.GradScaler()          # 혼합정밀도(FP16일 때)

for epoch in range(EPOCHS):
    model.train()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            loss = loss_fn(model(x), y)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update(); sched.step()

    model.eval()
    with torch.no_grad():
        val_loss = sum(loss_fn(model(x.to(device)), y.to(device)).item()
                       for x, y in val_loader) / len(val_loader)
    print(epoch, val_loss)
    # best 모델 저장 / early stopping
```

## 2. 디버깅 순서 (Karpathy 레시피 요약)
1. **데이터를 직접 눈으로 본다** (이미지/라벨/토큰)
2. 초기 손실이 이론값인지 확인 (C 클래스면 ≈ ln C)
3. **작은 배치 하나에 과적합**시켜 본다 → 안 되면 버그
4. 단순한 베이스라인부터, 한 번에 하나씩 바꾸기
5. 학습률부터 튜닝, 그다음 정규화
6. 손실 곡선·기울기 크기·활성값 분포 모니터링 (TensorBoard, W&B)

## 3. 흔한 버그
- `model.eval()` / `torch.no_grad()` 누락
- `optimizer.zero_grad()` 누락 → 기울기 누적
- CrossEntropyLoss에 softmax 출력을 넣음 (로짓 넣어야 함)
- 텐서 shape 브로드캐스팅 실수 (`(B,)` vs `(B,1)`)
- 데이터 정규화를 train/test 다르게 적용
- 라벨 셔플 안 됨 / 데이터 증강을 검증셋에 적용

## 4. 성능·메모리
- **혼합 정밀도** (bf16/fp16) → 속도↑ 메모리↓
- Gradient accumulation, Gradient checkpointing
- `torch.compile(model)`
- DataLoader: `num_workers`, `pin_memory=True`
- 분산 학습: DDP(데이터 병렬), FSDP/ZeRO(파라미터 샤딩), 텐서·파이프라인 병렬
- 고수준 도구: Hugging Face Accelerate / Trainer, PyTorch Lightning

## 5. 재현성
```python
import random, numpy as np, torch
def seed_all(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
```

## 6. 추천 학습 자료
- 『밑바닥부터 시작하는 딥러닝』 1~3권 (사이토 고키)
- Dive into Deep Learning (d2l.ai) — 무료, 코드 포함
- 『Deep Learning』 (Goodfellow et al.) — 이론
- Andrej Karpathy "Neural Networks: Zero to Hero" (YouTube)
- Stanford CS231n (비전), CS224n (NLP)
- fast.ai Practical Deep Learning
- Hugging Face NLP/LLM Course

## 7. 추천 프로젝트 순서
1. MNIST MLP → 2. CIFAR-10 CNN(+전이학습) → 3. 감성분석 LSTM
→ 4. nanoGPT 직접 구현 → 5. 오픈 LLM LoRA 파인튜닝 → 6. 간단한 Diffusion 학습

---

## 🧪 실습

**코드** · [`dl08_pytorch_training_template.py`](code/dl08_pytorch_training_template.py)  

### 📝 실습 후 정리

- 시드·AMP·clipping·accumulation·warmup+cosine·early stopping·체크포인트를 한 템플릿에 모아 프로젝트마다 재사용.
- --overfit_one_batch 모드로 손실이 0 근처로 가는지 먼저 확인하는 습관 → 버그를 가장 빨리 찾는 방법.
- weight decay를 bias·정규화 파라미터에서 제외하는 파라미터 그룹 설정까지 포함.
