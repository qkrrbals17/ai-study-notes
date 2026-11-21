"""
[실습 05-b] PyTorch로 문자 단위 미니 GPT(디코더 전용 Transformer) 구현 & 학습
※ 실행 환경: pip install torch   /  학습 텍스트: 같은 폴더의 input.txt (아무 한국어/영어 텍스트, 수백 KB 이상 권장)
참고: Karpathy nanoGPT 구조를 공부용으로 축약
"""
import math, torch, torch.nn as nn, torch.nn.functional as F
torch.manual_seed(1337)
device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

# ---- 하이퍼파라미터 ----
block_size, batch_size = 128, 64
n_layer, n_head, n_embd, dropout = 4, 4, 256, 0.1
max_iters, lr, eval_every = 3000, 3e-4, 300

text = open("input.txt", encoding="utf-8").read()
chars = sorted(set(text)); vocab = len(chars)
stoi = {c: i for i, c in enumerate(chars)}; itos = {i: c for c, i in stoi.items()}
encode = lambda s: [stoi[c] for c in s]; decode = lambda l: "".join(itos[i] for i in l)
data = torch.tensor(encode(text)); n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]

def get_batch(split):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x = torch.stack([d[i:i + block_size] for i in ix]); y = torch.stack([d[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)

class CausalSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(n_embd, 3 * n_embd); self.proj = nn.Linear(n_embd, n_embd); self.drop = nn.Dropout(dropout)
    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q, k, v = [t.view(B, T, n_head, C // n_head).transpose(1, 2) for t in (q, k, v)]
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=dropout if self.training else 0)
        return self.drop(self.proj(y.transpose(1, 2).contiguous().view(B, T, C)))

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(n_embd), nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention()
        self.mlp = nn.Sequential(nn.Linear(n_embd, 4 * n_embd), nn.GELU(), nn.Linear(4 * n_embd, n_embd), nn.Dropout(dropout))
    def forward(self, x):
        x = x + self.attn(self.ln1(x))      # Pre-LN + 잔차 연결
        return x + self.mlp(self.ln2(x))

class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok = nn.Embedding(vocab, n_embd); self.pos = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block() for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd); self.head = nn.Linear(n_embd, vocab, bias=False)
        self.head.weight = self.tok.weight   # weight tying
    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.tok(idx) + self.pos(torch.arange(T, device=idx.device))
        logits = self.head(self.ln_f(self.blocks(x)))
        loss = None if targets is None else F.cross_entropy(logits.view(-1, vocab), targets.view(-1))
        return logits, loss
    @torch.no_grad()
    def generate(self, idx, max_new, temperature=0.8, top_k=50):
        for _ in range(max_new):
            logits, _ = self(idx[:, -block_size:])
            logits = logits[:, -1] / temperature
            v, _ = torch.topk(logits, top_k); logits[logits < v[:, [-1]]] = -float("inf")
            idx = torch.cat([idx, torch.multinomial(F.softmax(logits, -1), 1)], dim=1)
        return idx

model = GPT().to(device)
print(f"파라미터 수: {sum(p.numel() for p in model.parameters())/1e6:.2f}M, 어휘 {vocab}")
opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
warmup = 200
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(s, max_iters) / max_iters)))

@torch.no_grad()
def estimate():
    model.eval(); out = {}
    for split in ("train", "val"):
        out[split] = torch.stack([model(*get_batch(split))[1] for _ in range(50)]).mean().item()
    model.train(); return out

for it in range(max_iters + 1):
    if it % eval_every == 0:
        l = estimate(); print(f"iter {it}: train {l['train']:.3f} | val {l['val']:.3f}")
    _, loss = model(*get_batch("train"))
    opt.zero_grad(set_to_none=True); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()

start = torch.tensor([encode(text[:1])], device=device)
print(decode(model.generate(start, 300)[0].tolist()))
