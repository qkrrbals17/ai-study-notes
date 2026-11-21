"""
[실습 07-b] Hugging Face + PEFT로 작은 오픈 LLM을 LoRA 지시 미세조정(SFT)
※ 실행 환경: pip install torch transformers datasets peft accelerate  (GPU 권장)
※ MODEL 은 Hugging Face Hub에서 사용할 수 있는 작은 instruct 모델 이름으로 바꾸세요.
"""
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model

MODEL = "<작은-instruct-모델-이름>"
tok = AutoTokenizer.from_pretrained(MODEL)
tok.pad_token = tok.pad_token or tok.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="auto")

# 1) 데이터: (지시, 응답) 쌍 → 모델의 chat template으로 변환
pairs = [
    {"q": "과적합이 뭐야?", "a": "훈련 데이터는 잘 맞히지만 새 데이터에서 성능이 떨어지는 현상입니다. 규제·데이터 증강·조기 종료로 완화합니다."},
    {"q": "LoRA를 한 줄로 설명해줘", "a": "가중치 변화량을 저랭크 행렬 두 개(B·A)로 근사해 아주 적은 파라미터만 학습하는 미세조정 기법입니다."},
    # ... 실제로는 수백~수천 개
]
def fmt(ex):
    msgs = [{"role": "user", "content": ex["q"]}, {"role": "assistant", "content": ex["a"]}]
    return {"text": tok.apply_chat_template(msgs, tokenize=False)}
ds = Dataset.from_list(pairs).map(fmt)
ds = ds.map(lambda e: tok(e["text"], truncation=True, max_length=512), remove_columns=ds.column_names)

# 2) LoRA 설정
cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
                 target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
model = get_peft_model(model, cfg)
model.print_trainable_parameters()          # 보통 전체의 1% 미만

# 3) 학습
args = TrainingArguments(output_dir="lora-out", per_device_train_batch_size=4, gradient_accumulation_steps=4,
                         num_train_epochs=3, learning_rate=2e-4, lr_scheduler_type="cosine", warmup_ratio=0.05,
                         logging_steps=10, bf16=True, save_strategy="epoch", report_to="none")
Trainer(model=model, args=args, train_dataset=ds,
        data_collator=DataCollatorForLanguageModeling(tok, mlm=False)).train()
model.save_pretrained("lora-out/adapter")   # 어댑터만 저장 (수십 MB)

# 4) 추론
model.eval()
prompt = tok.apply_chat_template([{"role": "user", "content": "과적합을 줄이는 방법은?"}], tokenize=False, add_generation_prompt=True)
ids = tok(prompt, return_tensors="pt").to(model.device)
out = model.generate(**ids, max_new_tokens=200, temperature=0.7, top_p=0.9, do_sample=True)
print(tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True))
