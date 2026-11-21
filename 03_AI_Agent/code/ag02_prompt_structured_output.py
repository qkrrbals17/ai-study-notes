"""
[실습 02] 프롬프트 템플릿 + Few-shot 조립 + 구조화 출력(JSON) 검증/재시도
- 프롬프트를 코드로 '조립'하는 습관 (역할 / 컨텍스트 / 작업 / 형식 / 예시를 분리)
- Pydantic으로 LLM 출력 검증, 실패 시 오류 메시지를 넣어 재요청 (self-repair)
"""
import json, re
from typing import Literal
from pydantic import BaseModel, Field, ValidationError

# ---------------- 1) 프롬프트 템플릿 ----------------
def build_prompt(role, task, context=None, fmt=None, examples=(), rules=()):
    parts = [f"<role>\n{role}\n</role>"]
    if context: parts.append(f"<context>\n{context}\n</context>")
    if rules: parts.append("<rules>\n" + "\n".join(f"- {r}" for r in rules) + "\n</rules>")
    for i, (inp, out) in enumerate(examples, 1):
        parts.append(f"<example id=\"{i}\">\n<input>{inp}</input>\n<output>{out}</output>\n</example>")
    if fmt: parts.append(f"<format>\n{fmt}\n</format>")
    parts.append(f"<task>\n{task}\n</task>")
    return "\n\n".join(parts)

class Ticket(BaseModel):
    category: Literal["결제", "배송", "계정", "기타"]
    urgency: int = Field(ge=1, le=5, description="1(낮음)~5(긴급)")
    summary: str = Field(max_length=60)
    needs_human: bool

prompt = build_prompt(
    role="당신은 쇼핑몰 고객센터의 문의 분류 담당자입니다.",
    rules=["카드 이중결제, 계정 도용은 긴급도 5", "개인정보는 summary에 넣지 않는다", "확신이 없으면 needs_human=true"],
    examples=[("배송이 일주일째 안 와요", '{"category":"배송","urgency":3,"summary":"배송 지연 1주","needs_human":false}')],
    fmt="JSON 한 개만 출력. 스키마: " + json.dumps(Ticket.model_json_schema()["properties"], ensure_ascii=False),
    task="고객 문의: '어제 결제했는데 카드에서 두 번 빠져나갔어요. 빨리 확인 부탁드립니다.'",
)
print("===== 조립된 프롬프트 =====\n" + prompt[:900] + "\n...\n")

# ---------------- 2) 출력 파싱 + 검증 + 재시도 ----------------
def extract_json(text):
    m = re.search(r"\{.*\}", text, re.S)                       # 앞뒤 잡담/코드펜스 제거
    if not m: raise ValueError("JSON 객체를 찾지 못함")
    return json.loads(m.group())

class FlakyLLM:
    """처음엔 형식을 틀리고, 오류 피드백을 받으면 고치는 가짜 LLM"""
    def __init__(self): self.calls = 0
    def __call__(self, prompt):
        self.calls += 1
        if self.calls == 1:
            return '네! 분류 결과입니다:\n```json\n{"category": "카드", "urgency": 9, "summary": "이중 결제", "needs_human": "yes"}\n```'
        return '{"category": "결제", "urgency": 5, "summary": "카드 이중 결제 확인 요청", "needs_human": true}'

def structured_call(llm, prompt, schema, max_retries=3):
    for attempt in range(1, max_retries + 1):
        raw = llm(prompt)
        try:
            obj = schema.model_validate(extract_json(raw))
            print(f"✅ 시도 {attempt}: 검증 통과 → {obj}")
            return obj
        except (ValidationError, ValueError, json.JSONDecodeError) as e:
            errs = e.errors() if isinstance(e, ValidationError) else [{"msg": str(e)}]
            short = "; ".join(f"{'.'.join(map(str, x.get('loc', [])))}: {x['msg']}" for x in errs)
            print(f"❌ 시도 {attempt}: 검증 실패 → {short}")
            prompt += f"\n\n<previous_error>\n이전 출력이 스키마를 위반했습니다: {short}\n올바른 JSON만 다시 출력하세요.\n</previous_error>"
    raise RuntimeError("구조화 출력 실패")

ticket = structured_call(FlakyLLM(), prompt, Ticket)
print("\n후속 처리 예:", "상담원 연결" if ticket.needs_human or ticket.urgency >= 4 else "자동 응답")

# ---------------- 3) 좋은 프롬프트 vs 나쁜 프롬프트 체크 ----------------
def lint_prompt(p):
    checks = {"역할/배경 명시": "<role>" in p or "당신은" in p, "출력 형식 지정": "<format>" in p or "JSON" in p,
              "예시 포함": "<example" in p, "데이터와 지시 분리(태그)": bool(re.search(r"<\w+>", p)),
              "구체적 규칙": "<rules>" in p}
    return {k: ("✓" if v else "✗") for k, v in checks.items()}
print("\n[프롬프트 점검] 나쁜 예 '이거 분류해줘':", lint_prompt("이거 분류해줘"))
print("[프롬프트 점검] 위에서 조립한 프롬프트:", lint_prompt(prompt))
