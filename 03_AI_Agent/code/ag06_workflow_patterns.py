"""
[실습 06] 5가지 에이전트 워크플로우 패턴을 가짜 LLM으로 구현해 흐름 익히기
① Prompt Chaining  ② Routing  ③ Parallelization(Sectioning/Voting)  ④ Orchestrator-Workers  ⑤ Evaluator-Optimizer
각 LLM 호출은 0.2초 지연 + 호출 횟수/비용을 기록 → 패턴별 비용·지연 차이를 체감
"""
import time, random, re
from concurrent.futures import ThreadPoolExecutor
random.seed(0)

class Meter:
    def __init__(self): self.calls, self.cost = 0, 0.0
    def llm(self, prompt, model="large", latency=0.2):
        self.calls += 1; self.cost += {"small": 0.1, "large": 1.0}[model]; time.sleep(latency)
        return fake_brain(prompt)
M = Meter()

def fake_brain(p):
    """프롬프트 첫 줄의 [역할]에 따라 그럴듯한 출력을 돌려주는 가짜 LLM"""
    role = re.match(r"\[(.+?)\]", p).group(1)
    body = p.split("\n", 1)[1] if "\n" in p else ""
    if role == "개요": return "1. 정의 2. 수식 3. 예시"
    if role == "본문": return f"본문({body[:15]}…을 바탕으로 작성)"
    if role == "검증": return "PASS" if "본문" in body else "FAIL"
    if role == "분류":
        return "환불" if "환불" in body else "기술" if any(w in body for w in ["오류", "에러", "안 돼"]) else "일반"
    if role.startswith("리뷰-"): return f"{role[3:]} 관점: 문제 {random.randint(0, 2)}건"
    if role == "투표": return random.choice(["안전", "안전", "위험"])
    if role == "분해": return "data_loader.py|model.py|train.py"
    if role == "워커": return f"{body.strip()} 수정 완료"
    if role == "작성": return body + " +개선"
    if role == "평가":
        n = body.count("+개선"); return f"점수 {min(10, 5 + 2*n)}/10" + (" OK" if n >= 2 else " - 예시를 더 추가할 것")
    return "?"

def section(title):
    print(f"\n===== {title} ====="); M.calls, M.cost = 0, 0.0; return time.time()
def footer(t0): print(f"   ⏱ {time.time()-t0:.2f}s | LLM 호출 {M.calls}회 | 비용 {M.cost:.1f}")

# ① Prompt Chaining: 개요 → [게이트] → 본문 → 검증
t0 = section("① Prompt Chaining")
outline = M.llm("[개요]\n어텐션 설명 글")
if "1." not in outline: raise SystemExit("게이트 실패: 개요 형식 오류")
draft = M.llm(f"[본문]\n{outline}")
verdict = M.llm("[검증]\n" + draft)
print(f"   개요: {outline}\n   본문: {draft}\n   검증: {verdict}"); footer(t0)

# ② Routing: 분류는 작은 모델, 처리 경로를 나눔
t0 = section("② Routing")
handlers = {"환불": "환불 정책 에이전트 (결제 도구 접근)", "기술": "기술 지원 에이전트 (로그 조회 도구)", "일반": "FAQ 응답 (작은 모델)"}
for q in ["환불하고 싶어요", "로그인이 안 돼요", "영업시간이 언제예요?"]:
    route = M.llm(f"[분류]\n{q}", model="small"); print(f"   '{q}' → {route} → {handlers[route]}")
footer(t0)

# ③ Parallelization
t0 = section("③-a Parallelization: Sectioning (코드 리뷰를 관점별 동시 실행)")
aspects = ["보안", "성능", "가독성", "테스트"]
with ThreadPoolExecutor(4) as ex:
    for r in ex.map(lambda a: M.llm(f"[리뷰-{a}]\ncode"), aspects): print("  ", r)
footer(t0)
t0 = section("③-b Parallelization: Voting (같은 질문 5번 → 다수결)")
with ThreadPoolExecutor(5) as ex: votes = list(ex.map(lambda _: M.llm("[투표]\n이 코드 배포해도 되나?"), range(5)))
print(f"   투표: {votes} → 결론: {max(set(votes), key=votes.count)}"); footer(t0)

# ④ Orchestrator-Workers: 하위 작업을 실행 중에 동적으로 결정
t0 = section("④ Orchestrator-Workers")
subtasks = M.llm("[분해]\n'학습 속도 개선' 이슈 해결").split("|")
print(f"   오케스트레이터가 나눈 작업: {subtasks}")
with ThreadPoolExecutor(len(subtasks)) as ex: results = list(ex.map(lambda s: M.llm(f"[워커]\n{s}"), subtasks))
for r in results: print("   워커 →", r)
print("   오케스트레이터 종합: 3개 파일 변경 사항 통합, 테스트 실행 필요"); footer(t0)

# ⑤ Evaluator-Optimizer: 기준 통과까지 반복 (최대 횟수 제한!)
t0 = section("⑤ Evaluator-Optimizer")
text = "LoRA 설명 초안"
for i in range(5):
    fb = M.llm(f"[평가]\n{text}"); print(f"   반복 {i}: '{text}' → {fb}")
    if "OK" in fb: break
    text = M.llm(f"[작성]\n{text}")
footer(t0)

print("\n정리: 체이닝=순차·정확, 라우팅=비용절감, 병렬=속도/신뢰도, 오케스트레이터=동적 분해, 평가-최적화=품질 반복개선")
