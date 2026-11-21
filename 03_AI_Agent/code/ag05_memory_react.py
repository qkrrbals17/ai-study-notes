"""
[실습 05] ReAct 파서 + 대화 메모리(슬라이딩 윈도우 / 요약) + 장기 메모리(검색형) + Plan-and-Execute
"""
import re, json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- 1) ReAct 형식 파싱 ----------------
REACT_PROMPT = """다음 형식을 반드시 지켜라.
Thought: 지금 무엇을 해야 하는지 생각
Action: 도구이름[입력]
Observation: (도구 결과가 여기에 들어온다)
... (반복)
Thought: 이제 답을 안다
Final Answer: 최종 답"""

def parse_react(text):
    if m := re.search(r"Final Answer:\s*(.+)", text, re.S): return {"final": m.group(1).strip()}
    m = re.search(r"Thought:\s*(.+?)\nAction:\s*(\w+)\[(.*?)\]", text, re.S)
    if not m: return {"error": "형식 오류: 'Action: 도구[입력]' 또는 'Final Answer:'가 필요합니다."}
    return {"thought": m.group(1).strip(), "tool": m.group(2), "input": m.group(3)}

KB = {"트랜스포머 발표 연도": "2017", "ResNet 발표 연도": "2015"}
tools = {"lookup": lambda q: KB.get(q, "정보 없음"), "calc": lambda e: str(eval(e, {"__builtins__": {}}))}
scripted = iter([   # 가짜 LLM이 단계마다 내놓는 텍스트
    "Thought: 트랜스포머 연도를 찾자\nAction: lookup[트랜스포머 발표 연도]",
    "Thought: ResNet 연도도 필요\nAction: lookup[ResNet 발표 연도]",
    "음.. 빼면 되겠네",                                                  # 형식 오류 상황
    "Thought: 두 연도 차이를 계산\nAction: calc[2017-2015]",
    "Thought: 이제 답을 안다\nFinal Answer: 트랜스포머는 ResNet보다 2년 늦게 발표되었다.",
])
print("===== ReAct 실행 기록 =====")
scratch = "Question: 트랜스포머는 ResNet보다 몇 년 뒤에 나왔나?\n"
for step in range(8):
    out = next(scripted); p = parse_react(out)
    if "final" in p: print(f"[Final] {p['final']}"); break
    if "error" in p:
        print(f"[!] {p['error']} → 모델에게 형식 오류를 관찰로 돌려줌"); scratch += f"{out}\nObservation: {p['error']}\n"; continue
    obs = tools[p["tool"]](p["input"])
    print(f"[Thought] {p['thought']}\n[Action] {p['tool']}({p['input']})\n[Observation] {obs}")
    scratch += f"{out}\nObservation: {obs}\n"
print(f"(누적 스크래치패드 길이: {len(scratch)}자)\n")

# ---------------- 2) 단기 메모리: 슬라이딩 윈도우 + 요약 ----------------
def n_tokens(msgs): return sum(len(m["content"]) for m in msgs) // 2      # 대략적인 토큰 추정 (실제는 토크나이저 사용)

def fake_summarize(msgs):
    facts = [m["content"] for m in msgs if m["role"] == "user"]
    return "이전 대화 요약: 사용자가 " + " / ".join(f[:20] for f in facts) + " 에 대해 이야기함"

class ConversationMemory:
    def __init__(self, budget=120, keep_last=4):
        self.msgs, self.summary, self.budget, self.keep_last = [], "", budget, keep_last
    def add(self, role, content):
        self.msgs.append({"role": role, "content": content})
        if n_tokens(self.msgs) > self.budget:                 # 예산 초과 → 오래된 것 요약(compaction)
            old, self.msgs = self.msgs[:-self.keep_last], self.msgs[-self.keep_last:]
            self.summary = (self.summary + " | " if self.summary else "") + fake_summarize(old)
            print(f"   🗜  컨텍스트 압축: 메시지 {len(old)}개 → 요약 1개")
    def context(self):
        return ([{"role": "system", "content": self.summary}] if self.summary else []) + self.msgs

mem = ConversationMemory()
dialog = ["내 이름은 민준이고 AI를 공부 중이야", "요즘 트랜스포머 논문을 읽고 있어", "어텐션 수식이 헷갈려",
          "멀티헤드는 왜 쓰는 거야?", "LoRA도 궁금해", "다음 주에 발표가 있어", "에이전트 프로젝트도 할 거야"]
print("===== 대화 메모리 =====")
for u in dialog:
    mem.add("user", u); mem.add("assistant", f"('{u[:8]}…'에 대한 답변입니다)")
print("최종 컨텍스트:")
for m in mem.context(): print(f"   [{m['role']}] {m['content'][:160]}")

# ---------------- 3) 장기 메모리: 저장 → 검색 → 주입 ----------------
class LongTermMemory:
    def __init__(self): self.items = []
    def remember(self, text, kind): self.items.append({"text": text, "kind": kind})
    def recall(self, query, k=2):
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 3)).fit([i["text"] for i in self.items] + [query])
        sims = cosine_similarity(vec.transform([query]), vec.transform([i["text"] for i in self.items]))[0]
        return [(round(float(sims[i]), 3), self.items[i]["text"]) for i in sims.argsort()[::-1][:k]]
ltm = LongTermMemory()
ltm.remember("사용자는 코드 예시를 파이썬으로 받는 것을 선호한다", "선호")
ltm.remember("사용자는 다음 주 화요일에 트랜스포머 발표를 한다", "일정")
ltm.remember("사용자는 PyTorch 2.x 환경을 쓴다", "환경")
ltm.remember("지난번 LoRA 설명 때 수식보다 그림 설명을 좋아했다", "피드백")
print("\n===== 장기 메모리 검색 =====")
for q in ["발표 준비 도와줘", "LoRA 예제 코드 보여줘"]:
    print(f"질문 '{q}' → 꺼낸 기억: {ltm.recall(q)}")

# ---------------- 4) Plan-and-Execute ----------------
print("\n===== Plan-and-Execute =====")
plan = [{"id": 1, "task": "어텐션 개념 정리", "done": False}, {"id": 2, "task": "수식 예제 계산", "done": False},
        {"id": 3, "task": "발표 슬라이드 개요 작성", "done": False}, {"id": 4, "task": "예상 질문 3개 준비", "done": False}]
def executor(task):
    if "수식" in task and "가정" not in task: return {"ok": False, "note": "d_k 값 정보가 없어 계산 불가"}
    return {"ok": True, "note": f"'{task}' 완료"}
for step in range(10):
    todo = [p for p in plan if not p["done"]]
    if not todo: print("모든 단계 완료 ✅"); break
    cur = todo[0]; r = executor(cur["task"])
    print(f"[실행] {cur['id']}. {cur['task']} → {r['note']}")
    if r["ok"]: cur["done"] = True
    else:                                                     # 실패 시 재계획(re-plan)
        print("   ↻ 재계획: 부족한 정보를 가정으로 보완해 같은 단계를 수정")
        cur["task"] = "d_k=64 가정 후 수식 예제 계산"
print("최종 계획 상태:", json.dumps([(p["id"], p["task"], p["done"]) for p in plan], ensure_ascii=False))
