"""
[실습 01] 에이전트 루프의 뼈대: LLM → 도구 호출 → 관찰 → 반복 → 종료
- API 키 없이 원리를 보기 위해 '규칙 기반 가짜 LLM(FakeLLM)'을 사용
- 실제 LLM으로 바꾸려면 FakeLLM.chat 만 API 호출로 교체하면 됨 (ag09 참고)
"""
import json, re, datetime

# ---------------- 도구 ----------------
def calculator(expression: str) -> str:
    if not re.fullmatch(r"[\d+\-*/(). ]+", expression):
        return "오류: 숫자와 + - * / ( ) 만 허용됩니다."
    return str(eval(expression, {"__builtins__": {}}))

FAKE_WEATHER = {"서울": ("맑음", 3), "부산": ("흐림", 9), "제주": ("비", 12)}
def get_weather(city: str) -> str:
    if city not in FAKE_WEATHER:
        return f"오류: '{city}'의 날씨 정보가 없습니다. 가능한 도시: {list(FAKE_WEATHER)}"
    s, t = FAKE_WEATHER[city]; return json.dumps({"city": city, "sky": s, "temp_c": t}, ensure_ascii=False)

def today() -> str:
    return datetime.date(2026, 11, 22).isoformat()

TOOLS = {"calculator": calculator, "get_weather": get_weather, "today": today}

# ---------------- 가짜 LLM ----------------
class FakeLLM:
    """대화 기록을 보고 '다음 행동'을 결정하는 척하는 규칙 기반 모델"""
    def chat(self, messages):
        user = messages[0]["content"]
        observed = [m for m in messages if m["role"] == "tool"]
        done = {m["name"] for m in observed}
        cities = [c for c in FAKE_WEATHER if c in user] or re.findall(r"(\w+)\s*날씨", user)
        # 1) 날씨가 필요하면 도시마다 조회
        if "날씨" in user:
            for c in cities:
                if not any(m["name"] == "get_weather" and c in m["args"] for m in observed):
                    return {"tool_call": {"name": "get_weather", "args": {"city": c}}}
        # 2) 기온 차이 계산이 필요하면 계산기
        if "차이" in user and "calculator" not in done:
            temps = [json.loads(m["content"])["temp_c"] for m in observed if m["name"] == "get_weather" and m["content"].startswith("{")]
            if len(temps) == 2:
                return {"tool_call": {"name": "calculator", "args": {"expression": f"{temps[1]} - {temps[0]}"}}}
        if "날짜" in user and "today" not in done:
            return {"tool_call": {"name": "today", "args": {}}}
        # 3) 충분하면 최종 답변
        facts = "; ".join(f"{m['name']}{m['args']} → {m['content']}" for m in observed)
        return {"text": f"[최종 답변] 수집한 정보: {facts}"}

# ---------------- 에이전트 루프 ----------------
def run_agent(goal, llm=FakeLLM(), max_steps=6):
    messages = [{"role": "user", "content": goal}]
    print(f"\n🎯 목표: {goal}")
    for step in range(1, max_steps + 1):
        resp = llm.chat(messages)
        if "text" in resp:
            print(f"  ✅ step {step}: 종료 → {resp['text']}")
            return resp["text"]
        call = resp["tool_call"]
        try:
            result = TOOLS[call["name"]](**call["args"])
        except Exception as e:                                       # 도구 오류도 '관찰'로 돌려줌
            result = f"도구 실행 오류: {e}"
        print(f"  🔧 step {step}: {call['name']}({call['args']}) → {result}")
        messages.append({"role": "tool", "name": call["name"], "args": json.dumps(call["args"], ensure_ascii=False), "content": result})
    print("  ⛔ 최대 단계 도달 (무한 루프 방지)")
    return None

if __name__ == "__main__":
    run_agent("서울 날씨 알려줘")
    run_agent("서울이랑 제주 날씨 차이가 몇 도야?")
    run_agent("오늘 날짜랑 부산 날씨")
    run_agent("광주 날씨 알려줘")                        # 도구가 오류를 반환하는 경우

    # 루프가 끝나지 않는 LLM을 넣어도 max_steps가 막아줌
    class LoopingLLM:
        def chat(self, messages): return {"tool_call": {"name": "today", "args": {}}}
    run_agent("계속 같은 행동만 반복하는 모델", llm=LoopingLLM(), max_steps=3)
