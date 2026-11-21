# 도구 사용 (Tool Use / Function Calling)

LLM은 텍스트만 생성한다. **도구**를 통해 검색·계산·코드 실행·외부 시스템 조작이 가능해진다.

## 1. 동작 방식
```
① 개발자가 도구 목록(이름, 설명, 입력 JSON 스키마)을 LLM에 제공
② LLM이 필요하다고 판단하면 "이 도구를 이 인자로 호출해줘"라는 구조화된 요청 출력
③ 애플리케이션 코드가 실제로 도구를 실행  ← LLM이 직접 실행하는 게 아님!
④ 결과를 다시 LLM에 전달
⑤ LLM이 최종 답변 또는 다음 도구 호출
```

## 2. 도구 정의 예시
```json
{
  "name": "get_weather",
  "description": "특정 도시의 현재 날씨를 조회합니다. 사용자가 날씨, 기온, 우산 필요 여부 등을 물을 때 사용하세요.",
  "input_schema": {
    "type": "object",
    "properties": {
      "city": {"type": "string", "description": "도시 이름, 예: 서울"},
      "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
    },
    "required": ["city"]
  }
}
```

## 3. 좋은 도구 설계 원칙 (ACI: Agent-Computer Interface)
> 사람을 위한 UI(HCI)에 공을 들이듯, 에이전트를 위한 도구 인터페이스에 공을 들여라.

1. **설명을 자세히**: 무엇을 하는지, 언제 쓰는지, 언제 쓰지 말지, 입력 예시
2. **명확한 이름**: `search_customer_by_email` > `query`
3. **적은 수의 고수준 도구**: API를 1:1로 노출하기보다 작업 단위로 묶기
   (예: `list_users` + `list_events` + `create_event` → `schedule_meeting`)
4. **실수하기 어렵게**: 상대경로 대신 절대경로 요구, enum 활용, 검증
5. **의미 있는 결과 반환**: 불필요한 ID/메타데이터 제거, 필요한 정보만, 길면 페이지네이션/요약
6. **유익한 에러 메시지**: "Error 400" ❌ → "날짜 형식이 잘못됨. YYYY-MM-DD 형식 사용" ✅
7. **네임스페이스**로 비슷한 도구 구분 (`github_search`, `jira_search`)
8. 모델에게 실제로 시켜보고 **오용 패턴을 관찰해 설명을 개선**

## 4. 도구 유형
| 유형 | 예시 | 위험도 |
|---|---|---|
| 읽기(조회) | 검색, DB 조회, 파일 읽기 | 낮음 |
| 계산/실행 | 코드 인터프리터 (샌드박스 필수) | 중간 |
| 쓰기(부작용) | 이메일 전송, 결제, 파일 삭제, DB 수정 | **높음 → 사람 확인** |

## 5. 실행 시 고려사항
- **병렬 도구 호출**: 독립적인 호출은 동시에 실행
- **타임아웃·재시도**, 멱등성(idempotency) 고려
- **최대 반복 횟수** 제한 (무한 루프 방지)
- 도구 결과는 **신뢰할 수 없는 데이터**로 취급 (웹페이지 안의 지시문 등)
- 권한 최소화 (읽기 전용 키, 범위 제한)
- 로그 남기기 (어떤 도구를 왜 호출했는지)

## 6. 파이썬 의사코드
```python
TOOLS = {"get_weather": get_weather, "search_web": search_web}

def run_agent(user_msg, max_steps=10):
    messages = [{"role": "user", "content": user_msg}]
    for _ in range(max_steps):
        resp = llm.chat(messages=messages, tools=TOOL_SCHEMAS)
        messages.append(resp.message)
        if not resp.tool_calls:
            return resp.text
        for call in resp.tool_calls:
            try:
                result = TOOLS[call.name](**call.args)
            except Exception as e:
                result = f"도구 오류: {e}. 인자를 확인하고 다시 시도하세요."
            messages.append({"role": "tool", "tool_call_id": call.id, "content": str(result)})
    return "최대 단계 도달"
```
※ 실제 메시지 형식은 사용하는 LLM API(Anthropic, OpenAI 등)마다 조금씩 다르니 공식 문서 확인

---

## 🧪 실습

**코드** · [`ag03_tool_registry.py`](code/ag03_tool_registry.py)  
**실행 로그** · [`ag03_tool_registry.txt`](code/outputs/ag03_tool_registry.txt)

### 📝 실습 후 정리

- 함수 시그니처·독스트링에서 JSON 스키마를 자동 생성 → 도구 설명과 실제 코드가 어긋나지 않음.
- 모델이 흔히 하는 실수(이름으로 검색, 타입 오류, 없는 enum, 없는 도구)를 '어떻게 고칠지' 알려주는 에러 메시지로 반환.
- 부작용 도구(send_email)는 승인 없이는 차단. 독립 호출 3개를 병렬 실행해 0.90s→0.30s.
