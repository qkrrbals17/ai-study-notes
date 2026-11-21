# 에이전트 프레임워크 · MCP · 생태계

> ⚠️ 이 분야는 매우 빠르게 바뀝니다. 아래는 개념 지도이며, 구체 API는 공식 문서에서 확인하세요.

## 1. 프레임워크를 써야 할까?
- 장점: 상태 관리, 재시도, 스트리밍, 추적, 멀티에이전트 구성 등을 빠르게
- 단점: 추상화 때문에 실제 프롬프트·응답이 가려져 디버깅이 어려움
- **추천 학습 순서**: LLM API로 직접 루프 구현([09 실습](09_실습_미니에이전트_만들기.md)) → 원리 이해 후 프레임워크

## 2. 주요 프레임워크 (개념 위주)
| 이름 | 특징 |
|---|---|
| **LangChain** | LLM 앱 구성요소(프롬프트, 체인, 검색기, 도구) 모음, 넓은 통합 |
| **LangGraph** | 에이전트를 **상태 그래프(노드·엣지)**로 정의, 루프·분기·체크포인트·사람 개입에 강함 |
| **LlamaIndex** | 데이터 연결·인덱싱·RAG 중심 |
| **CrewAI** | 역할 기반 멀티에이전트 협업 |
| **AutoGen / AG2** | 대화 기반 멀티에이전트 |
| **각 LLM 제공사의 Agent SDK** | 예: Claude Agent SDK, OpenAI Agents SDK 등 — 도구 루프·서브에이전트·컨텍스트 관리 내장 |
| **DSPy** | 프롬프트를 "프로그래밍 + 자동 최적화" |
| **Pydantic AI** | 타입 안정성 중심의 파이썬 에이전트 프레임워크 |

## 3. MCP (Model Context Protocol)
- Anthropic이 공개한 **AI 애플리케이션과 외부 도구·데이터를 연결하는 개방형 표준 프로토콜**
- "AI용 USB-C": 한 번 MCP 서버로 만들면 여러 클라이언트(앱/IDE/에이전트)에서 재사용

```
[MCP 호스트/클라이언트]  ⇄  [MCP 서버: GitHub]
  (AI 앱, IDE, 에이전트)  ⇄  [MCP 서버: 사내 DB]
                         ⇄  [MCP 서버: 파일시스템]
```
MCP 서버가 제공하는 것:
- **Tools**: 모델이 호출하는 함수
- **Resources**: 읽을 수 있는 데이터(파일, 레코드)
- **Prompts**: 재사용 프롬프트 템플릿

전송 방식: 로컬(stdio), 원격(HTTP 기반)

간단한 서버 예 (Python SDK의 FastMCP 스타일):
```python
from mcp.server.fastmcp import FastMCP
mcp = FastMCP("study-notes")

@mcp.tool()
def search_notes(keyword: str) -> str:
    """공부 노트에서 키워드가 포함된 줄을 찾습니다."""
    ...

if __name__ == "__main__":
    mcp.run()
```
⚠️ 보안: 신뢰할 수 있는 서버만 연결, 권한 최소화, 도구 결과 속 지시문(인젝션) 주의

## 4. 에이전트 간 통신
- 에이전트끼리 작업을 주고받는 표준(예: A2A 프로토콜)도 등장 → MCP(에이전트↔도구)와 보완 관계

## 5. 특수 능력 에이전트
| 분야 | 설명 |
|---|---|
| **코딩 에이전트** | 저장소 탐색·편집·테스트 실행 (Claude Code 등 CLI/IDE 에이전트) |
| **컴퓨터 사용(Computer Use)** | 스크린샷을 보고 마우스·키보드 조작 |
| **브라우저 에이전트** | 웹페이지 탐색·폼 입력 (Playwright 기반 등) |
| **코드 인터프리터** | 샌드박스에서 파이썬 실행 → 데이터 분석 |
| **Skills** | 작업별 지시문·스크립트 묶음을 필요할 때 불러오는 방식 (절차적 메모리) |

## 6. 관측성 & 운영 도구
- 트레이싱: LangSmith, Langfuse, Arize Phoenix, OpenTelemetry 기반 도구
- 평가: 각 플랫폼 eval 기능, promptfoo, RAGAS
- 서빙/로컬 모델: vLLM, Ollama, llama.cpp

## ✅ 체크리스트
- [ ] 프레임워크 없이 먼저 구현해야 하는 이유를 안다
- [ ] LangGraph의 상태 그래프 개념을 이해했다
- [ ] MCP의 Tools/Resources/Prompts를 설명하고 간단한 서버를 만들어 봤다

---

## 🧪 실습

**코드** · [`ag07_mcp_server_study_notes.py`](code/ag07_mcp_server_study_notes.py)  

### 📝 실습 후 정리

- 공부 노트용 MCP 서버를 설계: Tools(list/search/read), Resource(notes://), Prompt(quiz_me).
- 경로 탈출(../../) 방지, 긴 노트는 start_line으로 나눠 읽기 등 '에이전트가 실수하기 어려운' 도구 설계를 적용.
- 한 번 MCP 서버로 만들면 여러 AI 앱/IDE에서 같은 도구를 재사용할 수 있음.
