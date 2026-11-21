"""
[실습 07] 내 공부 노트를 검색하는 MCP 서버
- Tools: search_notes(키워드 검색), read_note(노트 전문), list_notes
- Resources: notes://{path} 로 노트 읽기
- Prompts: quiz_me(주제) → 퀴즈 출제 프롬프트 템플릿
※ 실행 환경: pip install "mcp[cli]"   (Python MCP SDK)
   실행: python ag07_mcp_server_study_notes.py  (stdio 서버)
   데스크톱 AI 앱/IDE의 MCP 설정에 이 파일 실행 명령을 등록하면 연결됨 (설정 방법은 각 앱 문서 참고)
"""
import glob, os
from mcp.server.fastmcp import FastMCP

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
mcp = FastMCP("study-notes")

def _notes():
    return sorted(os.path.relpath(p, ROOT) for p in glob.glob(os.path.join(ROOT, "**", "*.md"), recursive=True))

def _safe_path(rel: str) -> str:
    path = os.path.realpath(os.path.join(ROOT, rel))
    if not path.startswith(os.path.realpath(ROOT) + os.sep):          # 경로 탈출(../../etc/passwd) 방지
        raise ValueError("노트 폴더 밖의 파일은 읽을 수 없습니다.")
    return path

@mcp.tool()
def list_notes() -> list[str]:
    """공부 노트(md) 파일 목록을 반환한다. 어떤 노트가 있는지 모를 때 먼저 호출."""
    return _notes()

@mcp.tool()
def search_notes(keyword: str, max_results: int = 15) -> str:
    """노트 전체에서 키워드(대소문자 무시)가 들어간 줄을 '파일:줄번호: 내용' 형식으로 반환한다.
    개념 질문에 답하기 전에 관련 내용을 찾을 때 사용. 결과가 없으면 더 짧거나 다른 키워드로 다시 시도."""
    hits = []
    for rel in _notes():
        for i, line in enumerate(open(os.path.join(ROOT, rel), encoding="utf-8"), 1):
            if keyword.lower() in line.lower():
                hits.append(f"{rel}:{i}: {line.strip()[:200]}")
    return "\n".join(hits[:max_results]) or f"'{keyword}' 검색 결과 없음. 동의어나 영어/한국어 표현을 바꿔 보세요."

@mcp.tool()
def read_note(path: str, start_line: int = 1, max_lines: int = 200) -> str:
    """노트 파일 내용을 읽는다. path는 list_notes가 반환한 상대경로. 긴 노트는 start_line으로 나눠 읽기."""
    lines = open(_safe_path(path), encoding="utf-8").read().splitlines()
    chunk = lines[start_line - 1:start_line - 1 + max_lines]
    more = f"\n... ({len(lines)}줄 중 {start_line + len(chunk) - 1}줄까지. 이어 읽으려면 start_line={start_line + len(chunk)})" \
        if start_line - 1 + max_lines < len(lines) else ""
    return "\n".join(chunk) + more

@mcp.resource("notes://{path}")
def note_resource(path: str) -> str:
    """노트를 리소스로 노출 (클라이언트가 컨텍스트에 첨부할 수 있음)"""
    return open(_safe_path(path), encoding="utf-8").read()

@mcp.prompt()
def quiz_me(topic: str, n: int = 5) -> str:
    """주제에 대한 복습 퀴즈를 만드는 프롬프트 템플릿"""
    return (f"search_notes 도구로 '{topic}' 관련 노트를 찾아 읽은 뒤, 노트 내용만 근거로 객관식 {n}문제를 만들어줘. "
            f"각 문제에 정답과 해설, 근거가 된 노트 파일명을 달아줘. 한 문제씩 내고 내 답을 기다려줘.")

if __name__ == "__main__":
    mcp.run()          # 기본: stdio 전송
