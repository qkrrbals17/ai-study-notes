"""
[실습 03] 도구(Function Calling) 인프라 직접 만들기
- @tool 데코레이터: 함수 시그니처/독스트링 → JSON 스키마 자동 생성
- 인자 검증, 친절한 에러 메시지, 타임아웃, 병렬 실행
- 부작용 있는(위험한) 도구는 '사람 승인' 없이는 실행 안 되게
"""
import inspect, json, time, typing
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FTimeout

REGISTRY = {}
PY2JSON = {str: "string", int: "integer", float: "number", bool: "boolean", list: "array", dict: "object"}

def tool(side_effect=False, timeout=5.0):
    def deco(fn):
        sig = inspect.signature(fn); hints = typing.get_type_hints(fn)
        doc = inspect.getdoc(fn) or ""
        desc_lines, arg_desc = [], {}
        for line in doc.splitlines():                      # "Args:" 아래 "name: 설명" 파싱
            if ":" in line and line.strip().split(":")[0] in sig.parameters:
                k, v = line.strip().split(":", 1); arg_desc[k] = v.strip()
            elif line.strip() and not line.strip().startswith("Args"):
                desc_lines.append(line.strip())
        props, required = {}, []
        for name, p in sig.parameters.items():
            t = hints.get(name, str)
            if typing.get_origin(t) is typing.Literal:
                props[name] = {"type": "string", "enum": list(typing.get_args(t))}
            else:
                props[name] = {"type": PY2JSON.get(t, "string")}
            if name in arg_desc: props[name]["description"] = arg_desc[name]
            if p.default is inspect.Parameter.empty: required.append(name)
        schema = {"name": fn.__name__, "description": " ".join(desc_lines),
                  "input_schema": {"type": "object", "properties": props, "required": required}}
        REGISTRY[fn.__name__] = {"fn": fn, "schema": schema, "side_effect": side_effect, "timeout": timeout}
        return fn
    return deco

# ---------------- 도구 정의 ----------------
STUDENTS = {"kim@uni.ac.kr": {"name": "김OO", "score": 87}, "lee@uni.ac.kr": {"name": "이OO", "score": 92}}

@tool()
def search_student(email: str) -> str:
    """이메일로 학생의 이름과 점수를 조회한다. 학생 성적 질문에 사용. 이름으로는 검색할 수 없다.
    Args:
    email: 학교 이메일 전체 주소 (예: kim@uni.ac.kr)
    """
    if "@" not in email:
        return "오류: email 형식이 아닙니다. 'id@uni.ac.kr' 형태의 전체 주소를 넣으세요."
    s = STUDENTS.get(email)
    return json.dumps(s, ensure_ascii=False) if s else f"'{email}' 학생 없음. 오타를 확인하세요."

@tool()
def convert_grade(score: int, scale: typing.Literal["letter", "gpa"] = "letter") -> str:
    """점수(0~100)를 학점으로 변환한다.
    Args:
    score: 0~100 정수 점수
    scale: letter(A~F) 또는 gpa(4.5 만점)
    """
    if not 0 <= score <= 100: return "오류: score는 0~100 사이여야 합니다."
    letter = "A" if score >= 90 else "B" if score >= 80 else "C" if score >= 70 else "F"
    return letter if scale == "letter" else str({"A": 4.5, "B": 3.5, "C": 2.5, "F": 0.0}[letter])

@tool(timeout=0.5)
def slow_api(query: str) -> str:
    """느린 외부 API (타임아웃 실험용)."""
    time.sleep(2); return "늦은 응답"

@tool(side_effect=True)
def send_email(to: str, subject: str, body: str) -> str:
    """이메일을 실제로 보낸다. 되돌릴 수 없으므로 사용자 승인 필요."""
    return f"📧 {to} 에게 '{subject}' 전송 완료"

# ---------------- 실행기 ----------------
pool = ThreadPoolExecutor(8)

def validate(schema, args):
    s = schema["input_schema"]
    missing = [r for r in s["required"] if r not in args]
    if missing: return f"필수 인자 누락: {missing}"
    unknown = [k for k in args if k not in s["properties"]]
    if unknown: return f"알 수 없는 인자: {unknown}. 사용 가능: {list(s['properties'])}"
    for k, v in args.items():
        spec = s["properties"][k]
        if "enum" in spec and v not in spec["enum"]: return f"'{k}'는 {spec['enum']} 중 하나여야 합니다 (받은 값: {v})"
        if spec["type"] == "integer" and not isinstance(v, int): return f"'{k}'는 정수여야 합니다 (받은 값: {v!r})"
    return None

def execute(name, args, approver=lambda name, args: False):
    if name not in REGISTRY: return f"오류: '{name}' 도구는 없습니다. 사용 가능: {list(REGISTRY)}"
    t = REGISTRY[name]
    if (err := validate(t["schema"], args)): return f"인자 오류: {err}"
    if t["side_effect"] and not approver(name, args):
        return "⛔ 사용자 승인이 없어 실행하지 않았습니다. 사용자에게 내용을 보여주고 확인을 받으세요."
    fut = pool.submit(t["fn"], **args)
    try: return fut.result(timeout=t["timeout"])
    except FTimeout: return f"오류: {t['timeout']}초 타임아웃. 잠시 후 재시도하거나 다른 방법을 사용하세요."
    except Exception as e: return f"도구 내부 오류: {type(e).__name__}: {e}"

def execute_parallel(calls, **kw):
    t0 = time.time()
    futs = [pool.submit(execute, c["name"], c["args"], **kw) for c in calls]
    res = [f.result() for f in futs]
    return res, time.time() - t0

if __name__ == "__main__":
    print("===== 자동 생성된 도구 스키마 (LLM에 그대로 전달) =====")
    print(json.dumps(REGISTRY["convert_grade"]["schema"], ensure_ascii=False, indent=2))

    print("\n===== 실행 테스트 =====")
    tests = [("search_student", {"email": "kim@uni.ac.kr"}),
             ("search_student", {"email": "김OO"}),                    # 모델이 흔히 하는 실수
             ("convert_grade", {"score": 92, "scale": "gpa"}),
             ("convert_grade", {"score": "92점"}),                     # 타입 오류
             ("convert_grade", {"score": 92, "scale": "percent"}),     # enum 오류
             ("search_students", {"email": "x"}),                      # 없는 도구 (환각)
             ("slow_api", {"query": "x"}),                             # 타임아웃
             ("send_email", {"to": "prof@uni.ac.kr", "subject": "성적", "body": "..."})]
    for name, args in tests:
        print(f"{name}({args})\n   → {execute(name, args)}")

    print("\n[승인 후 실행]", execute("send_email", {"to": "prof@uni.ac.kr", "subject": "성적", "body": "..."},
                                  approver=lambda n, a: (print(f"   (사용자에게 확인 요청: {n} {a['to']}) → 승인"), True)[1]))

    def slow_lookup(email: str) -> str:
        time.sleep(0.3); return search_student(email)
    REGISTRY["slow_lookup"] = {**REGISTRY["search_student"], "fn": slow_lookup}
    calls = [{"name": "slow_lookup", "args": {"email": e}} for e in ["kim@uni.ac.kr", "lee@uni.ac.kr", "park@uni.ac.kr"]]
    t0 = time.time(); seq = [execute(c["name"], c["args"]) for c in calls]; t_seq = time.time() - t0
    par, t_par = execute_parallel(calls)
    print(f"\n[병렬 도구 호출] 순차 {t_seq:.2f}s vs 병렬 {t_par:.2f}s  결과 동일: {seq == par}")
