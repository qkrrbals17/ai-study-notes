"""
[실습 04] 내 공부 노트(md 파일들)로 RAG 검색 파이프라인 만들기 + 검색 품질 평가
- 청킹(헤더 기준 + 길이 제한 + overlap) / BM25 직접 구현 / TF-IDF(문자 n-gram) 벡터 검색
- 하이브리드 검색: RRF(Reciprocal Rank Fusion)
- 평가: Recall@k, MRR  /  최종 프롬프트(근거 + 인용) 생성
"""
import glob, math, os, re
from collections import Counter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

# ---------------- 1) 문서 로드 & 청킹 ----------------
def load_notes(root):
    files = sorted(glob.glob(os.path.join(root, "**", "*.md"), recursive=True))
    return {os.path.relpath(f, root): open(f, encoding="utf-8").read() for f in files if "README" not in f}

def chunk_markdown(doc_id, text, max_chars=700, overlap=120):
    chunks, title = [], ""
    sections = re.split(r"\n(?=#{1,3} )", text)
    for sec in sections:
        head = sec.strip().splitlines()[0] if sec.strip() else ""
        if head.startswith("# "): title = head[2:]
        body = sec.strip()
        start = 0
        while start < len(body):
            piece = body[start:start + max_chars]
            # 맥락 정보 붙이기 (contextual chunk): 어떤 문서의 어떤 섹션인지
            chunks.append({"id": f"{doc_id}#{len(chunks)}", "doc": doc_id, "section": head.lstrip("# "),
                           "text": f"[문서: {title}] [섹션: {head.lstrip('# ')}]\n{piece}"})
            if start + max_chars >= len(body): break
            start += max_chars - overlap
    return chunks

docs = load_notes(ROOT)
chunks = [c for d, t in docs.items() for c in chunk_markdown(d, t)]
print(f"문서 {len(docs)}개 → 청크 {len(chunks)}개 (평균 {np.mean([len(c['text']) for c in chunks]):.0f}자)")

# ---------------- 2) BM25 (키워드 검색) 직접 구현 ----------------
def tokenize(s):
    s = s.lower()
    words = re.findall(r"[a-z0-9]+|[가-힣]+", s)
    toks = []
    for w in words:                       # 한국어는 조사 때문에 단어 그대로 매칭이 어려움 → 2-gram 추가
        toks.append(w)
        if re.match(r"[가-힣]", w) and len(w) > 2: toks += [w[i:i+2] for i in range(len(w) - 1)]
    return toks

class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.toks = [tokenize(d) for d in docs]; self.k1, self.b = k1, b
        self.avgdl = np.mean([len(t) for t in self.toks]); self.N = len(docs)
        df = Counter(w for t in self.toks for w in set(t))
        self.idf = {w: math.log(1 + (self.N - n + 0.5) / (n + 0.5)) for w, n in df.items()}
        self.tf = [Counter(t) for t in self.toks]
    def scores(self, q):
        qs = tokenize(q); out = np.zeros(self.N)
        for i, (tf, toks) in enumerate(zip(self.tf, self.toks)):
            for w in qs:
                if w in tf:
                    f = tf[w]; out[i] += self.idf[w] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * len(toks) / self.avgdl))
        return out

texts = [c["text"] for c in chunks]
bm25 = BM25(texts)

# ---------------- 3) 벡터 검색 (TF-IDF 문자 n-gram을 '임베딩' 대용으로) ----------------
# 실제 서비스에선 여기를 임베딩 모델(sentence-transformers 등) + 벡터DB로 교체
vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True)
E = vec.fit_transform(texts)
def dense_scores(q): return (E @ vec.transform([q]).T).toarray().ravel()   # 코사인 유사도 (L2 정규화돼 있음)

# ---------------- 4) 하이브리드 (RRF) ----------------
def rrf(rank_lists, k=60):
    s = Counter()
    for ranks in rank_lists:
        for r, idx in enumerate(ranks): s[idx] += 1 / (k + r + 1)
    return [i for i, _ in s.most_common()]

def search(q, method="hybrid", k=5, pool=None):
    b, d = np.argsort(-bm25.scores(q)), np.argsort(-dense_scores(q))
    if pool is not None:                                     # 허용된 청크만 (메타데이터 필터)
        b, d = [i for i in b if i in pool], [i for i in d if i in pool]
    ranked = {"bm25": b, "dense": d, "hybrid": rrf([b[:50], d[:50]])}[method]
    return [chunks[i] for i in ranked[:k]]

# ---------------- 5) 평가셋 & 지표 ----------------
EVAL = [  # (자연어 질문 - 노트의 표현과 일부러 다르게, 정답 청크에 반드시 있어야 할 문구)
    ("L1 규제를 쓰면 왜 어떤 가중치는 아예 사라져?", "특성 선택"),
    ("정답 비율이 한쪽으로 쏠린 데이터는 어떻게 다뤄?", "SMOTE"),
    ("여러 트리를 순서대로 학습시켜서 앞 모델의 실수를 고치는 방법", "잔차"),
    ("군집 개수를 미리 안 정해도 되는 군집화 알고리즘", "DBSCAN"),
    ("암 진단처럼 놓치면 큰일나는 문제에서 봐야 할 지표", "Recall"),
    ("ReLU 쓸 때 가중치 처음 값은 어떻게 잡아?", "He"),
    ("층을 100개 넘게 쌓아도 학습되게 만든 아이디어", "잔차 연결"),
    ("순환망이 오래 전 정보를 기억하게 하는 게이트 구조", "Forget"),
    ("내적 값을 차원의 제곱근으로 나누는 이유", "포화"),
    ("텍스트 조건을 얼마나 강하게 따를지 조절하는 생성 기법", "Classifier-Free"),
    ("GPU 한 장으로 큰 언어모델을 미세조정하는 방법", "QLoRA"),
    ("학습 코드에 버그가 있는지 빠르게 확인하는 요령", "과적합"),
    ("웹페이지에 숨겨진 지시문으로 에이전트를 조종하는 공격", "간접"),
    ("중앙 모델이 작업을 나눠서 하위 모델에게 맡기는 구조", "오케스트레이터"),
    ("키워드 검색과 의미 검색을 섞는 방법", "BM25"),
    ("AI 앱과 외부 도구를 잇는 표준 프로토콜", "MCP"),
    ("생각하고 행동하고 관찰하기를 반복하는 패턴", "ReAct"),
    ("에이전트용 도구 인터페이스를 잘 설계하는 법", "ACI"),
]
def evaluate(method, k=5, pool=None):
    hits, rr = 0, 0
    for q, gold in EVAL:
        res = search(q, method, k=20, pool=pool)
        ranks = [i for i, c in enumerate(res) if gold.lower() in c["text"].lower()]
        if ranks and ranks[0] < k: hits += 1
        if ranks: rr += 1 / (ranks[0] + 1)
    return hits / len(EVAL), rr / len(EVAL)

# 개선: '체크리스트' 섹션은 질문 문장과 비슷해서 검색을 방해함 → 메타데이터로 제외
no_checklist = {i for i, c in enumerate(chunks) if "체크리스트" not in c["section"]}
print(f"\n평가 질문 {len(EVAL)}개 (노트와 다른 표현으로 작성)")
print("검색 방법            | Recall@1 | Recall@3 | Recall@5 | MRR")
for pool_name, pool in [("전체 청크", None), ("체크리스트 제외", no_checklist)]:
    for m in ("bm25", "dense", "hybrid"):
        r1, mrr = evaluate(m, 1, pool); r3, _ = evaluate(m, 3, pool); r5, _ = evaluate(m, 5, pool)
        print(f"{m:7s}({pool_name:7s}) |  {r1:.3f}   |  {r3:.3f}   |  {r5:.3f}   | {mrr:.3f}")
for q, gold in EVAL:
    top = search(q, "hybrid", 5, no_checklist)
    if not any(gold.lower() in c["text"].lower() for c in top):
        print(f"  ✗ 실패 사례: '{q}' → 1위 {top[0]['doc']} §{top[0]['section']}")
print("→ 표현이 다르면(동의어·풀어쓴 말) 단어/문자 겹침 기반 검색은 놓친다 = 의미 임베딩 + 리랭커가 필요한 이유")
print("→ 하이브리드는 Recall@5에서 가장 안정적: 한쪽이 놓친 것을 다른 쪽이 보완")

# ---------------- 6) 최종 프롬프트 조립 ----------------
def build_rag_prompt(q, k=3):
    ctx = search(q, "hybrid", k, no_checklist)
    docs_xml = "\n".join(f'<doc id="{i+1}" source="{c["doc"]}">\n{c["text"][:400]}\n</doc>' for i, c in enumerate(ctx))
    return (f"아래 문서만 근거로 답하고 문장 끝에 [1]처럼 근거 번호를 다세요. 근거가 없으면 '노트에 없음'이라고 답하세요.\n"
            f"<documents>\n{docs_xml}\n</documents>\n\n질문: {q}")
q = "랜덤포레스트가 단일 결정트리보다 좋은 이유는?"
print(f"\n===== 질문: {q} =====")
for c in search(q, "hybrid", 3, no_checklist): print(f"  - {c['doc']}  §{c['section']}")
print("\n===== LLM에 보낼 프롬프트(앞부분) =====\n" + build_rag_prompt(q)[:700] + "\n...")
