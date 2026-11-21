# RAG (Retrieval-Augmented Generation, 검색 증강 생성)

> LLM이 모르는 지식(사내 문서, 최신 정보)을 **검색해서 프롬프트에 넣어** 답하게 하는 방법

## 1. 왜 RAG?
- 지식 컷오프 / 비공개 데이터 문제 해결
- 환각 감소, **출처 인용** 가능
- 파인튜닝보다 저렴하고 지식 업데이트가 쉬움

## 2. 기본 파이프라인
```
[인덱싱 - 오프라인]
문서 수집 → 파싱(PDF/HTML) → 청킹(Chunking) → 임베딩 → 벡터 DB 저장

[질의 - 온라인]
질문 → (질의 변환) → 임베딩 → 유사 청크 검색(top-k) → (리랭킹)
     → 프롬프트에 삽입 → LLM 답변 + 출처
```

## 3. 청킹 (Chunking)
| 방법 | 설명 |
|---|---|
| 고정 길이 | N 토큰 + 겹침(overlap) 10~20% |
| 재귀적 분할 | 문단 → 문장 → 단어 순으로 자연스럽게 |
| 구조 기반 | 마크다운 헤더, 코드 함수 단위 |
| 의미 기반 | 임베딩 유사도가 바뀌는 지점에서 분할 |

- 너무 작으면 문맥 부족, 너무 크면 검색 정밀도↓ → 보통 200~1000 토큰에서 실험
- **Contextual chunking**: 청크마다 "이 청크는 어떤 문서의 어떤 부분인지" 짧은 설명을 붙여 검색 정확도 향상
- 메타데이터(출처, 날짜, 섹션) 함께 저장 → 필터링

## 4. 임베딩 & 벡터 검색
- 임베딩 모델: 텍스트 → 고정 차원 벡터 (의미가 비슷하면 가까움)
- 유사도: 코사인 유사도, 내적
- 근사 최근접 탐색(ANN): HNSW, IVF
- 벡터 DB: FAISS(라이브러리), Chroma, Qdrant, Weaviate, Milvus, pgvector(Postgres), Elasticsearch/OpenSearch
- 한국어 문서라면 **다국어/한국어 성능이 좋은 임베딩 모델**인지 확인

## 5. 검색 품질 높이기
| 기법 | 설명 |
|---|---|
| **하이브리드 검색** | 벡터(의미) + BM25(키워드) 결합 → 고유명사·코드·숫자에 강함 |
| **리랭킹 (Reranking)** | 넓게 검색(top-50) 후 Cross-encoder로 정밀 재정렬(top-5) |
| 질의 재작성 | 대화 맥락 반영, 여러 질의로 확장(Multi-query) |
| HyDE | 가상의 답변을 생성해 그걸로 검색 |
| 메타데이터 필터 | 날짜·부서·문서 유형으로 제한 |
| 부모-자식 청크 | 작은 청크로 검색, 큰 청크를 LLM에 전달 |

## 6. 발전된 RAG
- **Agentic RAG**: 에이전트가 검색 여부·질의·횟수를 스스로 결정, 결과가 부족하면 재검색
- **GraphRAG**: 엔티티-관계 지식 그래프 구축 → 전체 요약·다중 홉 질문에 강함
- 긴 컨텍스트 모델: 문서가 작으면 RAG 없이 전부 넣는 것이 더 나을 수도 있음 (+ 프롬프트 캐싱)

## 7. RAG 평가
| 구분 | 지표 |
|---|---|
| 검색 | Recall@k, Precision@k, MRR, nDCG |
| 생성 | 충실성(Faithfulness: 근거에 기반했는가), 답변 관련성, 정답성 |
| 도구 | RAGAS 등 평가 프레임워크, LLM-as-judge |

먼저 **검색이 문제인지 생성이 문제인지** 분리해서 진단!

## 8. 최소 구현
```python
from sentence_transformers import SentenceTransformer
import numpy as np

emb = SentenceTransformer("<multilingual-embedding-model>")
chunks = split_into_chunks(documents)              # 청킹
V = emb.encode(chunks, normalize_embeddings=True)  # 인덱싱

def retrieve(q, k=5):
    qv = emb.encode([q], normalize_embeddings=True)[0]
    idx = np.argsort(V @ qv)[::-1][:k]
    return [chunks[i] for i in idx]

def answer(q):
    ctx = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(retrieve(q)))
    prompt = f"""다음 문서만 근거로 답하고, 근거 번호를 [n] 형식으로 인용하세요.
근거가 없으면 '문서에서 찾을 수 없습니다'라고 답하세요.
<documents>
{ctx}
</documents>
질문: {q}"""
    return llm(prompt)
```

---

## 🧪 실습

**코드** · [`ag04_rag_pipeline.py`](code/ag04_rag_pipeline.py)  
**실행 로그** · [`ag04_rag_pipeline.txt`](code/outputs/ag04_rag_pipeline.txt)

### 📝 실습 후 정리

- 내 노트 25개를 274개 청크로 나누고 BM25를 직접 구현. 노트와 다르게 표현한 질문 18개로 평가셋을 만듦.
- Recall@5: BM25 0.72, 문자 n-gram 벡터 0.78, 하이브리드(RRF) 0.83 → 서로 놓친 것을 보완.
- 실패 사례는 모두 '같은 뜻 다른 표현' → 진짜 임베딩 모델과 리랭커가 필요한 이유를 데이터로 확인.
