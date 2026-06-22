# RAG Architecture Project

## 1. RAG 구현

**Document Loading 단계 입출력 명세**
| 항목 | 정의 |
| --- | --- |
| Input (입력) | 파일 경로(file path) — 문자열 |
| Output (출력) | Raw text — 단일 문자열(string) |
| 책임(Responsibility) | 파일을 읽어서 텍스트로 변환하는 것까지만. Chunking(분할)은 다음 단계의 책임이므로 여기서 하지 않음 |

**Fixed-size Chunking 단계 입출력 명세**
| 항목 | 정의 |
| --- | --- |
| Input (입력) | Raw text — 단일 문자열 |
| Output (출력) | Chunk 리스트 — `list[str]` |
| 핵심 파라미터 | `chunk_size` (chunk당 문자 수), `chunk_overlap` (인접 chunk 간 겹치는 문자 수) |
| 책임(Responsibility) | 텍스트를 정해진 길이로 자르는 것까지만. Embedding은 다음 단계의 책임 |

**Section-based Chunking 단계 입출력 명세**
| 항목 | 정의 |
| --- | --- |
| Input (입력) | Raw text — Document Loading 단계의 출력 |
| Output (출력) | Chunk 리스트 — `list[str]` (기존 `chunk_fixed_size`와 동일한 반환 타입) |
| 핵심 전략 | `##` 헤더로 섹션 분리 → 섹션이 `chunk_size`보다 길면 그 섹션 내부에서만 fixed-size로 재분할 |

- 하이브리드 방식을 적용하여 "섹션 경계는 절대 넘지 않되, 섹션이 너무 길면 그 안에서 추가로 자른다."
- 이렇게 하면 트러블슈팅 #8에서 발견한 문제(서로 다른 주제가 한 chunk에 섞이는 것)을 원천적으로 방지하면서, 섹션이 비정상적으로 길어서 임베딩 품질이 떨어지는 경우 대비할 수 있다.
- 구현 순서:
  1. 정규표현식으로 `## ` 패턴을 찾아 텍스트를 섹션 단위로 분리
  2. 각 섹션 `chunk_size` 이하면 그대로 1개 chunk
  3. `chunk_size`를 초과하면, 기존 `chunk_fixed_size()`를 그 섹션 텍스트에 재사용하여 내부적으로 잘게 나눔

**Embedding 단계 입출력 명세**
| 항목 | 정의 |
| --- | --- |
| Input (입력) | Chunk 리스트 — `list[str]` (Chunking 단계의 출력) |
| Output (출력) | Embedding 벡터 배열 — `numpy.ndarray`, shape `(chunk 개수, 384)` |
| 책임(Responsibility) | 텍스트를 벡터로 변환하는 것까지만. 저장(Storage)이나 검색(Retrieval)은 다음 단계의 책임 |

- 언어: 문서를 영어로 쓰기로 했으므로, 다국어 모델(paraphrase-multilingual 계열)은 불필요한 오버헤드임. 영어 전용 모델이 적합함.
- 학습 목적: 지금 단계는 "RAG 구조를 이해하는 것"이 목적이므로, 구조 학습용으로 `all-MiniLM-L6-v2`가 가장 널리 쓰이는 표준적인 선택임
- 주의사항: `all-MiniLM-L6-v2`는 최대 256 word piece까지만 처리하고, 그보다 긴 텍스트는 잘려서 임베딩된다는 제약이 존재함. Chunking 단계를 이미 거쳤으므로 chunk_size를 300자로 잘라뒀으므로, 이 제약에 걸릴 위험이 낮음.(**embedding 전에 chunking을 먼저하는 실질적인 이유**)

**Storage 방식 단계**
| 단계 | 명칭 | 핵심 특징 |
| --- | --- | --- |
| 0 | In-Memory Storage (메모리 보관) | 변수에 numpy 배열 그대로 유지, 영속성 없음 |
| 1 | File-based Storage (파일 기반 저장) | `.npy` 등으로 디스크에 저장, 영속성 확보 |
| 2 | Vector DB (벡터 데이터베이스) | FAISS/Chroma 등 전용 인덱싱 구조 사용, 대규모 검색 최적화 |
| 3 | Graph DB / Graph RAG | 문서 간 관계(엔티티, 관계)를 그래프로 표현, 구조적 추론 가능 |

**Storage 단계 간 핵심 차이**
| 전환 | 새로 추가되는 능력 | 해결하는 문제 |
| --- | --- | --- |
| 0 → 1 | 영속성(persistence) | 프로그램 재시작 시 재계산 불필요 |
| 1 → 2 | 효율적 유사도 검색(efficient similarity search) | chunk 수가 많아지면(수만 개 이상) brute-force 비교가 느려짐 |
| 2 → 3 | 관계 기반 추론(relational reasoning) | "A와 B가 어떻게 연결되는가" 같은 다중 홉(multi-hop) 질문에 대응 |

- 메모리 보관 단계에서 Graph DB 단계로 확장 예정
- 의존성(dependency) 관점 설명
  1. 메모리 보관(현재): chunk와 벡터를 그냥 Python 변수(리스트, numpy 배열)로 들고 있는 상태. 지금 목적(RAG 구조 이해)에는 충분함. — Indexing과 Query를 같은 실행(run) 안에서 처리하기 때문.
  2. 왜 다음이 "파일 저장"인가: 지금처럼 매번 스크립트를 실행할 때마다 16개 chunk를 다시 embedding하는 건 비효율(시간 낭비). **"한 번 계산한 걸 디스크에 저장해두고, 다음 실행에서는 다시 계산하지 않고 불러오기만 한다"**는 게 이 단계의 핵심 가치. 이건 FAISS 같은 라이브러리를 쓰든 안 쓰든 항상 필요한 개념이라서, VectorDB로 바로 건너뛰기 전에 "저장 자체"의 개념을 한 번 짚는 게 학습상 유리.
  3. 왜 그다음이 VectorDB인가: chunk가 16개일 때는 사용자 질문 벡터와 16개 벡터를 모두 비교(brute-force)해도 거의 즉시 끝남. 하지만 문서가 많아져서 chunk가 수만~수백만 개가 되면, 전체를 다 비교하는 방식은 너무 느려짐. FAISS 같은 VectorDB는 "정확도를 약간 희생하더라도 훨씬 빠르게 근사치를 찾는" 인덱싱 알고리즘(예: HNSW, IVF)을 제공함. 즉, **VectorDB의 본질은 "대규모 검색을 빠르게 하기 위한 자료구조"**.
  4. 왜 마지막이 Graph RAG인가: VectorDB는 "이 질문과 의미적으로 가장 가까운 chunk들"을 찾는 데는 강하지만, "제품 A의 에러코드와 제품 A의 설정값이 어떻게 연결되어 있는가" 같은 개체(entity) 간 명시적 관계를 다루는 데는 약함. Graph RAG는 문서에서 엔티티와 관계를 추출해 그래프로 만들고, 검색 시 그래프 구조를 함께 활용함. 이건 "단순 유사도 검색"에서 "구조화된 관계 추론"으로 능력이 한 단계 더 올라가는 것이라, VectorDB 다음에 오는 게 자연스러움.

**Storage 단계 입출력 명세**
| 항목 | 정의 |
| --- | --- |
| Input (입력) | Chunk 리스트(`list[str]`) + 벡터 배열(`numpy.ndarray`) |
| Output (출력) | 검색 가능한 저장소 객체 — chunk와 벡터를 1:1로 매핑하여 보관 |
| 책임(Responsibility) | chunk와 벡터를 짝지어 보관하고, 인덱스(index)로 원래 텍스트를 다시 찾을 수 있게 하는 것까지만. 유사도 계산/검색 로직(Retrieval)은 다음 단계의 책임 |

**Retrieval 단계 입출력 명세**
| 항목 | 정의 |
| --- | --- |
| Input (입력) | `query_vector` (shape `(384,)`) + `store` (`InMemoryVectorStore` 인스턴스) + `k` (정수) |
| Output (출력) | `(chunk 텍스트, 유사도 점수)` 튜플의 리스트, 길이 `k`, 점수 내림차순 정렬 |
| 책임(Responsibility) | 유사도 계산 + 순위화 + Top-k 추출 + 텍스트 복원까지만. Prompt 조립이나 생성은 다음 단계의 책임 |

**Prompt Augmentation 단계 입출력 명세**
| 항목 | 정의 |
| --- | --- |
| Input (입력) | 사용자 질문(string) + Retrieval 결과(`list[tuple[str, float]]`) |
| Output (출력) | LLM에 전달할 완성된 prompt(string) |
| 책임(Responsibility) | 검색된 chunk들을 질문과 함께 LLM이 이해할 수 있는 하나의 텍스트로 조립하는 것까지만. LLM 호출(Generation)은 다음 단계의 책임 |

- 중요한 설계 포인트: "모르면 모른다고 답하라"고 명시적으로 지시하기 (환각 억제)

## 2. FastAPI Wrapping

**FastAPI 앱 설계 명세**
| 항목 | 정의 |
| --- | --- |
| 엔드포인트 | `POST /query` |
| 요청 Body | `{"question": str, "k": int (선택, 기본값 3)}` |
| 응답 Body | `{"answer": str, "retrieved_chunks": list[{"text": str, "score": float}]}` |
| Indexing 시점 | FastAPI의 lifespan 이벤트로 서버 시작 시 1회 |

## 3. RAGAS 평가

**RAG 평가 기능 확장 로드맵**
| Level | 단계 이름 | 이 단계에서 하는 일 | 이전 단계의 한계점 | 확장 시 추가되는 가치 | 난이도 |
| --- | --- | --- | --- | --- | --- |
| **Level 0** | **최소 평가 스크립트 (전체 지표)** | Faithfulness, Context Precision, Answer Relevancy, Context Recall을 **가장 단순한 형태**로 모두 구현 | - | 모든 핵심 지표의 baseline을 빠르게 확보 | 낮음 |
| **Level 1** | LLM 기반 고도화 | 주요 지표를 **LLM-as-a-Judge** 방식으로 구현 | 의미 기반만으로는 한계가 있음 | 더 정확하고 유연한 평가 가능 | 중상 |
| **Level 2** | RAGAS 기본 연동 | RAGAS 라이브러리 도입 및 연동 | 직접 구현의 유지보수 비용 | 표준화된 평가 + 편의성 확보 | 중상 |
| **Level 3** | RAGAS 고도화 + 운영 | 전체 지표 통합 + 자동화 | 일회성 평가에 머무름 | 실 서비스 운영 수준의 평가 체계 구축 | 상 |

**Level 0: Faithfulness 명세**
| 항목 | 정의 |
| --- | --- |
| **Task Unit 이름** | Level 0: RAGAS 없는 최소 Faithfulness 평가 |
| **목표** | RAGAS 라이브러리 없이, 기존 TextGenerator만 사용하여 생성된 답변의 **Faithfulness(충실도)**를 가장 기본적인 형태로 측정 |
| **평가 지표** | Faithfulness (생성된 답변이 검색된 context에 근거하는가) |
| **평가 로직 형태** | LLM에게 각 주장을 Yes/No + 간단한 이유로 판단하게 함 (가장 단순한 형태) |
| **입력 (Input)** | RAG 파이프라인의 출력 결과 (question, answer, retrieved_chunks) |
| **출력 (Output)** | 각 주장에 대한 판단 결과 + 전체 Faithfulness 점수 (콘솔 출력) |
| **책임 (Responsibility)** | 평가 로직만 담당. RAG 파이프라인을 호출하거나 재구현하지 않음 |
| **LLM 사용** | 사용하지 않음 |
| **의존성** | 추가 의존성 없음 |
| **스크립트 위치** | debugs/evaluate_faithfulness.py |
| **평가 데이터** | debug_retrieval.py에서 이미 검증된 4개 질문 재사용 |

**Level 0: Context Precision 명세**
| 항목 | 내용 |
| --- | --- |
| **목표** | 검색된 context 중 질문과 관련 있는 context의 비율을 가장 단순한 방식으로 계산 |
| **구현 방식** | 키워드(단어) 겹침 기반 판단 |
| **Input** | `question` (str), `retrieved_chunks` (list[str]) |
| **Output** | `context_precision_score` (float, 0.0 ~ 1.0), 각 chunk에 대한 관련성 판단 결과 |
| **책임** | 검색 품질 평가 담당 |
| **LLM 사용** | 사용하지 않음 |
| **스크립트 위치** | `debugs/evaluate_context_precision.py` |
| **평가 데이터** | 기존에 사용하던 질문 세트 재사용 |

**Level 0: Answer Relevancy 명세**
| 항목 | 내용 |
| --- | --- |
| **지표** | **Answer Relevancy** |
| **평가 대상** | 생성 품질 (답변이 질문과 관련 있는가?) |
| **구현 방식** | 가장 단순한 키워드 겹침 기반 |
| **Input** | `question` (str), `answer` (str) |
| **Output** | `answer_relevancy_score` (float) + 각 판단 근거 |
| **책임** | 생성된 답변과 질문 간의 관련성 평가 담당 |
| **LLM 사용** | 사용하지 않음 |
| **Threshold** | 0.3 |
| **스크립트 위치** | `debugs/evaluate_answer_relevancy.py` |

**Level 0: Context Recall 명세**
| 항목 | 내용 |
| --- | --- |
| **지표** | **Context Recall** |
| **평가 대상** | 검색 품질 (정답에 필요한 정보가 검색됐는가?) |
| **구현 방식** | 가장 단순한 키워드 기반 (Ground Truth 필요) |
| **Input** | `question`, `retrieved_chunks`, `ground_truth` (핵심 정보 목록) |
| **Output** | `context_recall_score` (float), 각 핵심 정보의 포함 여부 |
| **책임** | 정답에 필요한 정보가 검색됐는지를 담당 |
| **LLM 사용** | 사용하지 않음 |
| **스크립트 위치** | `debugs/evaluate_context_recall.py` |
| **평가 데이터** | 기존 질문에 Ground Truth(핵심 정보)를 추가하여 사용 |
