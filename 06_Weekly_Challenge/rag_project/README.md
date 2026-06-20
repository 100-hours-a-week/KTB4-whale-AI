# RAG Architecture Project

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

**Embedding 단계 입출력 명세**
| 항목 | 정의 |
| --- | --- |
| Input (입력) | Chunk 리스트 — `list[str]` (Chunking 단계의 출력) |
| Output (출력) | Embedding 벡터 배열 — `numpy.ndarray`, shape `(chunk 개수, 384)` |
| 책임(Responsibility) | 텍스트를 벡터로 변환하는 것까지만. 저장(Storage)이나 검색(Retrieval)은 다음 단계의 책임 |

- 언어: 문서를 영어로 쓰기로 했으므로, 다국어 모델(paraphrase-multilingual 계열)은 불필요한 오버헤드임. 영어 전용 모델이 적합함.
- 학습 목적: 지금 단계는 "RAG 구조를 이해하는 것"이 목적이므로, 구조 학습용으로 `all-MiniLM-L6-v2`가 가장 널리 쓰이는 표준적인 선택임
- 주의사항: `all-MiniLM-L6-v2`는 최대 256 word piece까지만 처리하고, 그보다 긴 텍스트는 잘려서 임베딩된다는 제약이 존재함. Chunking 단계를 이미 거쳤으므로 chunk_size를 300자로 잘라뒀으므로, 이 제약에 걸릴 위험이 낮음.(**embedding 전에 chunking을 먼저하는 실질적인 이유**)
